"""Strict full-model initialization, separate from resuming this experiment."""
import argparse
import logging
import random
import numpy as np
import torch
import torch.distributed as dist

ARCH_FIELDS = ('arch', 'patch_size', 'out_dim', 'patch_out_dim', 'shared_head',
               'shared_head_teacher', 'norm_in_head', 'act_in_head', 'norm_last_layer')

def trusted_load(path):
    # Official full checkpoints contain argparse.Namespace, not only tensors.
    return torch.load(path, map_location='cpu', weights_only=False)

def bare_state(state):
    return {k.removeprefix('module.'): v for k, v in state.items()}

def infer_pretrained_architecture(states):
    """Recover the standard, unnormalized GELU ViT-B/16 head layout."""
    architecture = dict(arch='vit_base', patch_size=16, norm_in_head=None,
                        act_in_head='gelu', norm_last_layer=True)
    dimensions = []
    for role, state in states.items():
        patch_embed = state.get('backbone.patch_embed.proj.weight')
        if patch_embed is None or tuple(patch_embed.shape) != (768, 3, 16, 16):
            raise ValueError(f'{role}: checkpoint without args must contain ViT-B/16 patch embedding weights')
        cls = state.get('head.last_layer.weight_v')
        patch = state.get('head.last_layer2.weight_v')
        if cls is None or patch is None or cls.ndim != 2 or patch.ndim != 2:
            raise ValueError(f'{role}: checkpoint without args must contain both iBOT head projections')
        dimensions.append((cls.shape[0], patch.shape[0]))
        # torch.save preserves aliases for a shared CLS/patch projection.
        shared = cls.shape == patch.shape and cls.data_ptr() == patch.data_ptr()
        architecture['shared_head' if role == 'student' else 'shared_head_teacher'] = shared
    if dimensions[0] != dimensions[1]:
        raise ValueError('Student and teacher projection dimensions differ')
    architecture['out_dim'], architecture['patch_out_dim'] = dimensions[0]
    return architecture

def read_initial_checkpoint(args):
    path = args.resume or args.pretrained
    checkpoint = trusted_load(path)
    if 'student' not in checkpoint or 'teacher' not in checkpoint:
        raise ValueError('A Full Checkpoint with student AND teacher is required, not backbone-only weights')
    old = checkpoint.get('args')
    old = vars(old) if isinstance(old, argparse.Namespace) else old
    states = {role: bare_state(checkpoint[role]) for role in ('student', 'teacher')}
    for role, state in states.items():
        if not any(k.startswith('backbone.') for k in state) or not any(k.startswith('head.') for k in state):
            raise ValueError('%s is missing backbone or projection head' % role)
    if old is None and not args.resume:
        old = infer_pretrained_architecture(states)
        print(f'Checkpoint has no args; using inferred iBOT architecture: {old}', flush=True)
    if not isinstance(old, dict):
        raise ValueError('Checkpoint architecture args are invalid or missing for --resume')
    missing = [field for field in ARCH_FIELDS if field not in old]
    if missing:
        raise ValueError('Checkpoint is missing architecture metadata: %s' % missing)
    for field in ARCH_FIELDS:
        setattr(args, field, old[field])
    if args.arch != 'vit_base' or args.patch_size != 16:
        raise ValueError('This experiment requires official ViT-B/16 weights')
    return checkpoint

def load_models(checkpoint, student, teacher, initialize_decoder=False, initialize_identity=False):
    student_state = bare_state(checkpoint['student'])
    if initialize_decoder:
        # Official iBOT has no decoder. All original parameters still load strictly.
        for name, value in student.decoder.state_dict().items():
            student_state['decoder.' + name] = value
    if initialize_identity:
        if any(k.startswith(('decoder.', 'arcface.')) for k in student_state):
            raise ValueError('New identity experiment requires original iBOT weights, not a decoder/identity checkpoint')
        if checkpoint.get('experiment', '').startswith('progressive-crfrp'):
            raise ValueError('Start identity training from original iBOT weights, not stage1 output')
        for name, value in student.arcface.state_dict().items():
            student_state['arcface.' + name] = value
    student.load_state_dict(student_state, strict=True)
    teacher.load_state_dict(bare_state(checkpoint['teacher']), strict=True)

def load_centers(checkpoint, loss):
    state = checkpoint.get('ibot_loss', {})
    for name in ('center', 'center2'):
        if name in state:
            target = getattr(loss, name)
            if target.shape != state[name].shape:
                raise ValueError('Pretrained %s has incompatible shape' % name)
            target.copy_(state[name].to(target))

def capture_rng():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states

def restore_experiment(checkpoint, args, optimizer, scaler, loss):
    identity = args.identity_weight > 0
    expected = 'ibot-identity-v1' if identity else 'progressive-crfrp-v1'
    if checkpoint.get('experiment') != expected:
        raise ValueError('--resume only accepts a checkpoint from this experiment')
    saved_args = checkpoint['args']
    saved_args = vars(saved_args) if isinstance(saved_args, argparse.Namespace) else saved_args
    if identity:
        for key in ('mask_mode', 'identity_weight', 'arcface_margin', 'arcface_scale',
                    'num_classes', 'num_images', 'recordio_signature', 'blendface_masks'):
            if saved_args.get(key) != getattr(args, key):
                raise ValueError('Resume configuration differs: %s' % key)
    if saved_args.get('visible_decoder', False) != getattr(args, 'visible_decoder', False):
        raise ValueError('Resume configuration differs: visible_decoder')
    if getattr(args, 'visible_decoder', False):
        for key in ('decoder_depth', 'decoder_width'):
            if saved_args.get(key) != getattr(args, key):
                raise ValueError('Resume configuration differs: %s' % key)
    # Keep the experiment settings fixed when rebuilding schedules for the current loader.
    for key in ('stages', 'epochs', 'batch_size_per_gpu', 'lr', 'min_lr', 'warmup_epochs',
                'momentum_teacher', 'weight_decay', 'weight_decay_end', 'global_crops_scale',
                'local_crops_scale', 'global_crops_number', 'local_crops_number', 'seed',
                'data_path', 'require_full_organs', 'optimizer',
                'teacher_temp', 'teacher_patch_temp', 'warmup_teacher_temp_epochs', 'use_fp16'):
        if saved_args.get(key) != getattr(args, key):
            raise ValueError('Resume configuration differs: %s' % key)
    if not identity and saved_args.get('parsing_root') != args.parsing_root:
        raise ValueError('Resume configuration differs: parsing_root')
    states = checkpoint['rng_states']
    start_epoch = checkpoint['epoch']
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if rank == 0 and len(states) != world_size:
        logging.getLogger('ibot.training').info(
            'Resuming at epoch %d with %d GPU(s), previously %d. '
            'Global batch size changes from %d to %d and schedules use the current loader.',
            start_epoch + 1, world_size, len(states), args.batch_size_per_gpu * len(states),
            args.batch_size_per_gpu * world_size)
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss.load_state_dict(checkpoint['ibot_loss'], strict=True)
    if scaler is not None:
        scaler.load_state_dict(checkpoint['fp16_scaler'])
    if rank < len(states):
        state = states[rank]
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'])
    else:
        # A newly added rank has no saved RNG state of its own.
        seed = int(np.random.SeedSequence([args.seed, start_epoch, rank]).generate_state(1)[0])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return start_epoch
