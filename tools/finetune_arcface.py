"""Fine-tune iBOT Student on Fake and iBOT Teacher on Real for identity retrieval."""

import _bootstrap
import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import time
import logging
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from tqdm import tqdm

from progressive.data import EXTENSIONS, MEAN, STD
from progressive.checkpoints import trusted_load
from progressive.evaluation import RetrievalValidation, encoder_from_checkpoint
from progressive.recordio import open_recordio, read_record
from progressive.patch_alignment import (LocalPatchAlignment, projection_from_checkpoint,
                                         learning_rate_factor)


def save_checkpoint_atomically(state, path):
    """Return success; never replace an existing checkpoint with a partial file.

    An explicit file object gives Python control of write/flush errors. If the
    default zip writer fails, retry once with PyTorch's legacy serializer, which
    torch.load also supports. Persistent storage errors are logged, not hidden.
    """
    path = Path(path)
    for zip_format in (True, False):
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                                dir=path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, 'wb') as handle:
                torch.save(state, handle, _use_new_zipfile_serialization=zip_format)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return True
        except (OSError, RuntimeError):
            logging.exception('Checkpoint write failed: path=%s, format=%s%s',
                              path.resolve(), 'zip' if zip_format else 'legacy',
                              '; retrying with legacy serialization' if zip_format else
                              '; previous checkpoint preserved, this update was NOT saved')
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logging.exception('Could not remove incomplete checkpoint: %s', temporary)
    return False


def image_transform(random_flip=True):
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        *([T.RandomHorizontalFlip()] if random_flip else []),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class TargetIdentityDataset(Dataset):
    def __init__(self, manifest, root, real_recordio):
        root = Path(root)
        self.rows = []
        if manifest:
            data = json.loads(Path(manifest).read_text(encoding='utf-8'))
            for row in data['queries']:
                if 'target_id' in row:
                    self.rows.append((root / row['path'], str(row['target_id'])))
        else:
            if not root.is_dir():
                raise FileNotFoundError(root)
            pattern = re.compile(r'(?:id|n)?(\d+)_(?:id|n)?(\d+)', re.IGNORECASE)
            for folder in sorted(root.iterdir()):
                match = pattern.fullmatch(folder.name)
                if folder.is_dir() and match:
                    target = str(int(match[1]))
                    self.rows.extend((path, target) for path in sorted(folder.iterdir())
                                     if path.is_file() and path.suffix.lower() in EXTENSIONS)
            if not self.rows:
                raise ValueError('No Fake images found in target_source folders; check data_root or supply a manifest')
        self.classes = sorted({identity for _, identity in self.rows})
        if len(self.classes) < 2:
            raise ValueError('Training manifest needs fake queries from at least two target identities')
        self.class_index = {identity: index for index, identity in enumerate(self.classes)}
        self.transform = image_transform()
        self.real_root = Path(real_recordio)
        with (self.real_root / 'idx2label.json').open('rb') as file:
            labels = pickle.load(file)
        real_indices = defaultdict(list)
        for index, identity in labels.items():
            cls = self.class_index.get(str(int(identity)))
            if cls is not None:
                real_indices[cls].append(int(index))
        missing = [identity for identity, cls in self.class_index.items() if cls not in real_indices]
        if missing:
            raise ValueError(f'{len(missing)} Fake Targets have no MS1Mv3 Real records; example: {missing[0]}')
        self.real_indices = dict(real_indices)
        self.reader = None
        self.reader_pid = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, identity = self.rows[index]
        with Image.open(path) as image:
            fake = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        if self.reader_pid != os.getpid():
            self.reader = open_recordio(self.real_root)
            self.reader_pid = os.getpid()
        cls = self.class_index[identity]
        label, real = read_record(self.reader, random.choice(self.real_indices[cls]))
        if str(label) != identity:
            raise ValueError(f'Real RecordIO label does not match Fake Target {identity}: {path}')
        return fake, self.transform(real), cls


def training_data(args):
    if args.online_masks:
        from progressive.stage2_data import OnlineTargets
        return OnlineTargets(args.data_root, args.online_masks, image_transform(random_flip=False))
    return TargetIdentityDataset(args.manifest, args.data_root, args.real_recordio)


class ArcFace(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=.5, scale=64.):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, features, labels, weight=None):
        if weight is None:
            weight = F.normalize(self.weight.float())
        cosine = F.linear(F.normalize(features.float()), weight)
        cosine = cosine.clamp(-1., 1.)
        labels = labels[:, None]
        target = cosine.gather(1, labels)
        sine = (1. - target.square()).clamp_min(1e-7).sqrt()
        adjusted = target * math.cos(self.margin) - sine * math.sin(self.margin)
        adjusted = torch.where(target > math.cos(math.pi - self.margin), adjusted,
                               target - self.margin * math.sin(math.pi - self.margin))
        return cosine.scatter(1, labels, adjusted) * self.scale


class Stage2(nn.Module):
    def __init__(self, student, real_encoder, head, use_bf16=False,
                 student_patch_head=None, teacher_patch_head=None, patch_alignment=None):
        super().__init__()
        self.student = student
        self.real_encoder = real_encoder
        self.head = head
        self.use_bf16 = use_bf16
        self.student_patch_head = student_patch_head
        self.teacher_patch_head = teacher_patch_head
        self.patch_alignment = patch_alignment

    def forward(self, fake, real, labels):
        with torch.autocast(device_type=fake.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            fake_tokens = self.student(fake)
            real_tokens = self.real_encoder(real)
        fake_cls, real_cls = fake_tokens[:, 0].float(), real_tokens[:, 0].float()
        weight = F.normalize(self.head.weight.float())
        fake_loss = F.cross_entropy(self.head(fake_cls, labels, weight), labels)
        real_loss = F.cross_entropy(self.head(real_cls, labels, weight), labels)
        alignment = 1 - F.cosine_similarity(fake_cls, real_cls.detach()).mean()
        patch_loss = fake_cls.new_zeros(())
        if self.patch_alignment is not None:
            student_patch = self.student_patch_head(fake_tokens[:, 1:].float())
            with torch.no_grad():
                teacher_patch = self.teacher_patch_head(real_tokens[:, 1:].detach().float())
            patch_loss = self.patch_alignment(student_patch, teacher_patch)
        return fake_loss, real_loss, alignment, patch_loss


def device_batches(iterator, device, prefetch):
    if not prefetch:
        for batch in iterator:
            yield tuple(tensor.to(device) for tensor in batch)
        return

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)

    def load_next():
        batch = next(iterator, None)
        if batch is not None:
            # Pin here, after the loader workers have forked and CUDA is ready.
            with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                return tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in batch)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(load_next)
        while (batch := pending.result()) is not None:
            compute_stream.wait_stream(copy_stream)
            for tensor in batch:
                tensor.record_stream(compute_stream)
            pending = executor.submit(load_next)
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest')
    parser.add_argument('--data_root')
    parser.add_argument('--online_masks', help='BlendFace masks_packed; synthesize online from data_root RecordIO')
    parser.add_argument('--output')
    parser.add_argument('--real_recordio', help='MS1Mv3 Real RecordIO used for offline Fake training')
    parser.add_argument('--validation_config', help='YAML with gallery, test and test_ori image settings')
    parser.add_argument('--alignment_weight', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fp32', action='store_true', help='Disable BF16 encoder autocast')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers per GPU')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--student_lr', type=float, help='Peak encoder LR; defaults to lr')
    parser.add_argument('--head_lr', type=float, help='Peak ArcFace LR; defaults to lr')
    parser.add_argument('--patch_head_lr', type=float, help='Peak patch projection LR; defaults to lr')
    parser.add_argument('--real_lr', type=float, help='Peak Real encoder LR; defaults to lr')
    parser.add_argument('--warmup_ratio', type=float, default=0.)
    parser.add_argument('--min_lr_ratio', type=float, default=1.)
    parser.add_argument('--patch_alignment_weight', type=float, default=0.)
    parser.add_argument('--patch_temperature', type=float, default=.1)
    parser.add_argument('--patch_distance_weight', type=float, default=1.)
    parser.add_argument('--patch_neighborhood_size', type=int, default=3)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config:
        defaults = json.loads(Path(args.config).read_text(encoding='utf-8'))
        unknown = defaults.keys() - vars(args).keys()
        if unknown:
            parser.error(f'Unknown config keys: {sorted(unknown)}')
        parser.set_defaults(**defaults)
        args = parser.parse_args()
    if not all((args.checkpoint, args.data_root, args.output)):
        parser.error('checkpoint, data_root and output are required')
    output = Path(args.output)
    for name in ('student_lr', 'head_lr', 'patch_head_lr', 'real_lr'):
        value = getattr(args, name)
        setattr(args, name, args.lr if value is None else value)
    if (any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
            for name in ('student_lr', 'head_lr', 'patch_head_lr', 'real_lr'))
            or not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1
            or not math.isfinite(args.patch_alignment_weight) or args.patch_alignment_weight < 0
            or not math.isfinite(args.patch_temperature) or args.patch_temperature <= 0
            or not math.isfinite(args.patch_distance_weight) or args.patch_distance_weight < 0
            or args.patch_neighborhood_size < 1 or args.patch_neighborhood_size % 2 != 1):
        parser.error('Invalid patch alignment or learning rate settings')
    if args.patch_alignment_weight > 0 and not args.online_masks:
        parser.error('Patch alignment requires online paired synthesis; offline Real images are not spatial pairs')
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.alignment_weight < 0 or args.verbose < 1 or args.num_workers < 0):
        parser.error('Invalid training settings')
    if not args.online_masks and not args.real_recordio:
        parser.error('Set real_recordio to the MS1Mv3 RecordIO directory for offline Fake training')
    if args.online_masks and args.manifest:
        parser.error('Select either online synthesis or an offline fake manifest')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = world_size > 1
    is_main = rank == 0
    validation = None
    if args.validation_config and not Path(args.validation_config).is_file():
        parser.error(f'Validation config not found: {args.validation_config}')
    if is_main:
        run_dir = output.parent / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir.mkdir(parents=True, exist_ok=False)
        output = run_dir / output.name
        best_output = output.with_name(f'{output.stem}_best{output.suffix}')
        logging.basicConfig(level=logging.INFO, format='Training: %(asctime)s-%(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(output.with_suffix('.log'), encoding='utf-8')])
        logging.info('Stage 2 startup: %d process(es)', world_size)
        logging.info('Output directory: %s', run_dir.resolve())
        if args.validation_config:
            logging.info('Validation image selection starts at step %d', args.verbose)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    dataset_started = time.monotonic()
    logging.info('Loading training data: %s', args.data_root)
    dataset = training_data(args)
    logging.info('Training data ready in %.1fs', time.monotonic() - dataset_started)
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed)
               if distributed else None)
    workers = args.num_workers
    fork_workers = workers > 0 and sys.platform.startswith('linux')
    context = ('fork' if fork_workers else 'spawn') if workers else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=workers,
                        multiprocessing_context=context, persistent_workers=bool(workers))
    # Fork before CUDA/NCCL starts so workers can share the read-only RecordIO indices.
    if workers:
        logging.info('Starting %d loader workers per GPU (%s)', workers, context)
    loader_started = time.monotonic()
    first_batches = iter(loader)
    logging.info('DataLoader initialized in %.1fs', time.monotonic() - loader_started)

    device = torch.device(args.device)
    if distributed:
        if device.type == 'cuda':
            local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    prefetch = (device.type == 'cuda' and workers > 0
                and os.environ.get('IBOT_DISABLE_CUDA_PREFETCH') != '1')
    logging.info('CUDA batch prefetch: %s', 'on' if prefetch else 'off')
    logging.info('num_classes=%d  sample_pool=%d  fake_per_epoch=%d  batch_size=%d  '
                 'batch_size_per_rank=%d  world_size=%d  total_step=%d',
                 len(dataset.classes), len(dataset), len(loader.sampler) * world_size,
                 args.batch_size * world_size, args.batch_size, world_size, len(loader) * args.epochs)
    model_started = time.monotonic()
    logging.info('Loading Student and Teacher weights: %s', args.checkpoint)
    checkpoint = trusted_load(args.checkpoint)
    if 'student' not in checkpoint or 'teacher' not in checkpoint:
        raise ValueError('Stage 2 requires a full iBOT checkpoint with Student and Teacher weights')
    student = encoder_from_checkpoint(checkpoint, device, role='student').requires_grad_(True)
    real_encoder = encoder_from_checkpoint(checkpoint, device, role='teacher')
    student_patch_head = teacher_patch_head = patch_alignment = None
    if args.patch_alignment_weight > 0:
        student_patch_head = projection_from_checkpoint(checkpoint, 'student').to(device)
        teacher_patch_head = projection_from_checkpoint(checkpoint, 'teacher').to(device).eval().requires_grad_(False)
        patch_alignment = LocalPatchAlignment(
            neighborhood_size=args.patch_neighborhood_size, temperature=args.patch_temperature,
            distance_weight=args.patch_distance_weight).to(device)
    del checkpoint
    real_encoder.blocks[-2:].requires_grad_(True)
    real_encoder.norm.requires_grad_(True)
    head = ArcFace(student.embed_dim, len(dataset.classes)).to(device)
    use_bf16 = (device.type == 'cuda' and not args.fp32
                and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported())
    logging.info('Encoder precision: %s; ArcFace and alignment use FP32', 'BF16 autocast' if use_bf16 else 'FP32')
    model = Stage2(student, real_encoder, head, use_bf16,
                   student_patch_head, teacher_patch_head, patch_alignment)
    if distributed:
        device_ids = [device.index] if device.type == 'cuda' else None
        model = DistributedDataParallel(model, device_ids=device_ids, gradient_as_bucket_view=True)
    groups = []
    for name, module, peak_lr in (
            ('student', student, args.student_lr), ('arcface', head, args.head_lr),
            ('real', real_encoder, args.real_lr), ('patch_head', student_patch_head, args.patch_head_lr)):
        if module is not None:
            groups.append({'params': [p for p in module.parameters() if p.requires_grad],
                           'name': name, 'lr': peak_lr, 'peak_lr': peak_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    total_steps = len(loader) * args.epochs
    logging.info('Peak learning rates: %s; warmup=%.3f, final ratio=%.3f; patch weight=%.3f',
                 {group['name']: group['peak_lr'] for group in groups},
                 args.warmup_ratio, args.min_lr_ratio, args.patch_alignment_weight)
    logging.info('Models ready in %.1fs. Starting first training batch', time.monotonic() - model_started)

    def save_checkpoint(path, epoch, step, metrics):
        state = {'encoder': student.state_dict(), 'real_encoder': real_encoder.state_dict(),
                    'arcface': head.state_dict(), 'identities': dataset.classes,
                    'epoch': epoch, 'global_step': step, 'validation': metrics,
                    'args': {'arch': 'vit_base', 'patch_size': 16},
                    'training_args': vars(args)}
        if student_patch_head is not None:
            state.update(student_patch_head=student_patch_head.state_dict(),
                         teacher_patch_head=teacher_patch_head.state_dict(),
                         patch_projection='ibot_mlp_bottleneck')
        return save_checkpoint_atomically(state, path)

    best_top1 = -1.
    latest_validation = None
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        if sampler is not None and epoch:
            sampler.set_epoch(epoch)
        student.train()
        real_encoder.blocks[-2:].train()
        real_encoder.norm.train()
        if student_patch_head is not None:
            student_patch_head.train()
            teacher_patch_head.eval()
        totals = torch.zeros(5, device=device) if is_main else None
        samples = interval_images = interval_steps = 0
        loader_wait = 0.
        interval_start = wait_started = time.monotonic()
        epoch_batches = first_batches if epoch == 0 else iter(loader)
        batches = tqdm(device_batches(epoch_batches, device, prefetch), total=len(loader),
                       desc=f'Epoch {epoch + 1}/{args.epochs}',
                       disable=not is_main)
        for step, (fake, real, labels) in enumerate(batches, 1):
            global_step = epoch * len(loader) + step
            if is_main:
                loader_wait += time.monotonic() - wait_started
            if global_step == 1:
                logging.info('First batch ready (waited %.1fs after model setup); starting forward/backward',
                             time.monotonic() - interval_start)
            optimizer.zero_grad(set_to_none=True)
            factor = learning_rate_factor(global_step - 1, total_steps,
                                          args.warmup_ratio, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr'] = group['peak_lr'] * factor
            fake_loss, real_loss, alignment, patch_loss = model(fake, real, labels)
            loss = (fake_loss + real_loss + args.alignment_weight * alignment
                    + args.patch_alignment_weight * patch_loss)
            loss.backward()
            optimizer.step()
            if is_main:
                totals.add_(torch.stack([loss.detach(), fake_loss.detach(), real_loss.detach(),
                                         alignment.detach(), patch_loss.detach()]), alpha=len(labels))
            samples += len(labels)
            interval_images += len(labels)
            interval_steps += 1
            if is_main and (step == 1 or global_step % args.verbose == 0 or step == len(loader)):
                mean_losses = (totals / samples).tolist()
                elapsed = time.monotonic() - interval_start
                logging.info('Speed %.2f pairs/sec   Loss %.4f (Fake: %.4f, Real: %.4f, Align: %.4f, Patch: %.4f)   '
                             'LearningRate %.8g   Epoch: %d/%d   Global Step: %d/%d   '
                             'Loader wait: %.1fs   Elapsed: %.1fs   ETA: %.1f min',
                             interval_images * world_size / max(elapsed, 1e-6), *mean_losses,
                             optimizer.param_groups[0]['lr'], epoch + 1, args.epochs,
                             global_step, len(loader) * args.epochs,
                             loader_wait, time.monotonic() - training_started,
                             elapsed / interval_steps * (len(loader) * args.epochs - global_step) / 60)
                logging.info('Learning rates: %s',
                             {group['name']: group['lr'] for group in optimizer.param_groups})
                interval_start = time.monotonic()
                interval_images = interval_steps = 0
                loader_wait = 0.
            validate_now = args.validation_config and global_step % args.verbose == 0
            if validate_now and is_main:
                validation_started = time.monotonic()
                if validation is None:
                    logging.info('Step %d: selecting validation images', global_step)
                    validation = RetrievalValidation(args.validation_config, min(args.batch_size, 64),
                                                     0 if distributed else args.num_workers)
                    logging.info('Validation image selection took %.1fs',
                                 time.monotonic() - validation_started)
                logging.info('Step %d: running retrieval validation', global_step)
                metrics = validation.evaluate(student.eval(), real_encoder.eval(), device)
                latest_validation = {'step': global_step, **metrics}
                for name, scores in metrics.items():
                    label = 'fake' if name == 'fake' else 'ori'
                    logging.info('Step %d validation: %s Accuracy (Top-1): %.2f%%  Top-5: %.2f%%  MRR: %.4f',
                                 global_step, label, 100 * scores['top1'], 100 * scores['top5'], scores['mrr'])
                if metrics['fake']['top1'] > best_top1:
                    if save_checkpoint(best_output, epoch + 1, global_step, latest_validation):
                        best_top1 = metrics['fake']['top1']
                        logging.info('Saved best validation checkpoint: %s', best_output)
                    else:
                        logging.error('Continuing training without saving this best update at step %d; '
                                      'best saved Top-1 remains %.4f', global_step, best_top1)
                student.train()
                real_encoder.blocks[-2:].train()
                real_encoder.norm.train()
                logging.info('Step %d: validation finished in %.1fs', global_step,
                             time.monotonic() - validation_started)
            if validate_now:
                if distributed:
                    dist.barrier()
                interval_start = time.monotonic()
            wait_started = time.monotonic()
        if is_main:
            logging.info('Epoch %d/%d summary: loss=%.4f fake=%.4f real=%.4f alignment=%.4f patch=%.4f',
                         epoch + 1, args.epochs, *mean_losses)
            if (epoch + 1) % 5 == 0:
                snapshot = output.with_name(f'{output.stem}_epoch{epoch + 1:02d}{output.suffix}')
                if not save_checkpoint(snapshot, epoch + 1, global_step, latest_validation):
                    logging.error('Epoch snapshot was NOT saved; continuing training: %s', snapshot)

    final_saved = [None]
    if is_main:
        final_saved[0] = save_checkpoint(output, args.epochs, global_step, latest_validation)
        if final_saved[0]:
            logging.info('Saved final checkpoint: %s', output)
    if distributed:
        dist.broadcast_object_list(final_saved, src=0)
        dist.destroy_process_group()
    if not final_saved[0]:
        raise RuntimeError('Training updates finished, but the final checkpoint could not be saved. '
                           'See checkpoint write errors in the training log; earlier successful files remain valid.')


if __name__ == '__main__':
    main()
