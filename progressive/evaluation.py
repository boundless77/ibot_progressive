"""Frozen encoder and fixed-mask interfaces for retrieval evaluation."""
import hashlib
import logging
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from .data import MEAN, STD
from .masking import ProgressiveCRFRPMask, ORGANS, Stage
from .checkpoints import trusted_load, bare_state
from .identity import IdentityEncoder

def load_pair(path, data_root, parsing_root):
    # Apply EXIF once, in the parser preprocessing stage too.
    from PIL import ImageOps
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert('RGB')
    with Image.open(Path(parsing_root) / (str(Path(path).relative_to(data_root)) + '.png')) as im:
        labels = im.copy()
    if image.size != labels.size:
        raise ValueError('Image/parsing dimensions differ: %s' % path)
    return image, labels

def load_full_role(checkpoint_path, device='cpu', role='teacher'):
    """For patch diagnostics only: load original backbone plus its own iBOT head."""
    import models
    import utils
    from models.head import iBOTHead
    checkpoint=trusted_load(checkpoint_path)
    config=checkpoint['args']; config=vars(config) if not isinstance(config,dict) else config
    if config['arch']!='vit_base' or config['patch_size']!=16:
        raise ValueError('Patch diagnostics require ViT-B/16')
    backbone=models.vit_base(patch_size=16,return_all_tokens=True,masked_im_modeling=role=='student')
    shared=config['shared_head'] if role=='student' else config['shared_head_teacher']
    head=iBOTHead(backbone.embed_dim,config['out_dim'],patch_out_dim=config['patch_out_dim'],
                  norm=config.get('norm_in_head'),act=config.get('act_in_head','gelu'),shared_head=shared)
    if role == 'student' and config.get('visible_decoder'):
        from .visible_decoder import VisiblePatchStudent
        model = VisiblePatchStudent(backbone, head)
        model.add_decoder(config['decoder_depth'], config['decoder_width'])
    else:
        model = utils.MultiCropWrapper(backbone, head)
    model.load_state_dict(bare_state(checkpoint[role]),strict=True)
    centers=checkpoint.get('ibot_loss',{})
    center=centers.get('center2')
    model.to(device).eval().requires_grad_(False)
    return model,center

def load_encoder(checkpoint_path, device='cpu', role='teacher'):
    return encoder_from_checkpoint(trusted_load(checkpoint_path), device, role)


def encoder_from_checkpoint(ckpt, device='cpu', role='teacher'):
    """Load a Stage 1 backbone or a Stage 2 CLS encoder."""
    import models
    if role not in ('student', 'teacher'):
        raise ValueError('Identity encoder role must be student or teacher')
    if (ckpt.get('identity_readout') not in (None, 'cls')
            or ckpt.get('fake_preprocess') not in (None, 'none')
            or any(key in ckpt for key in ('predictor', 'face_parser', 'identity_projection'))):
        raise ValueError('This CLS baseline cannot evaluate checkpoints from other readout/preprocessing experiments')
    args = ckpt['args']; args = vars(args) if not isinstance(args, dict) else args
    if args['arch'] != 'vit_base' or args['patch_size'] != 16:
        raise ValueError('Evaluation expects ViT-B/16')
    model = models.vit_base(patch_size=16, return_all_tokens=True)
    if 'real_encoder' in ckpt:
        state = ckpt['real_encoder' if role == 'teacher' else 'encoder']
    else:
        state = ckpt.get('encoder')
    if state is None:
        state = {k[len('backbone.'):]: v for k,v in bare_state(ckpt[role]).items() if k.startswith('backbone.')}
    state = {key: value for key, value in state.items() if key != 'masked_embed'}
    model.load_state_dict(state,strict=True)
    model = IdentityEncoder(model, input_size=ckpt.get('input_size', 224))
    model.to(device).eval().requires_grad_(False)
    return model

def fixed_input(path, data_root, parsing_root, mode, mask_bank, seed=42, mask_ratio=None,
                image_size=224):
    """Fixed RGB-space occlusion at ImageNet mean; identical bank for every model."""
    from PIL import ImageOps
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB').resize((image_size,image_size),Image.Resampling.BICUBIC)
    image = TF.normalize(TF.to_tensor(rgb),MEAN,STD)
    if mode == 'clean':
        return image, np.zeros((image_size // 16, image_size // 16),bool)
    if image_size != 224:
        raise ValueError('Additional patch-occlusion modes require 224px images')
    if not parsing_root or not mask_bank:
        raise ValueError('Occluded evaluation needs parsing_root and a shared mask_bank')
    _, labels = load_pair(path,data_root,parsing_root)
    sem = np.asarray(labels.resize((224,224),Image.Resampling.NEAREST))
    generator = ProgressiveCRFRPMask()
    rel = str(Path(path).resolve().relative_to(Path(data_root).resolve())).replace('\\','/')
    key = hashlib.sha256(rel.encode()).hexdigest()
    fingerprint = hashlib.sha256(np.asarray(rgb).tobytes()+sem.tobytes()).hexdigest()
    bank = Path(mask_bank); bank.mkdir(parents=True,exist_ok=True)
    suffix = '' if mask_ratio is None else f'_ratio{mask_ratio:g}'
    entry = bank/(key+'_'+mode+suffix+'.npz')
    if entry.exists():
        with np.load(entry,allow_pickle=False) as saved:
            if str(saved['fingerprint']) != fingerprint or int(saved['seed']) != seed:
                raise ValueError('Mask bank input/seed changed: '+str(path))
            mask = saved['mask']
    else:
        rng = np.random.default_rng(int(key[:16],16) ^ seed)
        if mode == 'inner':
            mask = generator.patch_any(np.isin(sem,tuple(range(1,10))))
        else:
            count = {'single':1,'double':2,'triple':3}[mode]
            chosen = rng.choice(list(ORGANS),count,replace=False).tolist()
            if any(not np.isin(sem,ORGANS[k]).any() for k in chosen):
                raise ValueError('Missing organ in eval parsing: '+str(path))
            if mask_ratio is None:
                mask = np.logical_or.reduce([generator.organ_patches(sem,k) for k in chosen])
            else:
                generator = ProgressiveCRFRPMask([Stage(1, count, mask_ratio)])
                mask, stats = generator(sem, 0, rng, selected=chosen)
                if stats['overflow']:
                    raise ValueError('Selected organs exceed evaluation mask budget: ' + str(path))
        np.savez_compressed(entry,mask=mask,fingerprint=fingerprint,seed=seed,
                            relative_path=rel,mode=mode)
    pixel_mask = torch.from_numpy(mask).repeat_interleave(16,0).repeat_interleave(16,1)
    image[:,pixel_mask] = 0.  # normalized mean color, same for Stage 0..3
    return image,mask

@torch.inference_mode()
def extract(model, tensor, device):
    return F.normalize(model(tensor.unsqueeze(0).to(device)).float(), dim=-1).cpu()[0]


class RetrievalValidation:
    def __init__(self, config_path, batch_size, num_workers, image_size=224):
        from omegaconf import OmegaConf
        from .validation_data import validation_paths

        config = OmegaConf.load(config_path)
        gallery = validation_paths(config.gallery.data)
        fake = validation_paths(config.test.data, fake=True)
        real = validation_paths(config.test_ori.data, check_readable=False)
        identities = sorted(gallery)
        self.num_identities = len(identities)
        if self.num_identities < 2 or not fake:
            raise ValueError('Validation needs at least two gallery identities and a Fake query')
        class_index = {identity: i for i, identity in enumerate(identities)}

        def loader(idbook, skip_unreadable=False):
            missing = idbook.keys() - class_index.keys()
            if missing:
                raise ValueError(f'Validation query identity {min(missing)} is missing from the gallery')
            samples = [(Path(path), class_index[identity])
                       for identity, paths in idbook.items() for path in paths]
            return DataLoader(samples, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=partial(load_validation_batch, skip_unreadable=skip_unreadable,
                                                 image_size=image_size))

        self.gallery = loader(gallery)
        self.fake = loader(fake)
        self.real = loader(real, skip_unreadable=True)
        logging.info('Validation lists ready: %d gallery, %d Fake, %d Real candidates',
                     len(self.gallery.dataset), len(self.fake.dataset), len(self.real.dataset))

    @torch.inference_mode()
    def evaluate(self, student, real_encoder, device):
        centers = torch.zeros(self.num_identities, real_encoder.embed_dim, device=device)
        for images, labels in self.gallery:
            cls = F.normalize(real_encoder(images.to(device)).float(), dim=-1)
            centers.index_add_(0, labels.to(device), cls)
        centers = F.normalize(centers, dim=-1)
        candidates = torch.arange(self.num_identities, device=device)
        results = {}
        for name, model, loader in (('fake', student, self.fake), ('real', real_encoder, self.real)):
            if not len(loader.dataset):
                continue
            ranks = []
            count = 0
            for batch in loader:
                if batch is None:
                    continue
                images, labels = batch
                count += len(labels)
                labels = labels.to(device)[:, None]
                cls = F.normalize(model(images.to(device)).float(), dim=-1)
                scores = cls @ centers.T
                correct_scores = scores.gather(1, labels)
                ahead = (scores > correct_scores) | ((scores == correct_scores) & (candidates < labels))
                ranks.append(1 + ahead.sum(dim=1))
            if count < len(loader.dataset):
                logging.warning('Validation %s: skipped %d unreadable queries',
                                name, len(loader.dataset) - count)
            if not ranks:
                continue
            ranks = torch.cat(ranks).float()
            results[name] = {'top1': (ranks == 1).float().mean().item(),
                             'top5': (ranks <= 5).float().mean().item(),
                             'mrr': ranks.reciprocal().mean().item()}
        return results


def load_validation_batch(samples, skip_unreadable=False, image_size=224):
    images, labels = [], []
    for path, label in samples:
        try:
            images.append(fixed_input(path, None, None, 'clean', None, image_size=image_size)[0])
        except (OSError, ValueError):
            if not skip_unreadable:
                raise
            continue
        labels.append(label)
    if images:
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)
