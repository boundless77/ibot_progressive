"""Joint augmentation and progressive masking for MS1Mv3 RecordIO samples."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF, InterpolationMode as IM
from .masking import ProgressiveCRFRPMask, InnerFaceMask, ORGANS

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def joint_geometry(image, labels, box, size, flip):
    """One sampled crop and flip used by both RGB and integer semantic labels."""
    i, j, h, w = box
    image = TF.resized_crop(image, i, j, h, w, (size, size), IM.BICUBIC)
    labels = TF.resized_crop(labels, i, j, h, w, (size, size), IM.NEAREST)
    if flip:
        image, labels = TF.hflip(image), TF.hflip(labels)
    # Left/right IDs are intentionally not swapped: all region groups are bilateral.
    return image, labels

class PairedFaceAugmentation:
    def __init__(self, global_scale=(.4, 1.), local_scale=(.05, .4),
                 global_count=2, local_count=10, require_full_organs=True, crop_validator=None):
        self.global_scale, self.local_scale = global_scale, local_scale
        self.global_count, self.local_count = global_count, local_count
        self.require_full_organs = require_full_organs
        self.crop_validator = crop_validator
        self.color = T.Compose([T.RandomApply([T.ColorJitter(.4, .4, .2, .1)], p=.8),
                                T.RandomGrayscale(p=.2)])
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def sample_box(self, image, labels, scale, preserve):
        if tuple(scale) == (1., 1.):
            return (0, 0, image.height, image.width), False
        a = np.asarray(labels)
        if preserve:
            if any(not np.isin(a, ids).any() for ids in ORGANS.values()):
                raise ValueError('Parsing map must contain eyes, nose, mouth; filter it before training')
            ys, xs = np.where(np.isin(a, (4, 5, 6, 7, 8, 9)))
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        for _ in range(20):
            box = T.RandomResizedCrop.get_params(image, scale, (3/4, 4/3))
            i, j, h, w = box
            if not preserve or (i <= y0 and i+h > y1 and j <= x0 and j+w > x1):
                return box, False
        return (0, 0, image.height, image.width), True

    def __call__(self, image, labels):
        import utils  # unchanged iBOT photometric transforms
        views, maps, geometry = [], [], []
        for v in range(self.global_count + self.local_count):
            global_view = v < self.global_count
            size = 224 if global_view else 96
            scale = self.global_scale if global_view else self.local_scale
            box, fallback = self.sample_box(image, labels, scale,
                                            global_view and self.require_full_organs)
            flip = random.random() < .5
            rgb, sem = joint_geometry(image, labels, box, size, flip)
            if global_view and self.crop_validator is not None and not self.crop_validator(sem):
                box = (0, 0, image.height, image.width)
                rgb, sem = joint_geometry(image, labels, box, size, flip)
                fallback = True
            if global_view and self.require_full_organs and any(
                    not np.isin(np.asarray(sem), ids).any() for ids in ORGANS.values()):
                raise ValueError('An organ vanished during resize; remove tiny/invalid face images')
            rgb = self.color(rgb)
            rgb = utils.GaussianBlur(1. if v == 0 else (.1 if global_view else .5))(rgb)
            if global_view and v > 0:
                rgb = utils.Solarization(.2)(rgb)
            views.append(self.normalize(rgb))
            maps.append(np.asarray(sem))
            geometry.append({'box': list(box), 'flip': flip, 'full_image_fallback': fallback})
        return views, maps, geometry

class FaceMaskDataset(Dataset):
    """Common augmentation and mask generation for RecordIO samples."""

    def __init__(self, stages, patch_size, augmentation, seed=0, mask_mode='progressive'):
        self.generator = (InnerFaceMask(patch_size) if mask_mode == 'inner'
                          else ProgressiveCRFRPMask(stages, patch_size))
        self.strict_budget = True
        self.augmentation, self.seed = augmentation, seed
        # Persistent workers share the epoch rather than keeping a stale local copy.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def epoch(self):
        return int(self._epoch.item())

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def make_sample(self, index, image, labels):
        # Per-index/epoch seed is independent of worker scheduling and resume.
        seed = int.from_bytes(hashlib.sha256(('%d:%d:%d' % (self.seed, self.epoch, index)).encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        # Worker RNG is private; only the zero-worker path must restore the
        # training process RNG. Seed CPU directly without touching CUDA APIs.
        restore_rng = torch.utils.data.get_worker_info() is None
        if restore_rng:
            py_state, torch_state = random.getstate(), torch.random.get_rng_state()
        try:
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            views, maps, geometry = self.augmentation(image, labels)
        finally:
            if restore_rng:
                random.setstate(py_state)
                torch.random.set_rng_state(torch_state)
        masks, stats = [], []
        for v, sem in enumerate(maps):
            if v < self.augmentation.global_count:
                mask, stat = self.generator(sem, self.epoch, rng)
                if self.strict_budget and stat['overflow']:
                    raise ValueError(f'{self.sample_name(index)}: global view {v} needs '
                                     f'{stat["forced_patches"]} organ patches, exceeding the '
                                     f'{stat["budget_patches"]}-patch mask budget')
                stat['crop_fallback'] = geometry[v]['full_image_fallback']
                stats.append(stat)
            else:
                # Original iBOT does not mask local views or use their patch loss.
                p = self.generator.patch_size
                mask = np.zeros((sem.shape[0] // p, sem.shape[1] // p), dtype=bool)
            masks.append(mask)
        metrics = {k: float(np.mean([s[k] for s in stats])) for k in
                   ('actual_ratio', 'forced_ratio', 'overflow', 'crop_fallback')}
        if isinstance(self.generator, ProgressiveCRFRPMask):
            metrics['stage'] = stats[0]['stage']
        # Keep per-view information for auditing samples that exceed their budget.
        metrics.update(_index=index,
                       _actual=np.array([s['actual_ratio'] for s in stats]),
                       _forced=np.array([s['forced_ratio'] for s in stats]),
                       _overflow=np.array([s['overflow'] for s in stats]),
                       _organ_bits=np.array([sum({'eyes':1,'nose':2,'mouth':4}[k] for k in s['selected']) for s in stats]))
        # Labels are dummy zeros: there is no identity supervision or identity metric.
        return views, 0, masks, metrics

def audit_overflows(stats, dataset, output_dir, epoch, rank):
    """Per-rank JSONL records retain exact ratios, rather than only batch averages."""
    rows=[]
    for sample,view in stats['_overflow'].nonzero(as_tuple=False).tolist():
        index=int(stats['_index'][sample]);bits=int(stats['_organ_bits'][sample,view])
        rows.append({'epoch':epoch+1,'sample':dataset.sample_name(index),'global_view':view,
                     'selected':[name for name,bit in [('eyes',1),('nose',2),('mouth',4)] if bits & bit],
                     'actual_ratio':float(stats['_actual'][sample,view]),
                     'forced_ratio':float(stats['_forced'][sample,view])})
    if rows:
        with (Path(output_dir)/('overflow_rank%d.jsonl'%rank)).open('a',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
