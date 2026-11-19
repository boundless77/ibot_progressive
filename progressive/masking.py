"""CRFR-P proportional allocation with progressive, protected organ masks.

Algorithm reference: FSFM-CVPR25/fsfm-3c/util/datasets.py,
collate_fn_crfrp.variable_proportional_masking. This is an independent NumPy
implementation. Unlike FSFM's overflow branch, selected organs are never unmasked.
"""
from dataclasses import dataclass
import numpy as np

LAPA_LABELS = ['background', 'face', 'rb', 'lb', 're', 'le', 'nose',
               'ulip', 'imouth', 'llip', 'hair']
ORGANS = {'eyes': (4, 5), 'nose': (6,), 'mouth': (7, 8, 9)}
# Preserve FSFM's ordered, overlapping region groups; claim overlapping patches
# only once in this order, as in its masked_patches bookkeeping.
REGIONS = [('brows', (2, 3)), ('eyes', (4, 5)), ('nose', (6,)),
           ('mouth', (7, 8, 9)), ('boundary', None), ('hair', (10,)),
           ('skin', (1,)), ('background', (0,))]

@dataclass(frozen=True)
class Stage:
    end_epoch: int
    organs: int
    ratio: float

class ProgressiveCRFRPMask:
    def __init__(self, stages=None, patch_size=16):
        self.stages = [Stage(**s) if isinstance(s, dict) else s for s in
                       (stages or [Stage(5, 1, .3), Stage(10, 2, .4), Stage(15, 3, .5)])]
        self.patch_size = patch_size
        previous = 0
        for s in self.stages:
            if s.end_epoch <= previous or not 1 <= s.organs <= 3 or not 0 < s.ratio < 1:
                raise ValueError('Stages require increasing end_epoch, organs in 1..3, ratio in (0,1)')
            previous = s.end_epoch
        if patch_size < 1:
            raise ValueError('patch_size must be positive')

    def stage_at(self, epoch):
        """epoch is zero-based; end_epoch is the inclusive human epoch number."""
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        for i, stage in enumerate(self.stages):
            if epoch < stage.end_epoch:
                return i + 1, stage
        raise ValueError('Epoch exceeds configured stages; extend the schedule explicitly')

    def patch_any(self, pixels):
        h, w = pixels.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError('Parsing size must be divisible by patch_size')
        return pixels.reshape(h // p, p, w // p, p).any(axis=(1, 3))

    def organ_patches(self, labels, organ):
        return self.patch_any(np.isin(labels, ORGANS[organ]))

    def __call__(self, labels, epoch, rng=None, selected=None):
        rng = rng if rng is not None else np.random.default_rng()
        labels = np.asarray(labels)
        if labels.ndim != 2 or not np.isin(labels, np.arange(11)).all():
            raise ValueError('Expected a 2D FACER LaPa label map with IDs 0..10')
        stage_id, stage = self.stage_at(epoch)
        available = [k for k in ORGANS if np.isin(labels, ORGANS[k]).any()]
        if len(available) < stage.organs:
            raise ValueError('Too few visible organs for stage: %s; prefilter or change crop' % available)
        if selected is None:
            selected = rng.choice(available, stage.organs, replace=False).tolist()
        if len(set(selected)) != stage.organs or any(k not in available for k in selected):
            raise ValueError('Selected organs must match stage and be present')
        forced = np.logical_or.reduce([self.organ_patches(labels, k) for k in selected])
        mask = forced.copy().ravel()
        protected = mask.copy()
        n = mask.size
        budget = int(stage.ratio * n)  # FSFM truncates patch budget to integer.
        forced_count = int(mask.sum())
        if forced_count < budget:
            fraction = (budget - forced_count) / (n - forced_count)
            claimed = protected.copy()
            for name, ids in REGIONS:
                if name in selected:
                    continue
                if name == 'boundary':
                    region = self.patch_any(np.isin(labels, (10, 0))) & self.patch_any(labels == 1)
                else:
                    region = self.patch_any(np.isin(labels, ids))
                eligible = region.ravel() & ~claimed
                indices = np.flatnonzero(eligible)
                count = int(np.round(indices.size * fraction))
                mask[rng.permutation(indices)[:count]] = True
                claimed |= eligible
            # Correct rounding without ever removing a forced organ patch.
            delta = budget - int(mask.sum())
            candidates = np.flatnonzero(~mask if delta > 0 else (mask & ~protected))
            mask[rng.permutation(candidates)[:abs(delta)]] = delta > 0
        mask = mask.reshape(forced.shape)
        return mask, {'stage': stage_id, 'selected': selected, 'target_ratio': stage.ratio,
                      'actual_ratio': float(mask.mean()), 'forced_ratio': float(forced.mean()),
                      'overflow': forced_count > budget, 'forced_patches': forced_count,
                      'masked_patches': int(mask.sum()), 'budget_patches': budget}
