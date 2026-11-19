"""Training-only Target-guided matching of Student neighborhood patches."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchProjection(nn.Module):
    """Reuse the pretrained iBOT MLP bottleneck, before prototype logits.

    Only the MLP is registered: CLS/prototype weights are unnecessary for cosine
    matching and would introduce unused trainable parameters under DDP.
    """

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, patches):
        return F.normalize(self.mlp(patches.float()), dim=-1)


def projection_from_checkpoint(checkpoint, role):
    from models.head import iBOTHead
    from .checkpoints import bare_state

    config = checkpoint['args']
    config = vars(config) if not isinstance(config, dict) else config
    head = iBOTHead(
        768, config['out_dim'], patch_out_dim=config['patch_out_dim'],
        norm=config.get('norm_in_head'), act=config.get('act_in_head', 'gelu'),
        norm_last_layer=config.get('norm_last_layer', True),
        shared_head=config['shared_head' if role == 'student' else 'shared_head_teacher'],
    )
    state = {key[len('head.'):]: value
             for key, value in bare_state(checkpoint[role]).items()
             if key.startswith('head.')}
    head.load_state_dict(state, strict=True)
    if head.last_layer is None:
        raise ValueError('Patch alignment requires an iBOT bottleneck head')
    return PatchProjection(head.mlp)


class LocalPatchAlignment(nn.Module):
    """Each Teacher patch selects nearby Student patches; no global pooling.

    Matching weights and Teacher targets are detached. Gradients flow through
    the weighted Student features into its projection and encoder.
    """

    def __init__(self, grid_size=14, neighborhood_size=3, temperature=.1,
                 distance_weight=1.):
        super().__init__()
        if neighborhood_size < 1 or neighborhood_size % 2 != 1:
            raise ValueError('Patch neighborhood size must be positive and odd')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Patch temperature must be finite and positive')
        if not math.isfinite(distance_weight) or distance_weight < 0:
            raise ValueError('Patch distance weight must be finite and nonnegative')
        self.temperature = temperature
        self.distance_weight = distance_weight
        positions = torch.stack(torch.meshgrid(torch.arange(grid_size),
                                               torch.arange(grid_size), indexing='ij'), -1).reshape(-1, 2)
        offsets = positions[:, None] - positions[None, :]
        valid = offsets.abs().amax(-1) <= neighborhood_size // 2
        # Pad each row with its own center. Invalid slots are masked before softmax.
        width = int(valid.sum(-1).max())
        indices = torch.arange(len(positions))[:, None].expand(-1, width).clone()
        slots = torch.zeros_like(indices, dtype=torch.bool)
        distances = torch.zeros_like(indices, dtype=torch.float32)
        for index in range(len(positions)):
            candidates = valid[index].nonzero().flatten()
            count = len(candidates)
            indices[index, :count] = candidates
            slots[index, :count] = True
            distances[index, :count] = offsets[index, candidates].square().sum(-1).float()
        self.register_buffer('indices', indices)
        self.register_buffer('valid', slots)
        self.register_buffer('distances', distances)

    def forward(self, student, teacher):
        if student.shape != teacher.shape or student.ndim != 3:
            raise ValueError('Patch features must have matching [B, N, D] shapes')
        if student.shape[1] != self.indices.shape[0]:
            raise ValueError('Patch count does not match the configured grid')
        student = F.normalize(student.float(), dim=-1)
        teacher = F.normalize(teacher.detach().float(), dim=-1)
        with torch.no_grad():
            similarities = torch.stack([
                (student[:, self.indices[:, slot]] * teacher).sum(-1)
                for slot in range(self.indices.shape[1])
            ], dim=-1)
            logits = similarities / self.temperature - self.distance_weight * self.distances
            weights = logits.masked_fill(~self.valid, float('-inf')).softmax(-1)
        # Accumulate one neighbor at a time rather than allocating [B,N,K,D].
        fused = torch.zeros_like(student)
        for slot in range(self.indices.shape[1]):
            fused = fused + weights[..., slot, None] * student[:, self.indices[:, slot]]
        return (1 - F.cosine_similarity(fused, teacher, dim=-1)).mean()


def learning_rate_factor(step, total_steps, warmup_ratio=.05, min_ratio=.1):
    """Zero-based update index; reaches peak and final ratio exactly."""
    warmup_steps = min(int(total_steps * warmup_ratio), max(0, total_steps - 1))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.), 1.)
    return min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * progress))
