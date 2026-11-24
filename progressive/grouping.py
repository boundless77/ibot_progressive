"""GroupViT-inspired 4/4/4-stage student for Target/Source identity readout.

This is a ViT-B adaptation, not a reproduction of the text-supervised GroupViT.
Hard assignment competes over groups; pooling normalizes over assigned segments.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GroupingBlock(nn.Module):
    def __init__(self, dim, num_groups):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(1, num_groups, dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        self.segment_norm = nn.LayerNorm(dim, eps=1e-6)
        self.group_norm = nn.LayerNorm(dim, eps=1e-6)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, segments, groups):
        # [B, G, N]: each segment chooses one group, not the reverse.
        segments = self.segment_norm(segments)
        logits = (self.query(self.group_norm(groups)) @ self.key(segments).transpose(1, 2)) * self.scale
        if self.training:
            assignment = F.gumbel_softmax(logits.float(), tau=1.0, hard=True, dim=1)
        else:
            assignment = F.one_hot(logits.argmax(dim=1), num_classes=groups.shape[1]).transpose(1, 2).float()
        weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = weights.to(segments.dtype) @ self.value(segments)
        # An empty group retains its contextual group representation.
        return groups + self.proj(pooled), assignment


class GroupingBackbone(nn.Module):
    def __init__(self, vit):
        super().__init__()
        if len(vit.blocks) != 12:
            raise ValueError("Grouping student requires a 12-block ViT")
        self.embed_dim = vit.embed_dim
        self.patch_embed = vit.patch_embed
        self.pos_embed = nn.Parameter(vit.pos_embed[:, 1:].detach().clone())
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.grouping = nn.ModuleList(GroupingBlock(self.embed_dim, n) for n in (64, 16, 2))

    def forward(self, images, return_assignments=False):
        segments = self.patch_embed(images).flatten(2).transpose(1, 2)
        if segments.shape[1] != self.pos_embed.shape[1]:
            raise ValueError("Grouping student expects its pretrained 224px patch grid")
        segments = self.pos_drop(segments + self.pos_embed)
        assignments = []
        for stage, grouping in enumerate(self.grouping):
            groups = grouping.tokens.expand(images.shape[0], -1, -1)
            count = segments.shape[1]
            tokens = torch.cat((segments, groups), dim=1)
            for block in self.blocks[stage * 4:(stage + 1) * 4]:
                tokens = block(tokens)
            segments, assignment = grouping(tokens[:, :count], tokens[:, count:])
            if return_assignments:
                assignments.append(assignment)
        features = self.norm(segments)
        return (features, assignments) if return_assignments else features


class GroupIdentityEncoder(nn.Module):
    """Group 0 is Target; group 1 is Source. Default forward retrieves Target."""
    def __init__(self, backbone, input_size=224):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.input_size = input_size

    def forward(self, images, return_roles=False):
        features = self.backbone(images)
        return features if return_roles else features[:, 0]
