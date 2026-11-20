"""Identity readout and supervised Stage 2 objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class QueryBlock(nn.Module):
    """One query attends to every patch; patches retain their backbone positions."""

    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.patch_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, query, patches):
        patches = self.patch_norm(patches)
        update, _ = self.attention(self.query_norm(query), patches, patches, need_weights=False)
        query = query + update
        return query + self.mlp(query)


class QueryPredictor(nn.Module):
    """P(qT, H_patch): a single learned query, with output in Teacher CLS space."""

    def __init__(self, embed_dim, width=256, heads=4, depth=1):
        super().__init__()
        if min(embed_dim, width, heads, depth) < 1 or width % heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")
        self.config = dict(embed_dim=embed_dim, width=width, heads=heads, depth=depth)
        self.query = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.patch_projection = nn.Linear(embed_dim, width)
        self.blocks = nn.ModuleList([QueryBlock(width, heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embed_dim))

    def forward(self, patches):
        if patches.ndim != 3 or patches.shape[1] == 0:
            raise ValueError("Predictor expects nonempty [batch, patches, embed_dim] tokens")
        patches = self.patch_projection(patches)
        query = self.query.expand(patches.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, patches)
        return self.output(query[:, 0])


class IdentityEncoder(nn.Module):
    """Shared identity interface for training, validation, and retrieval.

    Fake uses qT over patch tokens only. Real and Stage 1 baselines use CLS.
    The optional projection supports older identity-projection checkpoints.
    """

    def __init__(self, backbone, predictor=None, projection=None):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor
        self.projection = projection
        self.embed_dim = projection.out_features if projection is not None else backbone.embed_dim

    def forward(self, images):
        tokens = self.backbone(images)
        features = self.predictor(tokens[:, 1:]) if self.predictor is not None else tokens[:, 0]
        return self.projection(features) if self.projection is not None else features


class ArcFace(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features.float()), F.normalize(self.weight.float()))
        cosine = cosine.clamp(-1.0, 1.0)
        labels = labels[:, None]
        target = cosine.gather(1, labels)
        sine = (1.0 - target.square()).clamp_min(1e-7).sqrt()
        adjusted = target * math.cos(self.margin) - sine * math.sin(self.margin)
        adjusted = torch.where(
            target > math.cos(math.pi - self.margin),
            adjusted,
            target - self.margin * math.sin(math.pi - self.margin),
        )
        return cosine.scatter(1, labels, adjusted) * self.scale


class Stage2(nn.Module):
    """Shared ArcFace supervision and one-way Fake -> Real cosine alignment."""

    def __init__(self, student, real_encoder, head, use_bf16=False):
        super().__init__()
        self.student = student
        self.real_encoder = real_encoder
        self.head = head
        self.use_bf16 = use_bf16

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen Real blocks deterministic, including stochastic depth.
        self.real_encoder.eval()
        if mode:
            self.real_encoder.backbone.blocks[-2:].train()
            self.real_encoder.backbone.norm.train()
        return self

    def forward(self, fake, real, labels):
        with torch.autocast(
            device_type=fake.device.type, dtype=torch.bfloat16, enabled=self.use_bf16
        ):
            fake_features = self.student(fake)
            real_features = self.real_encoder(real)
        fake_features, real_features = fake_features.float(), real_features.float()
        features = torch.cat((fake_features, real_features))
        targets = labels.repeat(2)
        fake_loss, real_loss = F.cross_entropy(
            self.head(features, targets), targets, reduction="none"
        ).chunk(2)
        alignment = 1 - F.cosine_similarity(fake_features, real_features.detach()).mean()
        return fake_loss.mean(), real_loss.mean(), alignment
