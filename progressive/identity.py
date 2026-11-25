"""Identity readout and supervised Stage 2 objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class IdentityEncoder(nn.Module):
    """Use backbone CLS consistently for training and retrieval."""

    def __init__(self, backbone, input_size=224):
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.embed_dim = backbone.embed_dim

    def forward(self, images, mask=None):
        return self.backbone(images, mask=mask)[:, 0]


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
