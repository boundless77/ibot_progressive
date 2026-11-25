"""Original iBOT forward with a parallel identity classifier on backbone CLS."""

import torch
from torch.nn import functional as F
import utils

from .identity import ArcFace


class IdentityStudent(utils.MultiCropWrapper):
    def add_classifier(self, num_classes, margin, scale):
        self.arcface = ArcFace(self.backbone.embed_dim, num_classes, margin, scale)

    def forward(self, images, mask=None, labels=None):
        features, outputs = super().forward(images, mask=mask, return_backbone_feat=True)
        if labels is None:
            return outputs
        targets = labels.repeat(len(images))
        # ArcFace's angular arithmetic and CE stay FP32 under encoder autocast.
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.arcface(features[:, 0].float(), targets)
            loss = F.cross_entropy(logits, targets)
            accuracy = (logits.argmax(1) == targets).float().mean()
        return outputs, loss, accuracy
