"""Student path with an iBOT ViT encoder and a small masked-token decoder."""

import torch
from torch import nn

from models.vision_transformer import Block
from utils import MultiCropWrapper


def gather_patches(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class PatchDecoder(nn.Module):
    def __init__(self, embed_dim, depth, width, num_patches):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, width))
        self.blocks = nn.Sequential(*(Block(width, num_heads=4, mlp_ratio=2., qkv_bias=True)
                                      for _ in range(depth)))
        self.output_proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, visible, visible_indices):
        visible = self.input_proj(visible)
        grid = self.mask_token.to(visible.dtype).expand(visible.shape[0], self.pos_embed.shape[1], -1)
        indices = visible_indices.unsqueeze(-1).expand_as(visible)
        grid = grid.scatter(1, indices, visible)
        return self.output_proj(self.blocks(grid + self.pos_embed.to(grid.dtype)))


class VisiblePatchStudent(MultiCropWrapper):
    def add_decoder(self, depth, width):
        self.decoder = PatchDecoder(self.backbone.embed_dim, depth, width,
                                    self.backbone.patch_embed.num_patches)
        # Keep the original parameter so full iBOT checkpoints load strictly.
        self.backbone.masked_embed.requires_grad_(False)

    def masked_tokens(self, images, mask):
        backbone = self.backbone
        tokens = backbone.prepare_tokens(images)
        mask = mask.flatten(1).bool()
        if mask.shape != tokens[:, 1:].shape[:2] or mask.shape[1] != self.decoder.pos_embed.shape[1]:
            raise ValueError('Decoder requires masks matching the full global patch grid')
        visible_counts = (~mask).sum(1)
        if not torch.all(visible_counts == visible_counts[0]) or not 0 < visible_counts[0] < mask.shape[1]:
            raise ValueError('Each masked global view needs the same nonzero visible patch count')
        visible_indices = (~mask).nonzero(as_tuple=False)[:, 1].reshape(mask.shape[0], -1)
        visible = gather_patches(tokens[:, 1:], visible_indices)
        encoded = torch.cat((tokens[:, :1], visible), dim=1)
        for block in backbone.blocks:
            encoded = block(encoded)
        encoded = backbone.norm(encoded)
        decoded = self.decoder(encoded[:, 1:], visible_indices)
        # iBOT's projection head reads CLS from the encoder and patches from the decoder.
        return torch.cat((encoded[:, :1], decoded), dim=1)

    def forward(self, x, mask=None, **kwargs):
        if mask is None:
            return super().forward(x, **kwargs)
        if not isinstance(x, list):
            x, mask = [x], [mask]
        return self.head(self.masked_tokens(torch.cat(x), torch.cat(mask)))
