"""ICT attention using the same Fake/Target pairs and plotting as visualize_attention.py.

Run from ibot_progressive with an ICT backbone checkpoint (e.g. ICT_Base.pth).
Architecture follows blendface/training/backbones/ict.py; no blendface imports needed.
"""

import _bootstrap
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from timm.models.vision_transformer import VisionTransformer

from progressive.checkpoints import trusted_load
from visualize_attention import image_sources, parse_args, select_pairs, selected_heads, save_figure


class ICT(VisionTransformer):
    """CombFace ViT-B/8 at 112px, with the original ICT parameter names."""

    def __init__(self):
        super().__init__(img_size=112, patch_size=8, embed_dim=384, depth=12,
                         num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.outer_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 198, self.embed_dim))
        self.head = None
        self.bn = nn.BatchNorm1d(self.embed_dim)
        # Recent timm versions otherwise bypass attn_drop via fused attention.
        for block in self.blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

    def forward(self, x):
        patches = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1),
                       self.outer_token.expand(x.shape[0], -1, -1), patches), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return (F.normalize(self.bn(x[:, 0].float()), dim=1),
                F.normalize(self.bn(x[:, 1].float()), dim=1))


def load_model(checkpoint, device):
    state = checkpoint
    for key in ("state_dict_backbone", "model_state_dict", "model"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = ICT()
    # Never visualize random or partially loaded weights.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class ICTAttentionCapture:
    """Capture actual softmax probabilities, before dropout, for both query tokens."""

    def __init__(self, model, layers, full):
        self.maps, self.matrices, self.handles = {}, {}, []
        if any(not 1 <= layer <= len(model.blocks) for layer in layers):
            raise ValueError(f"Layers must be in 1..{len(model.blocks)}")
        for layer in layers:
            def capture(module, inputs, layer=layer):
                attention = inputs[0].detach()[0].float()
                self.maps[layer] = attention[:, :2, 2:].cpu()
                if full:
                    self.matrices[layer] = attention.cpu().numpy()
            self.handles.append(model.blocks[layer - 1].attn.attn_drop.register_forward_pre_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def main():
    args = parse_args(model="ict")
    if len(set(args.tokens)) != len(args.tokens):
        raise ValueError("--tokens contains duplicates")
    pairs = select_pairs(image_sources(args), args.num_images, args.seed)
    checkpoint = trusted_load(args.checkpoint)
    model = load_model(checkpoint, args.device)
    heads = selected_heads(args.heads, model.blocks[0].attn.num_heads)
    capture = ICTAttentionCapture(model, args.layers, args.save_full_attention)
    output = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"args": vars(args), "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_epoch": checkpoint.get("epoch"), "pairs": [],
              "note": "ICT 112x112 RGB, bilinear resize, mean/std=0.5; no masking or alignment. "
                      "Token order: inner, outer, 196 patches. Raw softmax probabilities, no rollout "
                      "or renormalization after dropping both identity keys. Layers/heads are 1-based. "
                      "Exact Target original matching and image selection follow visualize_attention.py. "
                      "Shared color scale covers both images and all selected tokens per pair."}
    try:
        for index, pair in enumerate(pairs, 1):
            relative = (Path(f"target_{pair['target_id']}") / f"source_{pair['source_id']}"
                        / f"pair_{index:04d}_{pair['fake'].stem}")
            folder = output / relative
            folder.mkdir(parents=True)
            records, entries = [], {}
            for group in ("fake", "real"):
                with Image.open(pair[group]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB").resize(
                        (112, 112), Image.Resampling.BILINEAR)
                tensor = TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)
                capture.maps.clear()
                capture.matrices.clear()
                model(tensor.unsqueeze(0).to(args.device))
                if set(capture.maps) != set(args.layers):
                    raise RuntimeError("ICT attention hooks did not run; check the installed timm attention implementation")
                arrays = {}
                for token in args.tokens:
                    query = 0 if token == "inner" else 1
                    maps = {layer: capture.maps[layer][:, query].reshape(-1, 14, 14)
                            for layer in args.layers}
                    records.append({"image": image, "maps": maps, "group": group, "role": token})
                    arrays.update({f"layer_{layer}_{token}_to_patch": value.numpy()
                                   for layer, value in maps.items()})
                arrays.update({f"layer_{layer}_full": value for layer, value in capture.matrices.items()})
                np.savez_compressed(folder / f"{group}.npz", **arrays)
                entries[group] = {"source": str(pair[group]), "tokens": args.tokens,
                                  "weights": (relative / f"{group}.npz").as_posix()}
            title = (f"ICT | Target ID: {pair['target_id']} | Source ID: {pair['source_id']}\n"
                     f"Fake: {pair['fake'].name} | Target Real: {pair['real'].name}")
            save_figure(records, heads, args.layers, title, folder / "attention.png", args,
                        query_label="Inner / outer query -> patch keys; no rollout")
            report["pairs"].append({"target_id": pair["target_id"], "source_id": pair["source_id"],
                                    "input_size": 112, "grid_size": 14,
                                    "figure": (relative / "attention.png").as_posix(), **entries})
            print(f"Pair {index}/{len(pairs)} | Target={pair['target_id']} Source={pair['source_id']} | "
                  f"{pair['fake']} <-> {pair['real']}", flush=True)
    finally:
        capture.close()
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
