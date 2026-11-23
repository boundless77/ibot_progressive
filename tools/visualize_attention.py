"""Visualize actual per-layer ViT CLS attention on Fake and Real images.

Run from ibot_progressive; see visualize_attention.md for examples and semantics.
No training, masking, gradients, attention rollout, or face alignment is applied.
"""

import _bootstrap
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from progressive.checkpoints import trusted_load
from progressive.data import EXTENSIONS, MEAN, STD
from progressive.evaluation import encoder_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_config", default="configs/stage2_validation.yaml",
                        help="Use test/test_ori root_path when explicit image paths are absent")
    parser.add_argument("--fake_paths", nargs="+", help="Image files or recursively searched directories")
    parser.add_argument("--real_paths", nargs="+", help="Image files or recursively searched directories")
    parser.add_argument("--num_images", type=int, default=8, help="Count PER group; 0 selects all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 6, 9, 12],
                        help="1-based transformer layers")
    parser.add_argument("--heads", nargs="+", default=["mean"],
                        help="mean, all (mean plus every head), or 1-based head numbers")
    parser.add_argument("--fake_role", choices=["student", "teacher"], default="student")
    parser.add_argument("--real_role", choices=["student", "teacher"], default="student")
    parser.add_argument("--image_alpha", type=float, default=0.35,
                        help="Original-image opacity over white under the heatmap")
    parser.add_argument("--heatmap_alpha", type=float, default=0.65)
    parser.add_argument("--scale", choices=["shared", "panel"], default="shared",
                        help="Shared raw probability scale per image, or per-panel min/max")
    parser.add_argument("--save_full_attention", action="store_true",
                        help="Also save every selected layer's [head, query, key] matrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/attention")
    args = parser.parse_args()
    if args.num_images < 0:
        parser.error("--num_images must be >= 0")
    if not all(0 <= value <= 1 for value in (args.image_alpha, args.heatmap_alpha)):
        parser.error("Opacity must be in [0, 1]")
    if len(set(args.layers)) != len(args.layers):
        parser.error("--layers contains duplicates")
    return args


def image_sources(args):
    sources = {"fake": args.fake_paths, "real": args.real_paths}
    if any(paths is None for paths in sources.values()):
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.data_config)
        for group, section in (("fake", "test"), ("real", "test_ori")):
            if sources[group] is None:
                roots = config[section].data.root_path
                sources[group] = [roots] if isinstance(roots, str) else list(roots)
    return sources


def select_images(sources, count, seed):
    paths = set()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if source.is_dir():
            paths.update(path.resolve() for path in source.rglob("*")
                         if path.is_file() and path.suffix.lower() in EXTENSIONS)
        elif source.is_file() and source.suffix.lower() in EXTENSIONS:
            paths.add(source)
        else:
            raise ValueError(f"Not an image file or directory: {source}")
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {sources}")
    if count and count > len(paths):
        raise ValueError(f"Requested {count} images, but only found {len(paths)} in {sources}")
    return random.Random(seed).sample(paths, count) if count else paths


class AttentionCapture:
    """Read returned softmax matrices without changing the backbone forward pass."""

    def __init__(self, backbone, layers, full):
        self.maps = {}
        self.matrices = {}
        self.handles = []
        for layer in layers:
            if not 1 <= layer <= len(backbone.blocks):
                raise ValueError(f"Layer {layer} outside 1..{len(backbone.blocks)}")
        for layer in layers:
            def capture(module, inputs, output, layer=layer):
                attention = output[1].detach()[0].float()
                # Key 0 is CLS, so patch weights retain their original mass < 1.
                self.maps[layer] = attention[:, 0, 1:].cpu()
                if full:
                    self.matrices[layer] = attention.cpu().numpy()
            self.handles.append(backbone.blocks[layer - 1].attn.register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()


def selected_heads(values, count):
    if values == ["mean"]:
        return [None]
    if values == ["all"]:
        return [None] + list(range(count))
    try:
        heads = [int(value) - 1 for value in values]
    except ValueError as exc:
        raise ValueError("--heads expects mean, all, or 1-based numbers") from exc
    if len(set(heads)) != len(heads) or any(head < 0 or head >= count for head in heads):
        raise ValueError(f"Head numbers must be unique and in 1..{count}")
    return heads


def save_figure(image, maps, heads, layers, title, output, args):
    # Interpolate spatial maps only; never interpolate the token-token matrix.
    panels = torch.stack([
        torch.stack([maps[layer].mean(0) if head is None else maps[layer][head]
                     for layer in layers]) for head in heads
    ])
    rows, columns, height, width = panels.shape
    panels = F.interpolate(panels.reshape(-1, 1, height, width),
                           size=(image.height, image.width), mode="nearest")
    panels = panels[:, 0].reshape(rows, columns, image.height, image.width).numpy()
    shared = Normalize(vmin=0, vmax=max(float(panels.max()), 1e-12))
    fig, axes = plt.subplots(rows, columns + 1, figsize=(2.6 * (columns + 1), 2.8 * rows),
                             squeeze=False, layout="constrained")
    try:
        background = np.asarray(image, dtype=np.float32) / 255
        background = args.image_alpha * background + (1 - args.image_alpha)
        for row, head in enumerate(heads):
            label = "Head mean" if head is None else f"Head {head + 1}"
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"Input | {label}")
            for column, layer in enumerate(layers, 1):
                ax = axes[row, column]
                panel = panels[row, column - 1]
                norm = shared if args.scale == "shared" else Normalize(
                    vmin=float(panel.min()), vmax=max(float(panel.max()), float(panel.min()) + 1e-12))
                ax.imshow(background)
                artist = ax.imshow(panel, cmap="inferno", norm=norm, alpha=args.heatmap_alpha)
                ax.set_title(f"Layer {layer} | {label}")
                if args.scale == "panel":
                    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
        for ax in axes.flat:
            ax.axis("off")
        if args.scale == "shared":
            fig.colorbar(artist, ax=axes[:, 1:].ravel().tolist(), fraction=0.025,
                         pad=0.02, label="Attention probability (before overlay)")
        fig.suptitle(title + "\nCLS query -> patch keys; no rollout")
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


@torch.inference_mode()
def main():
    args = parse_args()
    sources = image_sources(args)
    samples = {group: select_images(paths, args.num_images, args.seed)
               for group, paths in sources.items()}
    checkpoint = trusted_load(args.checkpoint)
    output = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"args": vars(args), "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_epoch": checkpoint.get("epoch"),
              "checkpoint_step": checkpoint.get("global_step"), "images": [],
              "note": "Full unmasked RGB input; EXIF transpose, square bicubic resize, ImageNet normalization. "
                      "Raw softmax CLS-to-patch probabilities; no renormalization after dropping CLS key. "
                      "Layers/heads are 1-based. Shared color scale is per image, not across images. "
                      "Selection ignores validation ratio and does not imply paired Fake/Real samples."}
    for role in dict.fromkeys((args.fake_role, args.real_role)):
        model = encoder_from_checkpoint(checkpoint, args.device, role=role)
        size = model.input_size
        patch_size = model.backbone.patch_embed.patch_size
        if size % patch_size:
            raise ValueError("Checkpoint input size must be divisible by patch size")
        grid = size // patch_size
        heads = selected_heads(args.heads, model.backbone.blocks[0].attn.num_heads)
        capture = AttentionCapture(model.backbone, args.layers, args.save_full_attention)
        try:
            for group in ("fake", "real"):
                if getattr(args, f"{group}_role") != role:
                    continue
                folder = output / group
                folder.mkdir()
                for index, path in enumerate(samples[group], 1):
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB").resize(
                            (size, size), Image.Resampling.BICUBIC)
                    tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
                    capture.maps.clear()
                    capture.matrices.clear()
                    model(tensor.unsqueeze(0).to(args.device))
                    maps = {layer: capture.maps[layer].reshape(-1, grid, grid)
                            for layer in args.layers}
                    name = f"{index:04d}_{path.stem}"
                    save_figure(image, maps, heads, args.layers, f"{group.upper()} | {role} | {path.name}",
                                folder / f"{name}.png", args)
                    arrays = {f"layer_{layer}_cls_to_patch": value.numpy() for layer, value in maps.items()}
                    arrays.update({f"layer_{layer}_full": value for layer, value in capture.matrices.items()})
                    np.savez_compressed(folder / f"{name}.npz", **arrays)
                    report["images"].append({"group": group, "role": role, "source": str(path),
                                             "input_size": size, "grid_size": grid,
                                             "figure": f"{group}/{name}.png", "weights": f"{group}/{name}.npz"})
                    print(f"{group} {index}/{len(samples[group])}: {path}", flush=True)
        finally:
            capture.close()
        del model
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
