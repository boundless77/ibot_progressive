"""Visualize actual per-layer ViT CLS attention on Fake and Real images.

Run from ibot_progressive. Pair pre-generated Fake images with exact Target originals.
No training, masking, gradients, attention rollout, or face alignment is applied.
"""

import _bootstrap
import argparse
import json
import random
import re
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
from progressive.validation_data import get_all_image_files


def parse_args(model="ibot"):
    parser = argparse.ArgumentParser(description=__doc__ if model == "ibot" else
                                     "Visualize ICT inner/outer attention on exact Fake/Target pairs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_config", default="configs/stage2_validation.yaml",
                        help="Use test/test_ori root_path when explicit image paths are absent")
    parser.add_argument("--fake_paths", nargs="+", help="Image files or recursively searched directories")
    parser.add_argument("--real_paths", nargs="+", help="Image files or recursively searched directories")
    parser.add_argument("--num_images", type=int, default=8, help="Number of Fake / original Target pairs; 0 selects all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 6, 9, 12],
                        help="1-based transformer layers")
    parser.add_argument("--heads", nargs="+", default=["mean"],
                        help="mean, all (mean plus every head), or 1-based head numbers")
    if model == "ibot":
        parser.add_argument("--fake_role", choices=["student", "teacher"], default="student")
        parser.add_argument("--real_role", choices=["student", "teacher"], default="student")
    else:
        parser.add_argument("--tokens", nargs="+", choices=["inner", "outer"], default=["inner", "outer"])
    parser.add_argument("--image_alpha", type=float, default=0.35,
                        help="Original-image opacity over white under the heatmap")
    parser.add_argument("--heatmap_alpha", type=float, default=0.65)
    parser.add_argument("--scale", choices=["shared", "panel"], default="shared",
                        help="Shared raw probability scale per Fake/Real pair, or per-panel min/max")
    parser.add_argument("--save_full_attention", action="store_true",
                        help="Also save every selected layer's [head, query, key] matrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="/mnt/data/fuanlan/project/ibot_progressive/outputs/attention" +
                        ("_ict" if model == "ict" else ""))
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


def collect_images(sources):
    paths = set()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if source.is_dir():
            paths.update(Path(path).resolve() for path in get_all_image_files(source)
                         if Path(path).suffix.lower() in EXTENSIONS)
        elif source.is_file() and source.suffix.lower() in EXTENSIONS:
            paths.add(source)
        else:
            raise ValueError(f"Not an image file or directory: {source}")
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images found in {sources}")
    return paths


def select_pairs(sources, count, seed):
    """Use the exact Target-image naming rule from FSFM/trace_target/test_same_image.py.

    Fake: <target_id>_<source_id>/<target_stem>_<source_stem>.png
    Real: <target_id>/<target_stem>.jpg (other supported extensions also accepted).
    Missing or ambiguous originals are errors, never same-identity replacements.
    """
    fake_paths = collect_images(sources["fake"])
    if count and count > len(fake_paths):
        raise ValueError(f"Requested {count} pairs, but only found {len(fake_paths)} Fake images")
    selected = random.Random(seed).sample(fake_paths, count) if count else fake_paths
    real_index = {}
    identity_pattern = re.compile(r"(?:id|n)?(\d+)", re.IGNORECASE)
    for path in collect_images(sources["real"]):
        match = identity_pattern.fullmatch(path.parent.name)
        if match is None:
            raise ValueError(f"Expected a Real identity directory such as 12 or id12: {path}")
        key = (str(int(match[1])), path.stem)
        real_index.setdefault(key, []).append(path)
    pair_pattern = re.compile(r"(?:id|n)?(\d+)_(?:id|n)?(\d+)", re.IGNORECASE)
    pairs = []
    for fake in selected:
        match = pair_pattern.fullmatch(fake.parent.name)
        if match is None:
            raise ValueError(f"Expected a target_source Fake directory: {fake}")
        target, source = (str(int(value)) for value in match.groups())
        target_stem, separator, source_stem = fake.stem.partition("_")
        if not (target_stem and separator and source_stem):
            raise ValueError(f"Expected a <target_stem>_<source_stem> Fake filename: {fake}")
        originals = real_index.get((target, target_stem), [])
        if len(originals) != 1:
            raise ValueError(f"Expected exactly one original Target image for {fake}; "
                             f"Target={target}, image={target_stem}, matches={originals}")
        pairs.append({"fake": fake, "real": originals[0], "target_id": target, "source_id": source})
    return pairs


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


def save_figure(records, heads, layers, title, output, args,
                query_label="CLS query -> patch keys; no rollout"):
    # Interpolate spatial maps only; never interpolate the token-token matrix.
    panels = []
    for head in heads:
        for record in records:
            maps = record["maps"]
            panels.append(torch.stack([maps[layer].mean(0) if head is None else maps[layer][head]
                                       for layer in layers]))
    panels = torch.stack(panels)
    image = records[0]["image"]
    rows, columns, height, width = panels.shape
    panels = F.interpolate(panels.reshape(-1, 1, height, width),
                           size=(image.height, image.width), mode="nearest")
    panels = panels[:, 0].reshape(rows, columns, image.height, image.width).numpy()
    shared = Normalize(vmin=0, vmax=max(float(panels.max()), 1e-12))
    fig, axes = plt.subplots(rows, columns + 1, figsize=(2.6 * (columns + 1), 2.8 * rows),
                             squeeze=False, layout="constrained")
    try:
        for row in range(rows):
            head = heads[row // len(records)]
            record = records[row % len(records)]
            image = record["image"]
            background = np.asarray(image, dtype=np.float32) / 255
            background = args.image_alpha * background + (1 - args.image_alpha)
            label = "Head mean" if head is None else f"Head {head + 1}"
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"{record['group'].upper()} | {record['role']}\n{label}")
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
        fig.suptitle(title + "\n" + query_label)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


@torch.inference_mode()
def main():
    args = parse_args()
    sources = image_sources(args)
    pairs = select_pairs(sources, args.num_images, args.seed)
    checkpoint = trusted_load(args.checkpoint)
    if checkpoint.get("identity_readout") == "group_roles":
        raise ValueError("Grouping checkpoints have no CLS-to-patch maps; use tools/visualize_groups.py")
    output = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"args": vars(args), "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_epoch": checkpoint.get("epoch"),
              "checkpoint_step": checkpoint.get("global_step"), "pairs": [],
              "note": "Full unmasked RGB input; EXIF transpose, square bicubic resize, ImageNet normalization. "
                      "Raw softmax CLS-to-patch probabilities; no renormalization after dropping CLS key. "
                      "Layers/heads are 1-based. Shared color scale covers both images in each pair. "
                      "Fake naming: target_id_source_id/target_stem_source_stem; "
                      "Real is the exact Target original, matched by identity and image stem. "
                      "Selection ignores validation ratio; there is no online synthesis."}
    models, captures = {}, {}
    try:
        for role in dict.fromkeys((args.fake_role, args.real_role)):
            models[role] = encoder_from_checkpoint(checkpoint, args.device, role=role)
            captures[role] = AttentionCapture(models[role].backbone, args.layers, args.save_full_attention)
        reference = models[args.fake_role]
        size = reference.input_size
        patch_size = reference.backbone.patch_embed.patch_size
        if size % patch_size:
            raise ValueError("Checkpoint input size must be divisible by patch size")
        grid = size // patch_size
        heads = selected_heads(args.heads, reference.backbone.blocks[0].attn.num_heads)
        for model in models.values():
            if (model.input_size != size or model.backbone.patch_embed.patch_size != patch_size
                    or model.backbone.blocks[0].attn.num_heads != reference.backbone.blocks[0].attn.num_heads):
                raise ValueError("Paired visualizations require matching input size, patch size and head count")
        for index, pair in enumerate(pairs, 1):
            relative = (Path(f"target_{pair['target_id']}") / f"source_{pair['source_id']}"
                        / f"pair_{index:04d}_{pair['fake'].stem}")
            folder = output / relative
            folder.mkdir(parents=True)
            records, entries = [], {}
            for group in ("fake", "real"):
                path = pair[group]
                role = getattr(args, f"{group}_role")
                model, capture = models[role], captures[role]
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB").resize(
                        (size, size), Image.Resampling.BICUBIC)
                tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
                capture.maps.clear()
                capture.matrices.clear()
                model(tensor.unsqueeze(0).to(args.device))
                maps = {layer: capture.maps[layer].reshape(-1, grid, grid)
                        for layer in args.layers}
                records.append({"image": image, "maps": maps, "group": group, "role": role})
                arrays = {f"layer_{layer}_cls_to_patch": value.numpy() for layer, value in maps.items()}
                arrays.update({f"layer_{layer}_full": value for layer, value in capture.matrices.items()})
                np.savez_compressed(folder / f"{group}.npz", **arrays)
                entries[group] = {"source": str(path), "role": role,
                                  "weights": (relative / f"{group}.npz").as_posix()}
            title = (f"Target ID: {pair['target_id']} | Source ID: {pair['source_id']}\n"
                     f"Fake: {pair['fake'].name} | Target Real: {pair['real'].name}")
            save_figure(records, heads, args.layers, title, folder / "attention.png", args)
            report["pairs"].append({"target_id": pair["target_id"], "source_id": pair["source_id"],
                                    "input_size": size, "grid_size": grid,
                                    "figure": (relative / "attention.png").as_posix(), **entries})
            print(f"Pair {index}/{len(pairs)} | Target={pair['target_id']} Source={pair['source_id']} | "
                  f"{pair['fake']} <-> {pair['real']}", flush=True)
    finally:
        for capture in captures.values():
            capture.close()
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
