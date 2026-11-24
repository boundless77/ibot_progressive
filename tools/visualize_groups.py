"""Show hierarchical hard group assignments projected back to the input patch grid."""

import _bootstrap
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from progressive.checkpoints import trusted_load
from progressive.evaluation import encoder_from_checkpoint, fixed_input
from visualize_attention import collect_images


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True, help="Image files or directories")
    parser.add_argument("--num_images", type=int, default=8, help="0 selects all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/group_assignments")
    args = parser.parse_args()
    if args.num_images < 0:
        parser.error("num_images must be nonnegative")
    checkpoint = trusted_load(args.checkpoint)
    if checkpoint.get("identity_readout") != "group_roles":
        raise ValueError("Expected a grouping student checkpoint")
    model = encoder_from_checkpoint(checkpoint, args.device, role="student")
    del checkpoint
    paths = collect_images(args.images)
    if args.num_images:
        paths = paths[:args.num_images]
    output = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(paths, 1):
        tensor, _ = fixed_input(path, None, None, "clean", None, image_size=model.input_size)
        _, assignments = model.backbone(tensor.unsqueeze(0).to(args.device), return_assignments=True)
        # [64,196], [16,64]@[64,196], [2,16]@[16,196].
        projected = assignments[0][0].float()
        projected_stages = [projected.cpu().numpy()]
        for assignment in assignments[1:]:
            projected = assignment[0].float() @ projected
            projected_stages.append(projected.cpu().numpy())
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        axes[0].imshow(rgb)
        axes[0].set_title("Input")
        for stage, (axis, membership) in enumerate(zip(axes[1:], projected_stages), 1):
            labels = membership.argmax(axis=0).reshape(14, 14)
            labels = np.repeat(np.repeat(labels, 16, axis=0), 16, axis=1)
            axis.imshow(rgb)
            axis.imshow(labels, cmap=plt.get_cmap("turbo", membership.shape[0]) if stage < 3 else "coolwarm", alpha=0.5,
                        interpolation="nearest", vmin=0, vmax=membership.shape[0] - 1)
            axis.set_title(f"Stage {stage}: {membership.shape[0]} groups" if stage < 3
                           else "Final: blue Target / red Source")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{path.parent.name}/{path.name}\nGrouping routes, not CLS attention or identity-purity masks")
        fig.tight_layout()
        stem = f"{index:04d}_{path.parent.name}_{path.stem}"
        fig.savefig(output / f"{stem}.png", dpi=150)
        plt.close(fig)
        np.savez_compressed(output / f"{stem}.npz", **{
            f"stage_{stage}_patch_assignment": value
            for stage, value in enumerate(projected_stages, 1)
        })
    print(f"Saved {len(paths)} grouping visualizations: {output}")


if __name__ == "__main__":
    main()
