"""Compare Stage 0..3 on a common gallery/query manifest and occlusion bank."""

import _bootstrap
import argparse
import hashlib
import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from progressive.checkpoints import trusted_load
from progressive.evaluation import encoder_from_checkpoint, fixed_input, extract


def extract_rows(
    model,
    rows,
    root,
    device,
    name,
    parsing_root=None,
    mode="clean",
    mask_bank=None,
    fake_model=None,
):
    features = []
    logging.info("Extracting %s: %d images", name, len(rows))
    for index, row in enumerate(rows, 1):
        tensor, _ = fixed_input(root / row["path"], root, parsing_root, mode, mask_bank)
        encoder = fake_model if fake_model is not None and "target_id" in row else model
        features.append(extract(encoder, tensor, device))
        if index % 256 == 0 or index == len(rows):
            logging.info("%s: %d/%d", name, index, len(rows))
    return features


def retrieval_metrics(qfeats, clean, centers, identities, queries, mode):
    """Compute identity rankings in one batch; ties follow gallery identity order."""
    features = torch.stack(qfeats)
    scores = features @ centers.T
    identity_index = {identity: index for index, identity in enumerate(identities)}
    targets = torch.tensor(
        [identity_index[str(q.get("target_id", q.get("identity")))] for q in queries]
    )
    positive = scores.gather(1, targets[:, None]).squeeze(1)
    order = scores.argsort(dim=1, descending=True, stable=True)
    hit1 = order[:, 0] == targets
    hit5 = (order[:, :5] == targets[:, None]).any(1)
    is_fake = torch.tensor(["target_id" in q for q in queries])
    metrics = {
        "queries": len(queries),
        "gallery_identities": len(identities),
        "top1": hit1.float().mean().item(),
        "top5": hit5.float().mean().item(),
        "same_identity_cosine_mean": positive.mean().item(),
        "other_identity_cosine_mean": (
            (scores.sum() - positive.sum()) / (len(queries) * (len(identities) - 1))
        ).item(),
        "clean_occluded_cosine_mean": (features * torch.stack(clean)).sum(1).mean().item(),
    }
    source_scores = []
    for i, query in enumerate(queries):
        if "source_id" in query:
            source = scores[i, identity_index[str(query["source_id"])]]
            source_scores.append(
                {
                    "path": query["path"],
                    "s_target": positive[i].item(),
                    "s_source": source.item(),
                    "target_minus_source": (positive[i] - source).item(),
                }
            )
    if source_scores:
        metrics["fake_target_source_scores"] = source_scores
        metrics["target_gt_source_fraction"] = sum(
            row["target_minus_source"] > 0 for row in source_scores
        ) / len(source_scores)
    if is_fake.any():
        metrics["fake_top1"] = hit1[is_fake].float().mean().item()
        metrics["fake_top5"] = hit5[is_fake].float().mean().item()
    prefix = "fake_" if is_fake.any() else ""
    logging.info(
        "[%s] %sTop-1: %.2f%%   Top-5: %.2f%%",
        mode,
        prefix,
        100 * metrics[prefix + "top1"],
        100 * metrics[prefix + "top5"],
    )
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--manifest", required=True, help="JSON with gallery and queries lists")
    p.add_argument("--data_root", required=True)
    p.add_argument("--parsing_root")
    p.add_argument("--mask_bank")
    p.add_argument(
        "--modes",
        nargs="+",
        default=["clean", "single", "double", "triple", "inner"],
        choices=["clean", "single", "double", "triple", "inner"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="Testing: %(asctime)s-%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )
    logging.info("Manifest: %s   Device: %s", args.manifest, args.device)
    torch.set_num_threads(2)
    raw = Path(args.manifest).read_bytes()
    manifest = json.loads(raw)
    gallery, queries = manifest["gallery"], manifest["queries"]
    if not gallery or not queries:
        raise ValueError("Gallery and queries must be nonempty")
    root = Path(args.data_root).resolve()
    gallery_paths = {str((root / r["path"]).resolve()) for r in gallery}
    if any(str((root / q["path"]).resolve()) in gallery_paths for q in queries):
        raise ValueError("Gallery and query may not contain the exact same image")
    # Reject duplicate bytes even when files have different names.
    gallery_hashes = {hashlib.sha256((root / r["path"]).read_bytes()).digest() for r in gallery}
    if any(
        hashlib.sha256((root / q["path"]).read_bytes()).digest() in gallery_hashes for q in queries
    ):
        raise ValueError("Duplicate image content in gallery and queries")
    identities = sorted({str(r["identity"]) for r in gallery})
    if len(identities) < 2:
        raise ValueError("Gallery must contain at least two identities")
    for q in queries:
        target = str(q.get("target_id", q.get("identity")))
        if target not in identities or ("source_id" in q and str(q["source_id"]) not in identities):
            raise ValueError("Gallery must contain query Target and Source identities")
        if "source_id" in q and str(q["source_id"]) == target:
            raise ValueError("Fake Target and Source must be different")
    report = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "note": "Identity retrieval: Real CLS for gallery/Real queries; qT predictor for Stage 2 Fake queries.",
        "checkpoints": {},
    }
    for checkpoint in args.checkpoints:
        logging.info("Checkpoint: %s", checkpoint)
        state = trusted_load(checkpoint)
        model = encoder_from_checkpoint(state, args.device)
        fake_model = (
            encoder_from_checkpoint(state, args.device, role="student")
            if "real_encoder" in state
            else None
        )
        del state
        gfeats = extract_rows(model, gallery, root, args.device, "Gallery")
        clean = extract_rows(
            model, queries, root, args.device, "Queries (clean)", fake_model=fake_model
        )
        identity_index = {identity: index for index, identity in enumerate(identities)}
        gallery_labels = torch.tensor([identity_index[str(row["identity"])] for row in gallery])
        gallery_features = torch.stack(gfeats)
        centers = torch.zeros(len(identities), gallery_features.shape[1])
        centers.index_add_(0, gallery_labels, gallery_features)
        centers = F.normalize(centers, dim=1)
        result = {}
        for mode in args.modes:
            qfeats = (
                clean
                if mode == "clean"
                else extract_rows(
                    model,
                    queries,
                    root,
                    args.device,
                    f"Queries ({mode})",
                    args.parsing_root,
                    mode,
                    args.mask_bank,
                    fake_model,
                )
            )
            result[mode] = {
                "identity": retrieval_metrics(qfeats, clean, centers, identities, queries, mode)
            }
        report["checkpoints"][str(Path(checkpoint).resolve())] = result
        del model, fake_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved results: %s   Log: %s", out, log_path)


if __name__ == "__main__":
    main()
