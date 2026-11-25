"""Train the original CLS Fake/Real identity retrieval baseline."""

import _bootstrap
import argparse
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from progressive.checkpoints import trusted_load
from progressive.evaluation import RetrievalValidation, encoder_from_checkpoint
from progressive.identity import ArcFace, Stage2
from progressive.stage2_data import training_data


def save_checkpoint_atomically(state, path):
    """Save atomically; retry legacy format without replacing a valid file on failure."""
    path = Path(path)
    for zip_format in (True, False):
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=path.name + ".", suffix=".tmp", dir=path.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(state, handle, _use_new_zipfile_serialization=zip_format)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return True
        except (OSError, RuntimeError):
            logging.exception(
                "Checkpoint write failed: path=%s, format=%s%s",
                path.resolve(),
                "zip" if zip_format else "legacy",
                (
                    "; retrying with legacy serialization"
                    if zip_format
                    else "; previous checkpoint preserved, this update was NOT saved"
                ),
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logging.exception("Could not remove incomplete checkpoint: %s", temporary)
    return False


def device_batches(iterator, device, prefetch):
    if not prefetch:
        for batch in iterator:
            yield tuple(tensor.to(device) for tensor in batch)
        return

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)

    def load_next():
        batch = next(iterator, None)
        if batch is not None:
            # Pin here, after the loader workers have forked and CUDA is ready.
            with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                return tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in batch)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(load_next)
        while (batch := pending.result()) is not None:
            compute_stream.wait_stream(copy_stream)
            for tensor in batch:
                tensor.record_stream(compute_stream)
            pending = executor.submit(load_next)
            yield batch


def learning_rate_factor(step, total_steps, warmup_ratio=0.05, min_ratio=0.1):
    """Zero-based update index; reaches peak and final ratio exactly."""
    warmup_steps = min(int(total_steps * warmup_ratio), max(0, total_steps - 1))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--data_root")
    parser.add_argument(
        "--online_masks", help="BlendFace masks_packed; synthesize online from data_root RecordIO"
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--real_recordio", help="MS1Mv3 Real RecordIO used for offline Fake training"
    )
    parser.add_argument(
        "--validation_config", help="YAML with gallery, test and test_ori image settings"
    )
    parser.add_argument("--alignment_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true", help="Disable BF16 encoder autocast")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--student_lr", type=float, help="Peak encoder LR; defaults to lr")
    parser.add_argument("--head_lr", type=float, help="Peak ArcFace LR; defaults to lr")
    parser.add_argument("--real_lr", type=float, help="Peak Real encoder LR; defaults to lr")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--min_lr_ratio", type=float, default=1.0)
    parser.add_argument("--fake_weight", type=float, default=1.0)
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.config:
        defaults = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = defaults.keys() - vars(args).keys()
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
        args = parser.parse_args()
    if not all((args.checkpoint, args.data_root, args.output)):
        parser.error("checkpoint, data_root and output are required")
    for name in ("student_lr", "head_lr", "real_lr"):
        value = getattr(args, name)
        setattr(args, name, args.lr if value is None else value)
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
        for name in ("lr", "student_lr", "head_lr", "real_lr")
    ):
        parser.error("Learning rates must be finite and positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("Invalid learning rate schedule")
    weights = (args.alignment_weight, args.fake_weight, args.real_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
        parser.error(
            "Loss weights must be finite and nonnegative, with at least one positive weight"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.verbose < 1 or args.num_workers < 0:
        parser.error("Invalid training settings")
    if not args.online_masks and not args.real_recordio:
        parser.error("Set real_recordio to the MS1Mv3 RecordIO directory for offline Fake training")
    if args.online_masks and args.manifest:
        parser.error("Select either online synthesis or an offline fake manifest")
    if args.validation_config and not Path(args.validation_config).is_file():
        parser.error(f"Validation config not found: {args.validation_config}")
    return args


def build_model(args, device, num_classes):
    """Initialize from Stage 1 and apply the established backbone training policy."""
    logging.info("Loading Stage 1 Student and Teacher: %s", args.checkpoint)
    checkpoint = trusted_load(args.checkpoint)
    if "student" not in checkpoint or "teacher" not in checkpoint:
        raise ValueError(
            "Stage 2 requires a full Stage 1 checkpoint with Student and Teacher weights"
        )
    student = encoder_from_checkpoint(checkpoint, device, role="student").requires_grad_(True)
    real_encoder = encoder_from_checkpoint(checkpoint, device, role="teacher")
    student.input_size = real_encoder.input_size = 224
    real_encoder.backbone.blocks[-2:].requires_grad_(True)
    real_encoder.backbone.norm.requires_grad_(True)
    head = ArcFace(student.embed_dim, num_classes).to(device)
    use_bf16 = (
        device.type == "cuda"
        and not args.fp32
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    )
    logging.info("Encoder precision: %s; losses use FP32", "BF16 autocast" if use_bf16 else "FP32")
    return Stage2(student, real_encoder, head, use_bf16)


def build_optimizer(model, args):
    modules = (
        ("student", model.student.backbone, args.student_lr),
        ("arcface", model.head, args.head_lr),
        ("real", model.real_encoder, args.real_lr),
    )
    groups = [
        {
            "params": [p for p in module.parameters() if p.requires_grad],
            "name": name,
            "lr": lr,
            "peak_lr": lr,
        }
        for name, module, lr in modules
    ]
    logging.info(
        "Peak learning rates: %s; warmup=%.3f, final ratio=%.3f",
        {group["name"]: group["peak_lr"] for group in groups},
        args.warmup_ratio,
        args.min_lr_ratio,
    )
    logging.info(
        "Loss weights: align=%g, fake=%g, real=%g",
        args.alignment_weight,
        args.fake_weight,
        args.real_weight,
    )
    return torch.optim.AdamW(groups, weight_decay=0.01)


def save_checkpoint(model, identities, args, path, epoch, step, metrics):
    state = {
        "identity_readout": "cls",
        "input_size": model.student.input_size,
        "encoder": model.student.backbone.state_dict(),
        "real_encoder": model.real_encoder.backbone.state_dict(),
        "arcface": model.head.state_dict(),
        "identities": identities,
        "epoch": epoch,
        "global_step": step,
        "validation": metrics,
        "args": {"arch": "vit_base", "patch_size": 16},
        "training_args": vars(args),
    }
    return save_checkpoint_atomically(state, path)


def main():
    args = parse_args()
    output = Path(args.output)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    is_main = rank == 0
    validation = None
    if is_main:
        run_dir = output.parent / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        output = run_dir / output.name
        best_output = output.with_name(f"{output.stem}_best{output.suffix}")
        logging.basicConfig(
            level=logging.INFO,
            format="Training: %(asctime)s-%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(output.with_suffix(".log"), encoding="utf-8"),
            ],
        )
        logging.info("Stage 2 startup: %d process(es)", world_size)
        logging.info("Output directory: %s", run_dir.resolve())
        if args.validation_config:
            logging.info("Validation image selection starts at step %d", args.verbose)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    dataset_started = time.monotonic()
    logging.info("Loading training data: %s", args.data_root)
    dataset = training_data(args)
    logging.info("Training data ready in %.1fs", time.monotonic() - dataset_started)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.seed)
        if distributed
        else None
    )
    workers = args.num_workers
    fork_workers = workers > 0 and sys.platform.startswith("linux")
    context = ("fork" if fork_workers else "spawn") if workers else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=workers,
        multiprocessing_context=context,
        persistent_workers=bool(workers),
    )
    # Fork before CUDA/NCCL starts so workers can share the read-only RecordIO indices.
    if workers:
        logging.info("Starting %d loader workers per GPU (%s)", workers, context)
    loader_started = time.monotonic()
    first_batches = iter(loader)
    logging.info("DataLoader initialized in %.1fs", time.monotonic() - loader_started)

    device = torch.device(args.device)
    if distributed:
        if device.type == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    prefetch = (
        device.type == "cuda"
        and workers > 0
        and os.environ.get("IBOT_DISABLE_CUDA_PREFETCH") != "1"
    )
    logging.info("CUDA batch prefetch: %s", "on" if prefetch else "off")
    logging.info(
        "num_classes=%d  sample_pool=%d  fake_per_epoch=%d  batch_size=%d  "
        "batch_size_per_rank=%d  world_size=%d  total_step=%d",
        len(dataset.classes),
        len(dataset),
        len(loader.sampler) * world_size,
        args.batch_size * world_size,
        args.batch_size,
        world_size,
        len(loader) * args.epochs,
    )
    stage2 = build_model(args, device, len(dataset.classes))
    model = stage2
    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        model = DistributedDataParallel(stage2, device_ids=device_ids, gradient_as_bucket_view=True)
    optimizer = build_optimizer(stage2, args)
    student, real_encoder = stage2.student, stage2.real_encoder
    total_steps = len(loader) * args.epochs
    save = partial(save_checkpoint, stage2, dataset.classes, args)
    best_top1 = -1.0
    latest_validation = None
    loss_names = ("fake", "real", "alignment")
    training_started = time.monotonic()
    for epoch in range(args.epochs):
        if sampler is not None and epoch:
            sampler.set_epoch(epoch)
        stage2.train()
        totals = torch.zeros(1 + len(loss_names), device=device) if is_main else None
        samples = interval_images = interval_steps = 0
        loader_wait = 0.0
        interval_start = wait_started = time.monotonic()
        epoch_batches = first_batches if epoch == 0 else iter(loader)
        batches = tqdm(
            device_batches(epoch_batches, device, prefetch),
            total=len(loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main,
        )
        for step, batch in enumerate(batches, 1):
            batch_size = len(batch[0])
            global_step = epoch * len(loader) + step
            if is_main:
                loader_wait += time.monotonic() - wait_started
            if global_step == 1:
                logging.info(
                    "First batch ready (waited %.1fs after model setup); starting forward/backward",
                    time.monotonic() - interval_start,
                )
            optimizer.zero_grad(set_to_none=True)
            factor = learning_rate_factor(
                global_step - 1, total_steps, args.warmup_ratio, args.min_lr_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            components = model(*batch)
            fake_loss, real_loss, alignment = components
            loss = (
                args.fake_weight * fake_loss
                + args.real_weight * real_loss
                + args.alignment_weight * alignment
            )
            loss.backward()
            optimizer.step()
            if is_main:
                totals.add_(
                    torch.stack(
                        [loss.detach(), *(value.detach() for value in components)]
                    ),
                    alpha=batch_size,
                )
            samples += batch_size
            interval_images += batch_size
            interval_steps += 1
            if is_main and (step == 1 or global_step % args.verbose == 0 or step == len(loader)):
                mean_losses = (totals / samples).tolist()
                elapsed = time.monotonic() - interval_start
                logging.info(
                    "Speed %.2f fakes/sec   Loss %.4f   Components %s   "
                    "LearningRate %.8g   Epoch: %d/%d   Global Step: %d/%d   "
                    "Loader wait: %.1fs   Elapsed: %.1fs   ETA: %.1f min",
                    interval_images * world_size / max(elapsed, 1e-6),
                    mean_losses[0], dict(zip(loss_names, mean_losses[1:])),
                    optimizer.param_groups[0]["lr"],
                    epoch + 1,
                    args.epochs,
                    global_step,
                    len(loader) * args.epochs,
                    loader_wait,
                    time.monotonic() - training_started,
                    elapsed / interval_steps * (len(loader) * args.epochs - global_step) / 60,
                )
                logging.info(
                    "Learning rates: %s",
                    {group["name"]: group["lr"] for group in optimizer.param_groups},
                )
                interval_start = time.monotonic()
                interval_images = interval_steps = 0
                loader_wait = 0.0
            validate_now = args.validation_config and (
                global_step % args.verbose == 0 or global_step == total_steps
            )
            if validate_now and is_main:
                validation_started = time.monotonic()
                if validation is None:
                    logging.info("Step %d: selecting validation images", global_step)
                    validation = RetrievalValidation(
                        args.validation_config,
                        min(args.batch_size, 64),
                        0 if distributed else args.num_workers,
                        image_size=real_encoder.input_size,
                    )
                    logging.info(
                        "Validation image selection took %.1fs",
                        time.monotonic() - validation_started,
                    )
                logging.info("Step %d: running retrieval validation", global_step)
                metrics = validation.evaluate(student.eval(), real_encoder.eval(), device)
                latest_validation = {"step": global_step, **metrics}
                for name, scores in metrics.items():
                    label = {"fake": "fake_target", "real": "ori"}[name]
                    logging.info(
                        "Step %d validation: %s Accuracy (Top-1): %.2f%%  Top-5: %.2f%%  MRR: %.4f",
                        global_step,
                        label,
                        100 * scores["top1"],
                        100 * scores["top5"],
                        scores["mrr"],
                    )
                if metrics["fake"]["top1"] > best_top1:
                    if save(
                        best_output,
                        epoch + 1,
                        global_step,
                        latest_validation,
                    ):
                        best_top1 = metrics["fake"]["top1"]
                        logging.info("Saved best validation checkpoint: %s", best_output)
                    else:
                        logging.error(
                            "Continuing training without saving this best update at step %d; "
                            "best saved Top-1 remains %.4f",
                            global_step,
                            best_top1,
                        )
                stage2.train()
                logging.info(
                    "Step %d: validation finished in %.1fs",
                    global_step,
                    time.monotonic() - validation_started,
                )
            if validate_now:
                if distributed:
                    dist.barrier()
                interval_start = time.monotonic()
            wait_started = time.monotonic()
        if is_main:
            logging.info(
                "Epoch %d/%d summary: loss=%.4f components=%s",
                epoch + 1,
                args.epochs,
                mean_losses[0], dict(zip(loss_names, mean_losses[1:])),
            )
            if (epoch + 1) % 5 == 0:
                snapshot = output.with_name(f"{output.stem}_epoch{epoch + 1:02d}{output.suffix}")
                if not save(
                    snapshot,
                    epoch + 1,
                    global_step,
                    latest_validation,
                ):
                    logging.error("Epoch snapshot was NOT saved; continuing training: %s", snapshot)

    final_saved = [None]
    if is_main:
        final_saved[0] = save(output, args.epochs, global_step, latest_validation)
        if final_saved[0]:
            logging.info("Saved final checkpoint: %s", output)
    if distributed:
        dist.broadcast_object_list(final_saved, src=0)
        dist.destroy_process_group()
    if not final_saved[0]:
        raise RuntimeError(
            "Training updates finished, but the final checkpoint could not be saved. "
            "See checkpoint write errors in the training log; earlier successful files remain valid."
        )


if __name__ == "__main__":
    main()
