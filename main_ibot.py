# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "upstream"))
import datetime
import time
import math
import json
import logging
import warnings
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.head import iBOTHead
from progressive.data import FaceMaskDataset, PairedFaceAugmentation, audit_overflows
from progressive.checkpoints import (read_initial_checkpoint, load_models, load_centers,
                                     restore_experiment, capture_rng)
from progressive.visible_decoder import VisiblePatchStudent
from progressive.recordio import GlobalCropValidator, RecordIOFaceMaskDataset


class ProgressConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def create_run_directory(root):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directory = root / timestamp
    suffix = 1
    while True:
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            suffix += 1
            directory = root / f'{timestamp}_{suffix}'


def train_ibot(args):
    # Ignore the legacy API notices while keeping checkpoint keys and older Torch support.
    for api in ('torch.cuda.amp.GradScaler', 'torch.cuda.amp.autocast',
                'torch.nn.utils.weight_norm'):
        warnings.filterwarnings('ignore', message=rf'`{api.replace(".", "[.]")}(?:\([^`]*\))?` is deprecated',
                                category=FutureWarning)
    initial_checkpoint = read_initial_checkpoint(args)
    utils.init_distributed_mode(args)
    output_dir = [None]
    if utils.is_main_process():
        directory = Path(args.output_dir)
        if args.resume:
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = create_run_directory(directory)
        output_dir[0] = str(directory)
    dist.broadcast_object_list(output_dir, src=0)
    args.output_dir = output_dir[0]
    utils.fix_random_seeds(args.seed)
    logger = logging.getLogger('ibot.training')
    if utils.is_main_process():
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        log_mode = 'a' if args.resume else 'x'
        console = ProgressConsoleHandler(sys.stdout)
        console.setLevel(logging.INFO)
        for handler in (console,
                        logging.FileHandler(Path(args.output_dir) / 'training.log',
                                            mode=log_mode, encoding='utf-8')):
            handler.setFormatter(logging.Formatter('Training: %(asctime)s-%(message)s'))
            logger.addHandler(handler)
        logger.info('Output directory: %s', args.output_dir)
        with open(os.path.join(args.output_dir, 'resolved_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        logger.debug('Git: %s', utils.get_sha())
    cudnn.benchmark = True

    # ============ preparing data ... ============
    recordio = bool(args.recordio_shards)
    transform = PairedFaceAugmentation(
        args.global_crops_scale, args.local_crops_scale,
        args.global_crops_number, args.local_crops_number,
        args.require_full_organs,
        crop_validator=GlobalCropValidator(args.patch_size, .75) if recordio else None)
    if recordio:
        dataset = RecordIOFaceMaskDataset(args.data_path, args.parsing_root, args.stages,
                                         args.patch_size, transform, args.seed, args.recordio_shards,
                                         getattr(args, 'blendface_masks', None),
                                         getattr(args, 'packed_mask_dir', None))
        if dataset.packed_mask_path is not None:
            logger.info('Packed masks: %s', dataset.packed_mask_path)
        logger.debug('RecordIO counts: %s', dataset.counts)
    else:
        dataset = FaceMaskDataset(args.data_path, args.parsing_root, args.stages,
                                  args.patch_size, transform, args.seed, args.manifest,
                                  strict_budget=args.visible_decoder)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        # Workers start after NCCL initialization, so avoid inheriting it via fork.
        multiprocessing_context='spawn' if args.num_workers > 0 else None,
        pin_memory=True,
        drop_last=not recordio,
    )
    if len(data_loader) == 0:
        raise ValueError("Dataset is smaller than one distributed batch; reduce batch size")
    batch_global = args.batch_size_per_gpu * utils.get_world_size()
    logger.info('num_image=%d  batch_size_per_gpu=%d  total_batch_size=%d  '
                'steps_per_epoch=%d  warmup_step=%d  total_step=%d  scaled_peak_lr=%.8g',
                len(dataset), args.batch_size_per_gpu, batch_global, len(data_loader),
                args.warmup_epochs * len(data_loader), args.epochs * len(data_loader),
                args.lr * batch_global / 256.)

    # read_initial_checkpoint validates ViT-B/16 before model construction.
    student = models.vit_base(
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        return_all_tokens=True,
        masked_im_modeling=args.use_masked_im_modeling,
    )
    teacher = models.vit_base(patch_size=args.patch_size, return_all_tokens=True)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student_wrapper = VisiblePatchStudent if args.visible_decoder else utils.MultiCropWrapper
    student = student_wrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    if args.visible_decoder:
        student.add_decoder(args.decoder_depth, args.decoder_width)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # Preserve the separately pretrained student and EMA teacher, including heads.
    load_models(initial_checkpoint, student.module, teacher_without_ddp,
                initialize_decoder=args.visible_decoder and not args.resume)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    load_centers(initial_checkpoint, ibot_loss)

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(utils.get_params_groups(student))
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  

    # ============ optionally resume training ... ============
    start_epoch = 0
    if args.resume:
        start_epoch = restore_experiment(initial_checkpoint, args, optimizer, fp16_scaler, ibot_loss)
    else:
        # Stage 0 is the loaded pretrained pair, without old optimizer/epoch state.
        utils.save_on_master({'student': student.state_dict(), 'teacher': teacher.state_dict(),
                              'args': args, 'epoch': 0, 'ibot_loss': ibot_loss.state_dict(),
                              'experiment': 'progressive-crfrp-stage0',
                              'pretrained_source': args.pretrained},
                             os.path.join(args.output_dir, 'stage0.pth'))
    del initial_checkpoint

    start_time = time.time()
    logger.info('Starting iBOT training')
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
            'experiment': 'progressive-crfrp-v1',
            'rng_states': capture_rng(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        for stage_index, stage in enumerate(args.stages, 1):
            if epoch + 1 == stage['end_epoch']:
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'stage{stage_index}.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
            logger.debug('Epoch %d/%d summary: %s', epoch + 1, args.epochs,
                        json.dumps(train_stats, sort_keys=True))
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time %s', total_time_str)


def cuda_batches(data_loader):
    device = torch.cuda.current_device()
    logger = logging.getLogger('ibot.training')

    def move_batch(batch):
        images, labels, masks, mask_stats = batch
        return ([image.to(device=device, non_blocking=True) for image in images], labels,
                [mask.to(device=device, non_blocking=True) for mask in masks], mask_stats)

    # Without workers, data augmentation runs in this process and uses its RNG.
    if data_loader.num_workers == 0 or os.environ.get('IBOT_DISABLE_CUDA_PREFETCH') == '1':
        logger.info('Loading batches on the training thread with %d DataLoader workers',
                    data_loader.num_workers)
        for batch_index, batch in enumerate(data_loader):
            if batch_index == 0:
                logger.info('First CPU batch ready on the training thread')
            yield move_batch(batch)
        return

    logger.info('Initializing DataLoader with %d workers', data_loader.num_workers)
    loader_iterator = iter(data_loader)
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)

    def load_next(first_batch=False):
        if first_batch:
            logger.info('Waiting for the first CPU batch')
        batch = next(loader_iterator, None)
        if batch is not None:
            if first_batch:
                logger.info('First CPU batch ready; submitting CUDA copy')
            with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                batch = move_batch(batch)
            if first_batch:
                logger.info('First CUDA copy submitted')
        return batch

    # Fetch exactly one batch ahead without delaying the current batch's forward.
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = prefetch.submit(load_next, first_batch=True)
        while (batch := pending.result()) is not None:
            compute_stream.wait_stream(copy_stream)
            images, _, masks, _ = batch
            for tensor in images + masks:
                tensor.record_stream(compute_stream)
            pending = prefetch.submit(load_next)
            yield batch


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    logger = logging.getLogger('ibot.training')
    interval_start = None
    interval_steps = interval_images = 0
    steps = len(data_loader)
    progress = tqdm(cuda_batches(data_loader), total=steps, desc=f'Epoch {epoch + 1}/{args.epochs}',
                    file=sys.stdout, dynamic_ncols=True,
                    disable=not (utils.is_main_process() and sys.stdout.isatty()))
    
    for batch_index, (images, _, masks, mask_stats) in enumerate(progress, 1):
        audit_overflows(mask_stats, data_loader.dataset, args.output_dir, epoch, utils.get_rank())
        # update weight decay and learning rate according to their schedule
        it = steps * epoch + batch_index - 1
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        acc = (probs1[0].argmax(1) == probs2[1].argmax(1)).float().mean()

        # student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # Match EMA parameters by name; shared heads may have aliases.
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            student_parameters = dict(student.module.named_parameters())
            for name, param_k in teacher_without_ddp.named_parameters():
                if name in student_parameters:
                    param_k.data.mul_(m).add_((1 - m) * student_parameters[name].detach().data)

        # Finish this batch for logging while allowing the next copy to continue.
        torch.cuda.current_stream().synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(**{f"mask_{k}": v.float().mean().item() for k, v in mask_stats.items() if not k.startswith('_')})
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)
        if batch_index == 1:
            logger.info('Epoch: %d/%d  Global Step: %d  Loss %.4f (CLS %.4f, Patch %.4f)',
                        epoch + 1, args.epochs, it + 1, loss.item(),
                        all_loss['cls'].item(), all_loss['patch'].item())
            interval_start = time.monotonic()
            continue
        interval_steps += 1
        interval_images += len(images[0]) * utils.get_world_size()
        if batch_index % args.verbose == 0 or batch_index == steps:
            now = time.monotonic()
            elapsed = now - interval_start
            meters = metric_logger.meters
            remaining_steps = steps * args.epochs - it - 1
            remaining_hours = elapsed * remaining_steps / interval_steps / 3600
            fp16_text = (f'   Fp16 Grad Scale: {fp16_scaler.get_scale():.0f}'
                         if fp16_scaler is not None else '')
            logger.info('Speed %.2f samples/sec   Loss %.4f (CLS: %.4f, Patch: %.4f)   '
                        'LearningRate %.6f   Epoch: %d/%d   Global Step: %d%s   Required: %.1f hours',
                        interval_images / max(elapsed, 1e-6), meters['loss'].avg,
                        meters['cls'].avg, meters['patch'].avg, optimizer.param_groups[0]['lr'],
                        epoch + 1, args.epochs, it + 1, fp16_text, remaining_hours)
            interval_start = time.monotonic()
            interval_steps = interval_images = 0

    metric_logger.synchronize_between_processes()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1
        total_loss2 = total_loss2 / n_loss_terms2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

if __name__ == '__main__':
    raise SystemExit('Use python run_train.py')
