"""Index MS1Mv3 RecordIO with existing BlendFace maps, then pack accepted masks."""

import _bootstrap
import argparse
import json
import os
import random
import shutil
import time
from array import array
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from progressive.masking import ORGANS
from progressive.recordio import (BLENDFACE_INDEX, BLENDFACE_MASK_SHAPE, BLENDFACE_TO_LAPA, PACKED_MASKS, PACKED_MASK_SHAPE,
                                 GlobalCropValidator, image_indices, open_recordio,
                                 packed_mask_settings,
                                 read_record_with_id, source_signature)
from run_train import DEFAULT_STAGE1_CONFIG, read_stage1_config

BLENDFACE_SAVE_EVERY = 4096


def pack_mask_labels(labels):
    return (labels[:, ::2] << 4) | labels[:, 1::2]


def read_blendface_mask(root, label, image_id, image):
    path = root / str(label) / f'{image_id}.npy'
    raw = np.load(path, allow_pickle=False)
    if not isinstance(raw, np.ndarray):
        raw.close()
        raise ValueError(f'Expected a .npy array: {path}')
    raw = raw.squeeze()
    if (raw.shape != (image.height, image.width) or raw.dtype.kind not in 'biuf'
            or not np.isin(raw, np.arange(len(BLENDFACE_TO_LAPA))).all()):
        raise ValueError(f'Invalid label IDs or image/mask size mismatch: {path}')
    return BLENDFACE_TO_LAPA[raw.astype(np.intp)]


def write_json_atomically(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_blendface_progress(ids_path, progress_path, settings, counts, pending_ids):
    with ids_path.open('ab') as handle:
        np.asarray(pending_ids, dtype=np.int64).tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    write_json_atomically(progress_path, dict(settings, counts=counts))
    del pending_ids[:]


class RejectedRecord(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def read_blendface_records(reader, indices, mask_root):
    for record_id in indices:
        try:
            label, image_id, image = read_record_with_id(reader, record_id)
        except (ValueError, OSError, EOFError, IndexError) as exc:
            yield record_id, None, None, RejectedRecord('invalid_image', str(exc))
            continue
        path = mask_root / str(label) / f'{image_id}.npy'
        try:
            labels = read_blendface_mask(mask_root, label, image_id, image)
            missing = [name for name, ids in ORGANS.items() if not np.isin(labels, ids).any()]
            if missing:
                raise RejectedRecord('missing_organs', ', '.join(missing))
        except FileNotFoundError:
            labels = RejectedRecord('missing_mask', str(path))
        except RejectedRecord as exc:
            labels = exc
        except (OSError, ValueError, EOFError) as exc:
            labels = RejectedRecord('invalid_mask', f'{path}: {exc}')
        yield record_id, label, image, labels


def pack_blendface_masks(args, config, root, output, final):
    metadata = json.loads((output / 'index.json').read_text(encoding='utf-8'))
    if (metadata['source'] != source_signature(root) or metadata['mask_root'] != str(args.blendface_masks)
            or metadata['patch_size'] != config['patch_size']):
        raise ValueError('BlendFace index settings differ from the current configuration')
    indices = np.load(output / BLENDFACE_INDEX, mmap_mode='r', allow_pickle=False)
    if indices.ndim != 1 or not len(indices) or len(indices) != metadata['counts']['train']:
        raise ValueError('BlendFace index is empty or inconsistent')
    settings = packed_mask_settings(output, metadata)
    if final.exists():
        info = json.loads((final / 'progress.json').read_text(encoding='utf-8'))
        if info != dict(settings=settings, done=len(indices)) or not (final / 'masks.npy').is_file():
            raise ValueError(f'Existing packed masks do not match: {final}')
        print(f'Already completed: {final}', flush=True)
        return
    partial = final.with_name(final.name + '.partial')
    partial.mkdir(parents=True, exist_ok=True)
    progress = partial / 'progress.json'
    path = partial / 'masks.npy'
    shape = (len(indices), *PACKED_MASK_SHAPE)
    if progress.exists():
        info = json.loads(progress.read_text(encoding='utf-8'))
        done = info['done']
        if info['settings'] != settings or not 0 <= done <= len(indices):
            raise ValueError(f'Packed mask progress does not match: {partial}')
        masks = np.lib.format.open_memmap(path, mode='r+')
        if masks.shape != shape or masks.dtype != np.uint8:
            raise ValueError(f'Invalid partial mask array: {path}')
    else:
        done = 0
        required = int(np.prod(shape)) + 4096
        available = shutil.disk_usage(partial).free
        if available < required:
            raise OSError(f'Packed masks need {required / 2**30:.1f} GiB free in {partial.parent}; '
                          f'only {available / 2**30:.1f} GiB available')
        masks = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    print(f'Packing {done:,}/{len(indices):,} masks -> {final}', flush=True)
    if done < len(indices):
        with closing(open_recordio(root)) as reader, path.open('r+b') as handle:
            for row in range(done, len(indices)):
                record_id = int(indices[row])
                label, image_id, image = read_record_with_id(reader, record_id)
                mask = read_blendface_mask(args.blendface_masks, label, image_id, image)
                if mask.shape != BLENDFACE_MASK_SHAPE or any(
                        not np.isin(mask, ids).any() for ids in ORGANS.values()):
                    raise ValueError(f'Record {record_id}: expected a 112x112 map with eyes, nose and mouth')
                masks[row] = pack_mask_labels(mask)
                done = row + 1
                if done % BLENDFACE_SAVE_EVERY == 0 or done == len(indices):
                    masks.flush()
                    os.fsync(handle.fileno())
                    write_json_atomically(progress, dict(settings=settings, done=done))
                if done % args.progress_every == 0 or done == len(indices):
                    print(f'Packed {done:,}/{len(indices):,} masks', flush=True)
    del masks
    partial.rename(final)
    print(f'Completed packed masks: {final}', flush=True)


def prepare_blendface(args, config, root, output):
    validator = GlobalCropValidator(config['patch_size'], .75)
    reader = open_recordio(root)
    indices = image_indices(reader)
    seed = config.get('seed', 42)
    if args.limit is not None and args.limit < len(indices):
        indices = sorted(random.Random(seed).sample(indices, args.limit))
    settings = dict(source=source_signature(root), mask_root=str(args.blendface_masks),
                    patch_size=config['patch_size'], total_records=len(indices), limit=args.limit,
                    seed=seed, validation_percent=args.validation_percent,
                    validation_export=args.validation_export)
    progress_path = output / 'index-progress.json'
    ids_path = output / 'valid-records.partial'
    pending_ids = array('q')
    counts = dict(total=0, train=0, validation=0, rejected=0, rejection_reasons={})
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding='utf-8'))
        counts = saved.pop('counts')
        if saved != settings:
            raise ValueError('Index scan settings differ. Reuse the original arguments or choose another --parsing_root.')
        # A stop after appending but before publishing the counters can leave extra IDs.
        committed_bytes = counts['train'] * np.dtype(np.int64).itemsize
        with ids_path.open('r+b') as handle:
            if handle.seek(0, os.SEEK_END) < committed_bytes:
                raise ValueError(f'Incomplete index checkpoint: {ids_path}')
            handle.truncate(committed_bytes)
    else:
        ids_path.write_bytes(b'')
        save_blendface_progress(ids_path, progress_path, settings, counts, pending_ids)
    start_count = counts['total']
    print(f'BlendFace .npy progress: {start_count:,}/{len(indices):,} records processed', flush=True)
    images = output / 'validation_images' / '0000'
    maps = output / 'validation_parsing' / '0000'
    if args.validation_export:
        images.mkdir(parents=True, exist_ok=True)
        maps.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for record_id, label, image, labels in read_blendface_records(reader, indices[start_count:], args.blendface_masks):
        if not isinstance(labels, RejectedRecord):
            labels = Image.fromarray(labels)
            if not validator(labels.resize((224, 224), Image.Resampling.NEAREST)):
                labels = RejectedRecord('invalid_full_view', 'Cannot preserve all organs within the mask budget')
        if isinstance(labels, RejectedRecord):
            counts['rejected'] += 1
            reasons = counts['rejection_reasons']
            reasons[labels.reason] = reasons.get(labels.reason, 0) + 1
            if reasons[labels.reason] <= 3:
                print(f'Rejected record {record_id} ({labels.reason}): {labels}', flush=True)
        elif ((label * 2654435761) & 0xffffffff) % 100 < args.validation_percent:
            counts['validation'] += 1
            if counts['validation'] <= args.validation_export:
                name = f'{record_id}.png'
                image.save(images / name)
                labels.save(maps / (name + '.png'))
        else:
            counts['train'] += 1
            pending_ids.append(record_id)
        counts['total'] += 1
        done = counts['total']
        if done % BLENDFACE_SAVE_EVERY == 0 or done == len(indices):
            save_blendface_progress(ids_path, progress_path, settings, counts, pending_ids)
        if done == start_count + 1 or done % args.progress_every == 0 or done == len(indices):
            rate = (done - start_count) / max(time.monotonic() - started, 1e-6)
            print(f'BlendFace .npy: {done:,}/{len(indices):,}  train={counts["train"]:,}  '
                  f'val={counts["validation"]:,}  rejected={counts["rejected"]:,}  '
                  f'{rate:.1f} images/s', flush=True)
    if not counts['train']:
        raise ValueError(f'No valid training records: {counts}')
    with (output / 'valid-records.tmp').open('wb') as handle:
        np.save(handle, np.fromfile(ids_path, dtype=np.int64), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    (output / 'valid-records.tmp').replace(output / BLENDFACE_INDEX)
    write_json_atomically(output / 'index.json', dict(settings, counts=counts))
    progress_path.unlink()
    ids_path.unlink()
    print(f'Completed BlendFace index: {counts} -> {output / BLENDFACE_INDEX}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_STAGE1_CONFIG))
    p.add_argument('--data_root')
    p.add_argument('--parsing_root')
    p.add_argument('--blendface-masks')
    p.add_argument('--packed_mask_dir')
    p.add_argument('--pack_masks', action='store_true', help='Pack masks in an existing BlendFace index')
    p.add_argument('--validation_percent', type=int, default=1)
    p.add_argument('--validation_export', type=int, default=0, help='Maximum exported validation image/mask pairs')
    p.add_argument('--limit', type=int, help='Use a reproducible subset of records when building an index')
    p.add_argument('--progress_every', type=int, default=256)
    args = p.parse_args()
    config = read_stage1_config(args.config, {
        'data_path': args.data_root, 'parsing_root': args.parsing_root,
        'blendface_masks': args.blendface_masks, 'packed_mask_dir': args.packed_mask_dir})
    if not all(config.get(key) for key in ('data_path', 'parsing_root', 'blendface_masks')):
        p.error('Set data_path, parsing_root and blendface_masks in the stage-1 config')
    args.blendface_masks = Path(config['blendface_masks'])
    if not args.blendface_masks.is_dir():
        p.error(f'Missing BlendFace masks directory: {args.blendface_masks}')
    if args.pack_masks and args.limit is not None:
        p.error('--pack_masks uses the completed index and does not accept --limit')
    if not 0 <= args.validation_percent < 100 or args.validation_export < 0:
        p.error('Invalid validation split or export count')
    if args.progress_every < 1 or (args.limit is not None and args.limit < 1):
        p.error('progress_every and limit must be positive')
    root, output = Path(config['data_path']), Path(config['parsing_root'])
    if not all((root / name).is_file() for name in ('train.idx', 'train.rec')):
        p.error('data_path must contain train.idx and train.rec')
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.pack_masks:
            if not (output / 'index.json').is_file() or not (output / BLENDFACE_INDEX).is_file():
                p.error('Build the BlendFace index before using --pack_masks')
            final = Path(config['packed_mask_dir']) if config.get('packed_mask_dir') else output / PACKED_MASKS
            final.parent.mkdir(parents=True, exist_ok=True)
            with final.with_name(final.name + '.lock').open('a') as pack_lock:
                fcntl.flock(pack_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pack_blendface_masks(args, config, root, output, final)
        elif (output / 'index.json').is_file() and (output / BLENDFACE_INDEX).is_file():
            print(f'Already completed: {output}. Use another --parsing_root to build a new index.', flush=True)
        else:
            prepare_blendface(args, config, root, output)


if __name__ == '__main__':
    main()
