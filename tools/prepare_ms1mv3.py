"""Build resumable MS1Mv3 cache shards from BlendFace .npy maps or FACER."""

import _bootstrap
import argparse
import json
import os
import random
import shutil
import sqlite3
import time
import zlib
from array import array
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from progressive.masking import LAPA_LABELS, ORGANS
from progressive.recordio import (BLENDFACE_INDEX, BLENDFACE_MASK_SHAPE, PACKED_MASKS, PACKED_MASK_SHAPE,
                                 GlobalCropValidator, cache_path, image_indices, open_recordio,
                                 pack_mask_labels, packed_mask_settings, read_blendface_mask, read_record,
                                 read_record_with_id, source_signature)
from run_train import DEFAULT_STAGE1_CONFIG, read_stage1_config

BLENDFACE_SAVE_EVERY = 4096


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
    partial.mkdir(exist_ok=True)
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


def export_validation(conn, reader, output, shard, limit):
    rows = conn.execute('SELECT record_id FROM masks WHERE validation = 1 ORDER BY record_id')
    candidates = [row[0] for row in rows]
    chosen = random.Random(shard).sample(candidates, min(limit, len(candidates)))
    images = output / 'validation_images' / f'{shard:04d}'
    masks = output / 'validation_parsing' / f'{shard:04d}'
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in chosen:
        _, image = read_record(reader, index)
        blob, = conn.execute('SELECT labels FROM masks WHERE record_id = ?', (index,)).fetchone()
        name = f'{index}.png'
        image.save(images / name)
        Image.frombytes('L', image.size, zlib.decompress(blob)).save(masks / (name + '.png'))


def parse_records(reader, indices, batch_size, detector, parser, device):
    from prepare_parsing import parse_face, parse_faces

    for offset in range(0, len(indices), batch_size):
        batch = indices[offset:offset + batch_size]
        results = [None] * len(batch)
        groups = {}
        for position, record_id in enumerate(batch):
            try:
                label, image = read_record(reader, record_id)
                rgb = np.array(image)
            except (ValueError, OSError) as exc:
                results[position] = (None, exc)
            else:
                groups.setdefault(rgb.shape, []).append((position, label, rgb))
        for group in groups.values():
            images = [rgb for _, _, rgb in group]
            if len(images) == 1:
                try:
                    maps = [parse_face(images[0], detector, parser, device)]
                except (ValueError, OSError) as exc:
                    maps = [exc]
            else:
                maps = parse_faces(images, detector, parser, device)
            for (position, label, _), labels in zip(group, maps):
                results[position] = (label, labels)
        for record_id, (label, labels) in zip(batch, results):
            yield record_id, label, labels


def prepare(args, root, output, final):
    partial = final.with_suffix('.partial')
    signature = source_signature(root)
    validator = GlobalCropValidator(16, .75)
    settings = dict(patch_size=16, mask_ratio=.75, validation_percent=args.validation_percent,
                    validation_export=args.validation_export)
    import torch
    from prepare_parsing import build_facer

    settings.update(detector_weights=args.detector_weights, parser_weights=args.parser_weights)
    device = torch.device(args.device)
    detector, parser = build_facer(device, args.detector_weights, args.parser_weights)
    expected = dict(source=signature, shard=args.shard, shards=args.shards, labels=LAPA_LABELS,
                    parser='farl/lapa/448', settings=settings)
    reader = open_recordio(root)
    indices = image_indices(reader)
    if isinstance(indices, range):
        indices = indices[(args.shard - indices.start) % args.shards::args.shards]
    else:
        indices = [index for index in indices if index % args.shards == args.shard]
    with closing(sqlite3.connect(partial)) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS masks '
                     '(record_id INTEGER PRIMARY KEY, validation INTEGER, labels BLOB)')
        conn.execute('CREATE TABLE IF NOT EXISTS metadata (info TEXT)')
        row = conn.execute('SELECT info FROM metadata').fetchone()
        if row is None:
            counts = dict(total=0, train=0, validation=0, rejected=0)
            conn.execute('INSERT INTO metadata VALUES (?)', (json.dumps(dict(expected, counts=counts)),))
            conn.commit()
        else:
            previous = json.loads(row[0])
            counts = previous.pop('counts')
            if previous != expected:
                raise ValueError(f'Partial cache settings differ: {partial}')
        start_count = counts['total']
        stop_count = len(indices) if args.limit is None else min(len(indices), start_count + args.limit)
        records = parse_records(reader, indices[start_count:stop_count], args.batch_size, detector, parser, device)
        started = time.monotonic()
        print(f'FACER progress: {start_count:,}/{len(indices):,} records processed', flush=True)
        for record_id, label, labels in records:
            try:
                if isinstance(labels, Exception):
                    raise labels
                resized = Image.fromarray(labels).resize((224, 224), Image.Resampling.NEAREST)
                if not validator(resized):
                    raise ValueError('Full view cannot preserve all organs within the mask budget')
            except (ValueError, OSError) as exc:
                counts['rejected'] += 1
                if counts['rejected'] <= 5:
                    print(f'Rejected record {record_id}: {exc}', flush=True)
            else:
                # Group all photographs of an identity into the same split.
                is_val = ((label * 2654435761) & 0xffffffff) % 100 < args.validation_percent
                conn.execute('INSERT INTO masks VALUES (?, ?, ?)',
                             (record_id, int(is_val), zlib.compress(labels.tobytes(), level=1)))
                counts['validation' if is_val else 'train'] += 1
            counts['total'] += 1
            if counts['total'] % 256 == 0:
                conn.execute('UPDATE metadata SET info = ?', (json.dumps(dict(expected, counts=counts)),))
                conn.commit()
            done = counts['total']
            if done == start_count + 1 or done % args.progress_every == 0 or done == stop_count:
                rate = (done - start_count) / max(time.monotonic() - started, 1e-6)
                remaining_hours = (len(indices) - done) / rate / 3600
                print(f'FACER {args.shard + 1}/{args.shards}: {done:,}/{len(indices):,} '
                      f'({done / len(indices):.1%})  train={counts["train"]:,}  '
                      f'val={counts["validation"]:,}  rejected={counts["rejected"]:,}  '
                      f'{rate:.1f} images/s  ETA {remaining_hours:.1f}h', flush=True)
        conn.execute('UPDATE metadata SET info = ?', (json.dumps(dict(expected, counts=counts)),))
        conn.commit()
        if stop_count < len(indices):
            print(f'Saved progress at {stop_count:,}/{len(indices):,}: {counts} -> {partial}', flush=True)
            return
        export_validation(conn, reader, output, args.shard, args.validation_export)
    partial.replace(final)
    print(f'Completed shard {args.shard}: {counts} -> {final}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(DEFAULT_STAGE1_CONFIG))
    p.add_argument('--data_root')
    p.add_argument('--parsing_root')
    p.add_argument('--blendface-masks')
    p.add_argument('--pack_masks', action='store_true', help='Pack masks in an existing BlendFace index')
    p.add_argument('--facer', action='store_true', help='Use FACER with a separate parsing_root')
    p.add_argument('--shards', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--validation_percent', type=int, default=1)
    p.add_argument('--validation_export', type=int,
                   help='Maximum exported samples per shard (default: 0 for BlendFace, about 1024 total for FACER)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16, help='Images per FACER batch (1 for single-image inference)')
    p.add_argument('--limit', type=int, help='BlendFace sample size, or additional records for FACER')
    p.add_argument('--progress_every', type=int, default=256)
    p.add_argument('--detector_weights')
    p.add_argument('--parser_weights')
    args = p.parse_args()
    if args.facer and args.blendface_masks is not None:
        p.error('Choose either --facer or --blendface-masks')
    if args.facer and args.parsing_root is None:
        p.error('Set --parsing_root to the separate FACER cache directory')
    config = read_stage1_config(args.config, {'data_path': args.data_root,
                                             'parsing_root': args.parsing_root,
                                             'blendface_masks': '' if args.facer else args.blendface_masks,
                                             'detector_weights': args.detector_weights,
                                             'parser_weights': args.parser_weights})
    if not config.get('data_path') or not config.get('parsing_root'):
        p.error('Set data_path and parsing_root in the stage-1 config')
    if args.shards is None:
        args.shards = config['recordio_shards']
    args.blendface_masks = Path(config['blendface_masks']) if config.get('blendface_masks') else None
    if args.pack_masks and (not args.blendface_masks or args.limit is not None):
        p.error('--pack_masks requires an existing BlendFace index and does not accept --limit')
    if args.shards > 0 and args.validation_export is None:
        args.validation_export = 0 if args.blendface_masks else (1024 + args.shards - 1) // args.shards
    if not 0 <= args.shard < args.shards or not 0 <= args.validation_percent < 100 or args.validation_export < 0:
        p.error('Invalid shard or validation split')
    if args.progress_every < 1:
        p.error('progress_every must be positive')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('batch-size and limit must be positive')
    root, output = Path(config['data_path']), Path(config['parsing_root'])
    import fcntl
    if args.blendface_masks:
        if not args.blendface_masks.is_dir():
            p.error(f'Missing BlendFace masks directory: {args.blendface_masks}')
        if args.shards != 1:
            p.error('BlendFace index preparation uses one job')
        print(f'BlendFace .npy (CPU): {args.blendface_masks} -> {output}', flush=True)
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
                return
            if (output / 'index.json').is_file() and (output / BLENDFACE_INDEX).is_file():
                print(f'Already completed: {output}. Use another --parsing_root to build a new index.', flush=True)
                return
            prepare_blendface(args, config, root, output)
        return
    print(f'FACER shard {args.shard + 1}/{args.shards}: {root} -> {output} '
          f'(device={args.device}, batch_size={args.batch_size})', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    final = cache_path(output, args.shard)
    with final.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            print(f'Already completed: {final}')
        else:
            for name in ('detector_weights', 'parser_weights'):
                value = config.get(name)
                if not value or not Path(value).is_file():
                    p.error(f'Set {name} to an existing local weight file: {value or "(unset)"}')
                setattr(args, name, value)
                print(f'FACER {name}: {value}', flush=True)
            prepare(args, root, output, final)


if __name__ == '__main__':
    main()
