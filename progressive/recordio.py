"""Read MS1Mv3 images directly and look up their cached FACER label maps."""

import io
import json
import sqlite3
import struct
import warnings
import zlib
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from .data import FaceMaskDataset
from .masking import LAPA_LABELS, ORGANS, ProgressiveCRFRPMask


def open_recordio(root):
    # Older MXNet imports the alias removed in NumPy 1.24.
    if 'bool' not in vars(np):
        np.bool = bool
    import mxnet as mx
    return mx.recordio.MXIndexedRecordIO(str(root / 'train.idx'), str(root / 'train.rec'), 'r')


def source_signature(root):
    return {name: [stat.st_size, stat.st_mtime_ns]
            for name in ('train.idx', 'train.rec') for stat in [(root / name).stat()]}


def image_indices(reader):
    import mxnet as mx
    header, _ = mx.recordio.unpack(reader.read_idx(0))
    if header.flag > 0:
        return range(1, int(header.label[0]))
    return sorted(reader.keys)


BLENDFACE_INDEX = 'valid-records.npy'
PACKED_MASKS = 'packed-masks'
BLENDFACE_MASK_SHAPE = (112, 112)
PACKED_MASK_SHAPE = (112, 56)
# BiSeNet IDs 0..18 to LaPa IDs. Ears and glasses map to face.
BLENDFACE_TO_LAPA = np.array([0, 1, 3, 2, 5, 4, 1, 1, 1, 0, 6, 8, 7, 9, 0, 0, 0, 10, 0],
                            dtype=np.uint8)


def packed_mask_settings(root, index_metadata):
    index = (root / BLENDFACE_INDEX).stat()
    return dict(version=2, index=[index.st_size, index.st_mtime_ns],
                metadata=index_metadata, label_map=BLENDFACE_TO_LAPA.tolist())


def pack_mask_labels(labels):
    return (labels[:, ::2] << 4) | labels[:, 1::2]


def unpack_mask_labels(packed):
    labels = np.empty(BLENDFACE_MASK_SHAPE, dtype=np.uint8)
    labels[:, ::2] = packed >> 4
    labels[:, 1::2] = packed & 0x0f
    return labels


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


def read_record_with_id(reader, index):
    import mxnet as mx
    try:
        record = reader.read_idx(index)
        if not record:
            raise ValueError(f'Empty RecordIO record: {index}')
        header, encoded = mx.recordio.unpack(record)
    except (KeyError, struct.error) as exc:
        raise ValueError(f'Missing or invalid RecordIO record: {index}') from exc
    with Image.open(io.BytesIO(encoded)) as image:
        rgb = image.convert('RGB')
    return int(np.asarray(header.label).reshape(-1)[0]), int(header.id), rgb


def read_record(reader, index):
    label, _, rgb = read_record_with_id(reader, index)
    return label, rgb


def cache_path(root, shard):
    return root / f'parsing-{shard:04d}.sqlite'


def read_cache(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


class GlobalCropValidator:
    def __init__(self, patch_size, ratio):
        self.grid = ProgressiveCRFRPMask(patch_size=patch_size)
        self.ratio = ratio

    def __call__(self, labels):
        masks = [self.grid.organ_patches(np.asarray(labels), organ) for organ in ORGANS]
        combined = np.logical_or.reduce(masks)
        return all(mask.any() for mask in masks) and combined.sum() <= int(self.ratio * combined.size)


class RecordIOFaceMaskDataset(FaceMaskDataset):
    def __init__(self, data_root, parsing_root, stages, patch_size, augmentation, seed, shards,
                 blendface_masks=None, packed_mask_dir=None):
        self.data_root, self.parsing_root = Path(data_root), Path(parsing_root)
        self.generator = ProgressiveCRFRPMask(stages, patch_size)
        self.augmentation, self.seed, self.epoch = augmentation, seed, 0
        self.strict_budget = True
        self.shards = shards
        self.reader = None
        self.caches = {}
        self.blendface_masks = Path(blendface_masks) if blendface_masks else None
        self.packed_mask_path = None
        self.packed_masks = None
        signature = source_signature(self.data_root)
        if self.blendface_masks:
            metadata = json.loads((self.parsing_root / 'index.json').read_text())
            if (metadata['source'] != signature or metadata['mask_root'] != str(self.blendface_masks)
                    or metadata['patch_size'] != patch_size):
                raise ValueError('BlendFace index settings differ; rerun tools/prepare_ms1mv3.py')
            self.indices = np.load(self.parsing_root / BLENDFACE_INDEX, mmap_mode='r')
            self.counts = metadata['counts']
            if not len(self.indices) or len(self.indices) != self.counts['train']:
                raise ValueError('BlendFace index is empty or inconsistent')
            packed = Path(packed_mask_dir) if packed_mask_dir else self.parsing_root / PACKED_MASKS
            if packed.is_dir():
                info = json.loads((packed / 'progress.json').read_text(encoding='utf-8'))
                if (info['settings'] != packed_mask_settings(self.parsing_root, metadata)
                        or info['done'] != len(self.indices)):
                    raise ValueError('Packed masks do not match this BlendFace index')
                self.packed_mask_path = packed / 'masks.npy'
            self.warned = False
            return
        selections = []
        self.counts = dict(total=0, train=0, validation=0, rejected=0)
        reference_settings = None
        for shard in range(shards):
            path = cache_path(self.parsing_root, shard)
            if not path.is_file():
                raise FileNotFoundError(f'Missing completed parsing shard: {path}')
            with closing(read_cache(path)) as conn:
                metadata = json.loads(conn.execute('SELECT info FROM metadata').fetchone()[0])
                if (metadata['source'] != signature or metadata['shards'] != shards
                        or metadata['shard'] != shard or metadata['labels'] != LAPA_LABELS
                        or metadata['parser'] != 'farl/lapa/448'):
                    raise ValueError(f'Parsing shard does not match this dataset: {path}')
                settings = metadata['settings']
                if settings['patch_size'] != patch_size or settings['mask_ratio'] != 0.75:
                    raise ValueError(f'Parsing shard has incompatible crop validation: {path}')
                if reference_settings is not None and settings != reference_settings:
                    raise ValueError(f'Parsing shards were made with different settings: {path}')
                reference_settings = settings
                for key in self.counts:
                    self.counts[key] += metadata['counts'][key]
                ids = np.fromiter((row[0] for row in conn.execute(
                    'SELECT record_id FROM masks WHERE validation = 0 ORDER BY record_id')),
                    dtype=np.int64, count=metadata['counts']['train'])
                selections.append(ids)
        self.indices = np.concatenate(selections)
        if not len(self.indices):
            raise ValueError('No accepted training images in the completed parsing cache')

    def __len__(self):
        return len(self.indices)

    def sample_name(self, index):
        return f'MS1Mv3 record {int(self.indices[index])}'

    def __getitem__(self, index):
        # Open file handles lazily in each DataLoader worker.
        if self.reader is None:
            self.reader = open_recordio(self.data_root)
        if self.blendface_masks:
            if self.packed_mask_path is not None and self.packed_masks is None:
                self.packed_masks = np.load(self.packed_mask_path, mmap_mode='r', allow_pickle=False)
                if (self.packed_masks.shape != (len(self.indices), *PACKED_MASK_SHAPE)
                        or self.packed_masks.dtype != np.uint8):
                    raise ValueError(f'Invalid packed mask array: {self.packed_mask_path}')
            for offset in range(min(32, len(self.indices))):
                candidate = (index + offset) % len(self.indices)
                record_id = int(self.indices[candidate])
                try:
                    label, image_id, image = read_record_with_id(self.reader, record_id)
                    if self.packed_masks is None:
                        labels = read_blendface_mask(self.blendface_masks, label, image_id, image)
                        if any(not np.isin(labels, ids).any() for ids in ORGANS.values()):
                            raise ValueError('Missing eyes, nose or mouth')
                    else:
                        labels = unpack_mask_labels(self.packed_masks[candidate])
                        if labels.shape != (image.height, image.width):
                            raise ValueError('Packed mask size differs from RecordIO image')
                    return self.make_sample(candidate, image, Image.fromarray(labels))
                except (OSError, ValueError, EOFError, IndexError) as exc:
                    if not self.warned:
                        warnings.warn(f'Skipping unavailable BlendFace sample {record_id}: {exc}')
                        self.warned = True
                    continue
            raise RuntimeError(f'No readable BlendFace sample near {self.sample_name(index)}')
        record_id = int(self.indices[index])
        shard = record_id % self.shards
        if shard not in self.caches:
            self.caches[shard] = read_cache(cache_path(self.parsing_root, shard))
        blob, = self.caches[shard].execute(
            'SELECT labels FROM masks WHERE record_id = ?', (record_id,)).fetchone()
        _, image = read_record(self.reader, record_id)
        label_map = Image.frombytes('L', image.size, zlib.decompress(blob))
        return self.make_sample(index, image, label_map)
