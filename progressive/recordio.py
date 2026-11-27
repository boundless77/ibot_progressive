"""MS1Mv3 RecordIO images and indexed, nibble-packed BlendFace semantic masks."""

import io
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate

from .data import FaceMaskDataset
from .masking import InvalidInnerFaceMask, ORGANS, ProgressiveCRFRPMask


class PackedMasks:
    def __init__(self, root):
        root = Path(root)
        with (root / "meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        self.keys = meta["keys"]
        self.stride = meta["key_stride"]
        self.shape = (meta["height"], meta["width"])
        self.bits_path = root / "bits.npy"
        self.bits = None

    def __getstate__(self):
        return dict(self.__dict__, bits=None)

    def get(self, label, image_id):
        key = label * self.stride + image_id
        position = np.searchsorted(self.keys, key)
        if position == len(self.keys) or self.keys[position] != key:
            raise KeyError(f"Packed mask missing for identity {label}, image {image_id}")
        if self.bits is None:
            self.bits = np.load(self.bits_path, mmap_mode="r")
        return (
            np.unpackbits(self.bits[position], count=self.shape[0] * self.shape[1])
            .reshape(self.shape)
            .astype(np.float32)
        )


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


def unpack_mask_labels(packed):
    labels = np.empty(BLENDFACE_MASK_SHAPE, dtype=np.uint8)
    labels[:, ::2] = packed >> 4
    labels[:, 1::2] = packed & 0x0f
    return labels


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


class GlobalCropValidator:
    def __init__(self, patch_size, ratio):
        self.grid = ProgressiveCRFRPMask(patch_size=patch_size)
        self.ratio = ratio

    def __call__(self, labels):
        masks = [self.grid.organ_patches(np.asarray(labels), organ) for organ in ORGANS]
        combined = np.logical_or.reduce(masks)
        return all(mask.any() for mask in masks) and combined.sum() <= int(self.ratio * combined.size)


class RecordIOFaceMaskDataset(FaceMaskDataset):
    """Row i in masks.npy corresponds to record valid-records.npy[i]."""

    def __init__(self, data_root, parsing_root, stages, patch_size, augmentation, seed,
                 packed_mask_dir=None):
        super().__init__(stages, patch_size, augmentation, seed)
        self.data_root, self.parsing_root = Path(data_root), Path(parsing_root)
        metadata = json.loads((self.parsing_root / 'index.json').read_text(encoding='utf-8'))
        if metadata['source'] != source_signature(self.data_root) or metadata['patch_size'] != patch_size:
            raise ValueError('BlendFace index settings differ; rerun tools/prepare_ms1mv3.py')
        self.index_path = self.parsing_root / BLENDFACE_INDEX
        self.indices = np.load(self.index_path, mmap_mode='r', allow_pickle=False)
        self.counts = metadata['counts']
        self.size = self.counts['train']
        if (self.indices.ndim != 1 or self.indices.dtype != np.int64
                or not self.size or len(self.indices) != self.size):
            raise ValueError('BlendFace index must be a nonempty int64 array matching the training count')
        packed = Path(packed_mask_dir) if packed_mask_dir else self.parsing_root / PACKED_MASKS
        self.packed_mask_path = packed / 'masks.npy'
        info = json.loads((packed / 'progress.json').read_text(encoding='utf-8'))
        if info != dict(settings=packed_mask_settings(self.parsing_root, metadata), done=self.size):
            raise ValueError('Packed masks do not match this BlendFace index')
        masks = np.load(self.packed_mask_path, mmap_mode='r', allow_pickle=False)
        if masks.shape != (self.size, *PACKED_MASK_SHAPE) or masks.dtype != np.uint8:
            raise ValueError(f'Invalid packed mask array: {self.packed_mask_path}')
        self.reader = self.packed_masks = self.reader_pid = None

    def __getstate__(self):
        # Spawn workers reopen mappings; do not serialize the multi-million-row index.
        return dict(self.__dict__, reader=None, reader_pid=None, indices=None, packed_masks=None)

    def __len__(self):
        return self.size

    def sample_name(self, index):
        if self.indices is None:
            self.indices = np.load(self.index_path, mmap_mode='r', allow_pickle=False)
        return f'MS1Mv3 record {int(self.indices[index])}'

    def __getitem__(self, index):
        if self.reader_pid != os.getpid():
            self.reader = open_recordio(self.data_root)
            self.indices = np.load(self.index_path, mmap_mode='r', allow_pickle=False)
            self.packed_masks = np.load(self.packed_mask_path, mmap_mode='r', allow_pickle=False)
            self.reader_pid = os.getpid()
        record_id = int(self.indices[index])
        _, image = read_record(self.reader, record_id)
        labels = unpack_mask_labels(self.packed_masks[index])
        if labels.shape != (image.height, image.width):
            raise ValueError(f'Record {record_id}: packed mask size differs from RecordIO image')
        return self.make_sample(index, image, Image.fromarray(labels))


def collate_identity_samples(samples):
    """Drop unusable-mask samples without refetching or changing loader step counts."""
    valid = [sample for sample in samples if sample is not None]
    if valid:
        images, labels, masks, stats = default_collate(valid)
    else:
        images, labels, masks, stats = [], torch.empty(0, dtype=torch.long), [], {}
    stats['_skipped'] = len(samples) - len(valid)
    return images, labels, masks, stats


class RecordIOIdentityMaskDataset(FaceMaskDataset):
    """All RecordIO images with existing identity labels and BlendFace binary masks."""

    def __init__(self, data_root, blend_masks, patch_size, augmentation, seed):
        super().__init__([], patch_size, augmentation, seed, mask_mode='inner')
        self.data_root = Path(data_root)
        self.source = source_signature(self.data_root)
        # Only inspect the RecordIO image range here, not image contents.
        reader = open_recordio(self.data_root)
        try:
            self.indices = image_indices(reader)
        finally:
            reader.close()
        self.size = len(self.indices)
        if not self.size:
            raise ValueError('RecordIO contains no image records')
        # This is the existing BlendFace label metadata, not a new generated index.
        with (self.data_root / 'idx2label.json').open('rb') as handle:
            labels = pickle.load(handle)
        self.classes = np.unique(np.fromiter(
            (int(labels[int(record)]) for record in self.indices), dtype=np.int64, count=self.size))
        if len(self.classes) < 2:
            raise ValueError('Identity training needs at least two identities')
        self.class_index = {int(label): cls for cls, label in enumerate(self.classes)}
        self.counts = dict(total=self.size, train=self.size)
        self.masks = PackedMasks(blend_masks)
        self.reader = self.reader_pid = None

    def __len__(self):
        return self.size

    def __getstate__(self):
        return dict(self.__dict__, reader=None, reader_pid=None)

    def sample_name(self, index):
        return f'MS1Mv3 record {int(self.indices[index])}'

    def __getitem__(self, index):
        if self.reader_pid != os.getpid():
            self.reader = open_recordio(self.data_root)
            self.reader_pid = os.getpid()
        record = int(self.indices[index])
        label, image_id, image = read_record_with_id(self.reader, record)
        if label not in self.class_index:
            raise ValueError(f'Record {record}: identity {label} missing from idx2label.json')
        try:
            mask = self.masks.get(label, image_id)
        except KeyError:
            # No mask for this image: use the same online skip path.
            return None
        if mask.shape != (image.height, image.width):
            return None
        if not mask.any() or mask.all():
            return None
        try:
            views, _, masks, stats = self.make_sample(index, image, Image.fromarray(mask.astype(np.uint8)))
        except InvalidInnerFaceMask:
            # Resizing/cropping and patch intersection can also remove every
            # visible or masked patch, even when the original mask is mixed.
            return None
        except ValueError as exc:
            raise ValueError(
                f'Record {record}, identity={label}, image_id={image_id}, '
                f'original_mask_shape={mask.shape}, '
                f'original_mask_coverage={np.count_nonzero(mask) / mask.size:.6f}: {exc}') from exc
        return views, self.class_index[label], masks, stats
