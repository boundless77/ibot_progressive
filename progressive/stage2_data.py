"""RecordIO pairs with the same online blending used by BlendFace training."""
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import functional as TF

from .recordio import image_indices, open_recordio


def read_pickle(path):
    with path.open('rb') as file:
        return pickle.load(file)


class PackedMasks:
    def __init__(self, root):
        root = Path(root)
        meta = read_pickle(root / 'meta.pkl')
        self.keys = meta['keys']
        self.stride = meta['key_stride']
        self.shape = (meta['height'], meta['width'])
        self.bits_path = root / 'bits.npy'
        self.bits = None

    def __getstate__(self):
        return dict(self.__dict__, bits=None)

    def get(self, label, image_id):
        key = label * self.stride + image_id
        position = np.searchsorted(self.keys, key)
        if position == len(self.keys) or self.keys[position] != key:
            raise KeyError(f'Packed mask missing for identity {label}, image {image_id}')
        if self.bits is None:
            self.bits = np.load(self.bits_path, mmap_mode='r')
        return np.unpackbits(self.bits[position], count=self.shape[0] * self.shape[1]).reshape(
            self.shape).astype(np.float32)


def blend(target, source, target_mask, source_mask):
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_pixels = target_lab[target_mask == 1]
    source_pixels = source_lab[source_mask == 1]
    scale = (target_pixels.std(0) + 1e-6) / (source_pixels.std(0) + 1e-6)
    source_lab = (source_lab - source_pixels.mean(0)) * scale + target_pixels.mean(0)
    source = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    alpha = cv2.GaussianBlur(target_mask * source_mask, (11, 11), 0)
    alpha = cv2.GaussianBlur((alpha > .99).astype(np.float32), (11, 11), 0)
    maximum = alpha.max()
    if maximum > 0:
        alpha /= maximum
    alpha = alpha[..., None]
    return (source.astype(np.float32) * alpha + target.astype(np.float32) * (1 - alpha)).astype(np.uint8)


class OnlineTargets(Dataset):
    def __init__(self, root, masks, transform):
        self.root = Path(root)
        self.transform = transform
        self.reader = open_recordio(self.root)
        self.reader_pid = os.getpid()
        indices = np.asarray(image_indices(self.reader))
        empty = self.root / 'idx2nearestidxs_empty.pkl'
        self.indices = indices[~np.isin(indices, read_pickle(empty))] if empty.is_file() else indices
        self.neighbors = read_pickle(self.root / 'idx2nearestidxs.pkl')
        self.masks = PackedMasks(masks)
        self.labels, self.image_ids = self.read_metadata(indices)
        self.classes = sorted({str(self.labels[int(index)]) for index in self.indices})
        if len(self.classes) < 2:
            raise ValueError('Online training requires at least two target identities')
        self.class_index = {int(label): index for index, label in enumerate(self.classes)}

    def read_metadata(self, indices):
        label_path = self.root / 'idx2label.json'
        id_path = self.root / 'idx2id.json'
        if label_path.is_file() and id_path.is_file():
            return read_pickle(label_path), read_pickle(id_path)
        if os.name == 'posix':
            import fcntl

            with (self.root / '.stage2-metadata.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if label_path.is_file() and id_path.is_file():
                    return read_pickle(label_path), read_pickle(id_path)
                return self.build_metadata(indices, label_path, id_path)
        return self.build_metadata(indices, label_path, id_path)

    def build_metadata(self, indices, label_path, id_path):
        import mxnet as mx

        labels, image_ids = {}, {}
        for index in tqdm(indices, desc='Reading RecordIO labels'):
            header, _ = mx.recordio.unpack(self.reader.read_idx(int(index)))
            labels[int(index)] = int(np.asarray(header.label).reshape(-1)[0])
            image_ids[int(index)] = int(header.id)
        for path, value in ((label_path, labels), (id_path, image_ids)):
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as file:
                pickle.dump(value, file)
            temporary.replace(path)
        return labels, image_ids

    def __getstate__(self):
        return dict(self.__dict__, reader=None, reader_pid=None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.reader_pid != os.getpid():
            self.reader = open_recordio(self.root)
            self.reader_pid = os.getpid()
        import mxnet as mx

        target_index = int(self.indices[index])
        candidates = self.neighbors[target_index]
        source_index = int(candidates[np.random.randint(len(candidates))])
        images, masks = [], []
        for record in (target_index, source_index):
            _, encoded = mx.recordio.unpack(self.reader.read_idx(record))
            images.append(mx.image.imdecode(encoded).asnumpy())
            masks.append(self.masks.get(self.labels[record], self.image_ids[record]))
        target, source = images
        fake = blend(target, source, *masks)
        label = self.class_index[self.labels[target_index]]
        fake, real = Image.fromarray(fake), Image.fromarray(target)
        if random.random() < .5:
            fake, real = TF.hflip(fake), TF.hflip(real)
        return self.transform(fake), self.transform(real), label
