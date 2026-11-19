"""Image selection rules used by the BlendFace training validation."""
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2


def get_all_image_files(directory):
    extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return [os.path.abspath(os.path.join(root, name)) for root, _, files in os.walk(directory)
            for name in files if name.lower().endswith(extensions)]


def sample_source_ids(image_files, max_source_ids, seed=42):
    if max_source_ids is None or max_source_ids <= 0:
        return image_files
    source_dirs = defaultdict(dict)
    pattern = re.compile(r'(?:id|n)?(\d+)_(?:id|n)?(\d+)')
    for path in image_files:
        folder = Path(path).parent
        match = pattern.fullmatch(folder.name)
        if match and not any(part.lower() == 'real' for part in folder.parts):
            source_dirs[(folder.parent, match[1])][int(match[2])] = folder
    rng = random.Random(seed)
    excluded = set()
    for key in sorted(source_dirs):
        directories = source_dirs[key]
        chosen = rng.sample(sorted(directories), min(max_source_ids, len(directories)))
        excluded.update(directories[source] for source in directories.keys() - set(chosen))
    return [path for path in image_files if Path(path).parent not in excluded]


def select_readable_images_per_pair(image_files, ratio):
    grouped = defaultdict(list)
    for path in image_files:
        grouped[os.path.dirname(path)].append(path)
    selected, unreadable = [], []
    for folder in sorted(grouped):
        take_last = ratio < -1
        limit = int(abs(ratio)) if ratio > 1 or take_last else None
        readable = []
        for path in sorted(grouped[folder], reverse=take_last):
            try:
                if cv2.imread(path) is None:
                    raise ValueError('OpenCV could not decode the image')
            except (OSError, ValueError, cv2.error) as exc:
                unreadable.append({'path': path, 'error': str(exc)})
                continue
            readable.append(path)
            if limit is not None and len(readable) == limit:
                break
        if take_last:
            selected.extend(reversed(readable))
        elif 0 < ratio < 1:
            selected.extend(readable[:int(len(readable) * ratio)])
        elif -1 <= ratio < 0:
            selected.extend(readable[int(len(readable) * ratio):])
        else:
            selected.extend(readable)
    return selected, unreadable


def validation_paths(settings, fake=False, check_readable=True):
    roots = settings.root_path
    if isinstance(roots, str):
        roots = [roots]
    paths = [path for root in roots for path in get_all_image_files(root)]
    if fake:
        paths = sample_source_ids(paths, max_source_ids=5, seed=42)
    selected = paths
    if check_readable or settings.ratio not in (0, 1):
        selected, _ = select_readable_images_per_pair(paths, settings.ratio)
    identities = defaultdict(list)
    for path in selected:
        folder = Path(path).parent.name
        identity = folder.split('_')[0] if fake else folder
        identities[int(re.sub(r'^(?:id|n)', '', identity))].append(path)
    return identities
