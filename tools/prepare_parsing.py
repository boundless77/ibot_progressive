"""Cache full-resolution FACER/LaPa predictions before multi-worker training."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from progressive.data import image_paths, cache_path
from progressive.masking import LAPA_LABELS, ORGANS

def build_facer(device, detector_weights, parser_weights):
    import facer
    detector_kw = {'model_path': detector_weights} if detector_weights else {}
    parser_kw = {'model_path': parser_weights} if parser_weights else {}
    detector = facer.face_detector('retinaface/mobilenet', device=device, **detector_kw)
    parser = facer.face_parser('farl/lapa/448', device=device, **parser_kw)
    if list(parser.label_names) != LAPA_LABELS:
        raise ValueError('FACER parsing label schema differs from LaPa; do not mix CelebM maps')
    return detector, parser


@torch.inference_mode()
def parse_face(rgb, detector, parser, device, allow_multiple_faces=False):
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    faces = detector(tensor)
    n = len(faces['rects'])
    if n == 0 or (n != 1 and not allow_multiple_faces):
        raise ValueError('Expected one detected face, got %d' % n)
    if n > 1:
        rects = faces['rects']
        index = ((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])).argmax().item()
        faces = {key: value[index:index+1] for key, value in faces.items()}
    faces = parser(tensor, faces)
    labels = faces['seg']['logits'][0].argmax(0).cpu().numpy().astype('uint8')
    check_labels(labels, rgb.shape[:2])
    return labels


def check_labels(labels, shape):
    if labels.shape != shape:
        raise ValueError('Parser did not return original image coordinates')
    if any(not np.isin(labels, ids).any() for ids in ORGANS.values()):
        raise ValueError('Missing eyes, nose or mouth in semantic map')


@torch.inference_mode()
def parse_faces(images, detector, parser, device):
    """Parse an equally sized batch, preserving input order and per-image rejects."""
    tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
    faces = detector(tensor)
    image_ids = faces['image_ids'].cpu().numpy().astype(np.intp, copy=False)
    counts = np.bincount(image_ids, minlength=len(images))
    results = [ValueError(f'Expected one detected face, got {count}') if count != 1 else None
               for count in counts]
    selected = np.flatnonzero(counts[image_ids] == 1)
    if selected.size:
        positions = torch.from_numpy(selected).to(device)
        faces = {key: value.index_select(0, positions) for key, value in faces.items()}
        logits = parser(tensor, faces)['seg']['logits']
        maps = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
        for index, labels in zip(image_ids[selected], maps):
            try:
                check_labels(labels, images[index].shape[:2])
            except ValueError as exc:
                results[index] = exc
            else:
                results[index] = labels
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--parsing_root', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--detector_weights', default=None)
    p.add_argument('--parser_weights', default=None)
    p.add_argument('--allow_multiple_faces', action='store_true', help='Select largest detected face')
    args = p.parse_args()
    device = torch.device(args.device)
    detector, parser = build_facer(device, args.detector_weights, args.parser_weights)
    root, output = Path(args.data_root).resolve(), Path(args.parsing_root).resolve()
    if root == output or root in output.parents:
        raise ValueError('Keep parsing_root outside data_root so cached PNGs are not parsed again')
    output.mkdir(parents=True, exist_ok=True)
    paths = image_paths(root)
    if not paths:
        raise ValueError('No face images at data_root')
    valid, failures, records = [], [], []
    for path in paths:
        rel = str(path.relative_to(root))
        try:
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
            labels = parse_face(rgb, detector, parser, device, args.allow_multiple_faces)
            dest = cache_path(path, root, output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(dest)
            valid.append(rel)
            records.append({'image': rel, 'width': rgb.shape[1], 'height': rgb.shape[0],
                            'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
        except Exception as exc:
            failures.append({'image': rel, 'error': str(exc)})
        print('%d/%d valid=%d failed=%d' % (len(valid)+len(failures), len(paths), len(valid), len(failures)))
    (output/'valid_images.txt').write_text('\n'.join(valid)+'\n', encoding='utf-8')
    (output/'failures.json').write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps({'parser': 'farl/lapa/448', 'labels': LAPA_LABELS,
                                                  'exif_transposed': True, 'images': records}, indent=2,
                                                 ensure_ascii=False), encoding='utf-8')
    if not valid:
        raise RuntimeError('No valid parsing results; inspect failures.json')

if __name__ == '__main__':
    main()
