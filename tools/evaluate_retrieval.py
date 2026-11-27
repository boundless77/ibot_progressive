"""Evaluate checkpoints using the same YAML retrieval pipeline as training."""

import _bootstrap
import argparse
from io import StringIO
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from progressive.checkpoints import trusted_load
from progressive.evaluation import RetrievalValidation, encoder_from_checkpoint


def method_configs(config, requested=None):
    methods = config.test.get('methods')
    if methods is None:
        if requested:
            raise ValueError('--methods requires named methods under test.methods in the YAML')
        yield 'default', config
        return
    if not OmegaConf.is_dict(methods) or not methods:
        raise ValueError('test.methods must map method names to nonempty path lists')
    if 'root_path' in config.test.data:
        raise ValueError('Use either test.methods or test.data.root_path')
    if requested:
        missing = set(requested) - set(methods)
        if missing:
            raise ValueError(f'Unknown methods: {sorted(missing)}; available: {list(methods)}')
    for name, roots in methods.items():
        if requested and name not in requested:
            continue
        if not OmegaConf.is_list(roots) or not roots or any(not isinstance(p, str) for p in roots):
            raise ValueError(f'test.methods.{name} must be a nonempty path list')
        selected = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        del selected.test.methods
        selected.test.data.root_path = roots
        yield name, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--configs', nargs='+', required=True,
                        help='Training validation YAML files, evaluated separately')
    parser.add_argument('--methods', nargs='+',
                        help='Evaluate only these YAML method names (default: all methods)')
    parser.add_argument('--batch_size', type=int,
                        help='Override YAML batch_size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True, help='JSON output path; also writes a .log file')
    args = parser.parse_args()
    if (args.batch_size is not None and args.batch_size < 1) or args.num_workers < 0:
        parser.error('batch_size must be positive and num_workers must be nonnegative')
    output = Path(args.output)
    if output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.log')
    logging.basicConfig(
        level=logging.INFO,
        format='Testing: %(asctime)s-%(message)s',
        handlers=[logging.FileHandler(log_path, mode='w', encoding='utf-8'),
                  logging.StreamHandler(sys.stdout)],
    )
    device = torch.device(args.device)
    report = {'note': 'Training retrieval: full images, Real encoder for gallery/Real, '
                      'Student for Fake; top1/top5 are fractions.',
              'checkpoints': {}}
    for checkpoint_path in args.checkpoints:
        logging.info('Checkpoint: %s', checkpoint_path)
        checkpoint = trusted_load(checkpoint_path)
        student = encoder_from_checkpoint(checkpoint, device, role='student')
        real_encoder = encoder_from_checkpoint(checkpoint, device, role='teacher')
        del checkpoint
        results = {}
        report['checkpoints'][str(Path(checkpoint_path).resolve())] = results
        for config_path in args.configs:
            logging.info('Configuration: %s', config_path)
            config = OmegaConf.load(config_path)
            batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 128)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError(f'{config_path}: batch_size must be a positive integer')
            method_results = {}
            results[str(Path(config_path).resolve())] = method_results
            for method, selected in method_configs(config, args.methods):
                logging.info('Method: %s  Batch size: %d', method, batch_size)
                validation = RetrievalValidation(
                    StringIO(OmegaConf.to_yaml(selected, resolve=True)), batch_size, args.num_workers,
                    image_size=student.input_size,
                )
                metrics = validation.evaluate(student.eval(), real_encoder.eval(), device)
                method_results[method] = {
                    'config': OmegaConf.to_container(selected, resolve=True),
                    'batch_size': batch_size,
                    'gallery_identities': validation.num_identities,
                    'gallery_images': len(validation.gallery.dataset),
                    'fake_query_candidates': len(validation.fake.dataset),
                    'real_query_candidates': len(validation.real.dataset),
                    'metrics': metrics,
                }
                for role, scores in metrics.items():
                    logging.info('%s/%s: Top-1 %.2f%%  Top-5 %.2f%%  MRR %.4f',
                                 method, role, 100 * scores['top1'], 100 * scores['top5'], scores['mrr'])
                output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
                del validation
        del student, real_encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    logging.info('Saved results: %s  Log: %s', output, log_path)


if __name__ == '__main__':
    main()
