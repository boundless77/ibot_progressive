"""Configuration entry point for MS1Mv3 stage 1."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE1_CONFIG = ROOT / 'configs/ms1mv3_stage1.json'

TRAINING_DEFAULTS = {
    # Model and masking
    'arch': 'vit_base',
    'patch_size': 16,
    'out_dim': 8192,
    'patch_out_dim': 8192,
    'shared_head': False,
    'shared_head_teacher': True,
    'norm_last_layer': True,
    'norm_in_head': None,
    'act_in_head': 'gelu',
    'use_masked_im_modeling': True,
    'drop_path': 0.1,
    'visible_decoder': False,
    'decoder_depth': 2,
    'decoder_width': 256,
    'pred_start_epoch': 0,
    'lambda1': 1.0,
    'lambda2': 1.0,
    # Teacher temperatures and momentum
    'warmup_teacher_temp': 0.04,
    'teacher_temp': 0.04,
    'warmup_teacher_patch_temp': 0.04,
    'teacher_patch_temp': 0.07,
    'warmup_teacher_temp_epochs': 30,
    'momentum_teacher': 0.996,
    # Optimization
    'use_fp16': True,
    'weight_decay': 0.04,
    'weight_decay_end': 0.4,
    'clip_grad': 3.0,
    'batch_size_per_gpu': 128,
    'epochs': 100,
    'freeze_last_layer': 1,
    'lr': 0.0005,
    'warmup_epochs': 10,
    'min_lr': 1e-6,
    'optimizer': 'adamw',
    # Multi-crop
    'global_crops_number': 2,
    'global_crops_scale': (0.14, 1.0),
    'local_crops_number': 0,
    'local_crops_scale': (0.05, 0.4),
    # Data, logging and distributed training
    'data_path': None,
    'output_dir': '.',
    'saveckp_freq': 40,
    'seed': 0,
    'num_workers': 10,
    'dist_url': 'env://',
    'local_rank': 0,
    'verbose': 2000,
    'mask_mode': 'progressive',
    'identity_weight': 0.0,
    'arcface_margin': 0.5,
    'arcface_scale': 64.0,
    'validation_config': '',
}

# Older project configs can contain these unused upstream options.
LEGACY_OPTIONS = ('window_size', 'pred_ratio', 'pred_ratio_var', 'pred_shape', 'load_from')


def read_stage1_config(path=DEFAULT_STAGE1_CONFIG, overrides=None):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for name in ('data_path', 'parsing_root', 'pretrained', 'resume', 'output_dir',
                 'blendface_masks', 'packed_mask_dir', 'validation_config'):
        if config.get(name):
            config[name] = str((ROOT / Path(config[name]).expanduser()).resolve())
    return config


def parse_config(default_config=DEFAULT_STAGE1_CONFIG):
    cli = argparse.ArgumentParser()
    cli.add_argument('--config', default=str(default_config))
    for name in ('data_path', 'parsing_root', 'pretrained', 'resume', 'output_dir', 'packed_mask_dir', 'blendface_masks', 'validation_config'):
        cli.add_argument('--' + name)
    cli.add_argument('--batch_size_per_gpu', type=int)
    cli.add_argument('--num_workers', type=int)
    cli.add_argument('--lr', type=float)
    cli.add_argument('--verbose', type=int)
    cli.add_argument('--check_only', action='store_true')
    cli.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parsed = cli.parse_args()
    keys = ('data_path', 'parsing_root', 'pretrained', 'resume', 'output_dir',
            'batch_size_per_gpu', 'num_workers', 'lr', 'verbose', 'packed_mask_dir', 'blendface_masks', 'validation_config')
    overrides = {name: getattr(parsed, name) for name in keys}
    if parsed.resume and parsed.pretrained is None:
        overrides['pretrained'] = ''
    config = read_stage1_config(parsed.config, overrides)
    identity = config.get('identity_weight', 0) > 0
    required = ('data_path',) + (() if identity else ('parsing_root',))
    required += ('resume',) if config.get('resume') else ('pretrained',)
    missing = [k for k in required if not config.get(k) or not Path(config[k]).exists()]
    if missing:
        if 'pretrained' in missing:
            cli.exit(2, 'Missing iBOT ViT-B/16 Full Checkpoint. See the download link in README.md. '
                     f"Expected checkpoint: {config.get('pretrained') or '<set pretrained in config>'}. "
                     'Missing: ' + ', '.join(missing) + '\n')
        raise SystemExit('Please set existing paths in config or CLI: ' + ', '.join(missing))
    if config.get('resume') and config.get('pretrained'):
        raise SystemExit('Use only pretrained (new experiment) OR resume (this experiment)')
    if config.get('resume') and parsed.output_dir is None:
        config['output_dir'] = str(Path(config['resume']).parent)
    if config.get('verbose', 2000) < 1 or config.get('lr', 5e-5) <= 0:
        raise SystemExit('verbose and lr must be positive')
    if config.get('validation_config') and not Path(config['validation_config']).is_file():
        cli.error('validation_config does not exist: ' + config['validation_config'])
    data_root = Path(config['data_path'])
    if not all((data_root / name).is_file() for name in ('train.idx', 'train.rec')):
        cli.error('data_path must contain MS1Mv3 train.idx and train.rec')
    if identity:
        unused = set(config) & {'parsing_root', 'packed_mask_dir'}
        if unused:
            cli.error('Identity training no longer uses stage1 index/mask paths; remove: ' + ', '.join(sorted(unused)))
        if not config.get('blendface_masks'):
            cli.error('Identity experiment requires blendface_masks pointing to masks_packed')
        blend_masks = Path(config['blendface_masks'])
        required_cache = [data_root / 'idx2label.json', blend_masks / 'meta.pkl', blend_masks / 'bits.npy']
    else:
        cache_root = Path(config['parsing_root'])
        packed_root = Path(config['packed_mask_dir']) if config.get('packed_mask_dir') else cache_root / 'packed-masks'
        required_cache = [cache_root / 'index.json', cache_root / 'valid-records.npy',
                          packed_root / 'progress.json', packed_root / 'masks.npy']
    missing_cache = [str(path) for path in required_cache if not path.is_file()]
    if missing_cache:
        cli.error('Missing data/index files: ' + ', '.join(missing_cache))
    if config.get('mask_mode', 'progressive') not in ('progressive', 'inner'):
        cli.error('mask_mode must be progressive or inner')
    if identity:
        if (config.get('mask_mode') != 'inner' or config.get('visible_decoder', False)
                or config.get('local_crops_number', 0) != 0
                or config.get('global_crops_number', 2) != 2
                or config.get('require_full_organs', True)
                or list(config.get('global_crops_scale', (1, 1))) != [1, 1]):
            cli.error('Identity experiment requires binary BlendFace inner mask, require_full_organs=false, original iBOT, two full global views and no local views')
        if config.get('stages'):
            cli.error('Identity experiment does not use a progressive schedule')
    else:
        if config.get('mask_mode', 'progressive') != 'progressive':
            cli.error('inner mask is reserved for the identity experiment')
        from progressive.masking import ProgressiveCRFRPMask
        generator = ProgressiveCRFRPMask(config['stages'], config['patch_size'])
        if config['epochs'] != generator.stages[-1].end_epoch:
            raise SystemExit('epochs must equal final stage end_epoch')
    if config['pred_start_epoch'] != 0 or config['lambda1'] != 1 or config['lambda2'] != 1:
        raise SystemExit('Keep original CLS + MIM active with unit weights for this experiment')
    if config.get('visible_decoder'):
        if any(stage.ratio != 0.75 for stage in generator.stages):
            raise SystemExit('The visible decoder experiment requires a fixed 75% mask budget')
        depth, width = config.get('decoder_depth', 2), config.get('decoder_width', 256)
        if depth < 1 or width < 4 or width % 4:
            raise SystemExit('Decoder needs positive depth and width divisible by four')
    for key in LEGACY_OPTIONS:
        config.pop(key, None)
    unknown = set(config) - TRAINING_DEFAULTS.keys() - {
        'parsing_root', 'pretrained', 'resume', 'stages', 'require_full_organs',
        'blendface_masks', 'packed_mask_dir',
    }
    if unknown:
        cli.error('Unknown config keys: ' + str(sorted(unknown)))
    if config.get('optimizer', TRAINING_DEFAULTS['optimizer']) != 'adamw':
        cli.error('This experiment uses optimizer=adamw')
    if parsed.check_only:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return None
    args = argparse.Namespace(**(TRAINING_DEFAULTS | config))
    args.local_rank = int(os.environ.get('LOCAL_RANK', parsed.local_rank))
    return args

if __name__ == '__main__':
    args = parse_config()
    if args is not None:
        import main_ibot
        main_ibot.train_ibot(args)
