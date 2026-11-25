"""One-phase inner-face masked iBOT + ArcFace training from official iBOT weights."""

from run_train import ROOT, parse_config


if __name__ == '__main__':
    args = parse_config(ROOT / 'configs/ms1mv3_identity.json')
    if args is not None:
        if args.identity_weight <= 0:
            raise SystemExit('run_identity.py requires identity_weight > 0')
        import main_ibot
        main_ibot.train_ibot(args)
