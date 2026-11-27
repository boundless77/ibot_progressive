for dataset in celeba64_std_100c cdfv2_8 vgg2_100p_256; do
    python tools/evaluate_retrieval.py \
        --checkpoints "/mnt/data/fuanlan/project/ibot_progressive/outputs/ms1mv3_stage2/2026-11-17_15-23-34/online_best.pth" \
        --configs "configs/eval/$dataset.yaml" \
        --output "/mnt/data/fuanlan/project/ibot_progressive/outputs/evaluation/$dataset.json"
done
