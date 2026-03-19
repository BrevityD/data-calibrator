#!/bin/bash
# 完整训练：1k 数据，按步数重新配比

cd "$(dirname "$0")/../.."

python -m fit_metric.sft_via_geoguide \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-1.7B \
    --dataset_pool_size 1000 \
    --total_train_size 1000 \
    --max_segments 1000 \
    --rebalance_steps 20 \
    --max_steps 401 \
    --target_loss 0.2 \
    --save_steps 19 \
    --eval_steps 999 \
    --no-eval_on_start \
    --train_device cuda:4 \
    --geo_device cuda:5 \
    --report_to wandb
