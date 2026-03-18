#!/bin/bash
# 完整训练：1k 数据，多 epoch

cd "$(dirname "$0")/../.."

python -m fit_metric.sft_via_geoguide \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-4B \
    --dataset_pool_size 1000 \
    --total_train_size 1000 \
    --num_epochs 10 \
    --max_steps 401 \
    --save_steps 19 \
    --eval_steps 1 \
    --eval_on_start \
    --train_device cuda:0 \
    --geo_device cuda:4 \
    --report_to wandb
