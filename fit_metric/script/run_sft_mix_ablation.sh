#!/bin/bash
# 消融实验：纯 math vs 9:1 混合配比 SFT，eval loss 曲线对比

cd "$(dirname "$0")/../.."

source /public/home/jza/data_calibrate/data_mixture/.venv/bin/activate

python -m fit_metric.sft_mix_ablation \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-1.7B \
    --dataset_pool_size 1000 \
    --train_size 1000 \
    --mix_ratio 0.9 \
    --max_steps 100 \
    --eval_domain math \
    --eval_steps 1 \
    --train_device cuda:0 \
    --report_to wandb
