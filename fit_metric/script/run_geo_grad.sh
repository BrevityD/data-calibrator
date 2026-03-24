#!/bin/bash
# 正式训练：有限差分测地线方向 SFT，与 run_geoguide.sh 参数对齐

cd "$(dirname "$0")/../.."

source /public/home/jza/data_calibrate/data_mixture/.venv/bin/activate

python -m fit_metric.train_with_geo \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-1.7B \
    --dataset_pool_size 1000 \
    --total_train_size 1000 \
    --max_epochs 1000 \
    --rebalance_steps 19 \
    --max_steps 500 \
    --target_loss 0.2 \
    --save_steps 19 \
    --train_device cuda:3 \
    --report_to wandb
