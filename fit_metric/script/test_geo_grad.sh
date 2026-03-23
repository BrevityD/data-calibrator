#!/bin/bash
# 快速测试 train_with_geo.py 参数是否正确，小数据量 + 少 epoch

cd "$(dirname "$0")/../.."

source /public/home/jza/data_calibrate/data_mixture/.venv/bin/activate

python -m fit_metric.train_with_geo \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-1.7B \
    --dataset_pool_size 200 \
    --max_epochs 3 \
    --per_device_train_batch_size 2 \
    --save_steps 1 \
    --train_device cuda:3 \
    --report_to none
