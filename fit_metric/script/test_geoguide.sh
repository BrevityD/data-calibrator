#!/bin/bash
# 快速测试 sft_via_geoguide.py 参数是否正确，小数据量 + 少步数

cd "$(dirname "$0")/../.."

source /public/home/jza/data_calibrate/data_mixture/.venv/bin/activate

python -m fit_metric.sft_via_geoguide \
    --base_model_path /public/home/jza/share_model/Qwen/Qwen3-1.7B \
    --dataset_pool_size 200 \
    --total_train_size 100 \
    --max_segments 2 \
    --rebalance_steps 19 \
    --max_steps 10 \
    --save_steps 10 \
    --eval_steps 999 \
    --no-eval_on_start \
    --train_device cuda:4 \
    --geo_device cuda:5 \
    --report_to none
