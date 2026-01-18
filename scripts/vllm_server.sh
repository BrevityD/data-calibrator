#!/bin/bash

MODEL_NAME="Qwen3-8B"
# MODEL_NAME="Qwen3-VL-8B-Instruct"
MODEL_PATH="/public/models/${MODEL_NAME}"

export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1

vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --task "generate" \
    --dtype "bfloat16" \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --host 127.0.0.1 \
    --port 13133