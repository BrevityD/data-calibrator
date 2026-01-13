#!/bin/bash

# This script starts a vLLM server, runs the evaluation, and then stops the server.

# Server Configuration
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
# Calculate number of devices from the comma-separated list
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
echo "Detected ${NUM_DEVICES} devices: ${CUDA_VISIBLE_DEVICES}"

MODEL_NAME="checkpoint-10"
MODEL_PATH="/public/home/dzj/data-calibrator/samples/experiment1/code_domain/outputs-1e5/${MODEL_NAME}"
HOST="127.0.0.1"
PORT="24444"
SERVER_URL="http://${HOST}:${PORT}"

# Start the vLLM server in the background
echo "Starting vLLM server..."                    
uv run vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --task "generate" \
    --dtype "auto" \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --data-parallel-size ${NUM_DEVICES} \
    --host "${HOST}" \
    --port "${PORT}" &

SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"

# Trap to kill server on script exit
trap 'echo "Stopping server..."; kill ${SERVER_PID};' EXIT

# Wait for the server to be ready
echo "Waiting for server to be ready at ${SERVER_URL}/health..."
while ! curl -s "${SERVER_URL}/health" > /dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null;
 then
        echo "Server process crashed. Exiting."
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo "Server is ready!"

# Run the evaluation (Decoupled: action=all runs both generation and evaluation)
echo "Running evaluation..."
PYTHONPATH=. uv run python -m evaluation \
    --action all \
    --domain code \
    --server_url "${SERVER_URL}" \
    --split test \
    --output_dir outputs \
    --model "${MODEL_NAME}" \
    --max_tokens 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20

# Example of running ONLY evaluation on existing results:
# PYTHONPATH=. uv run python datacalibrator/eval.py \
#     --action evaluate \
#     --domain code \
#     --input_file outputs/code_train_TIMESTAMP/generated_responses.jsonl \
#     --output_dir outputs

echo "Evaluation process completed."

