#!/bin/bash
set -e

# Configuration
MODEL_NAME="Qwen3-8B"
MODEL_PATH="/public/models/${MODEL_NAME}"
PORT=13133
HOST="127.0.0.1"

# Setup cleanup trap to kill vLLM when script exits
cleanup() {
    echo "Stopping vLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID || true
    fi
}
trap cleanup EXIT

echo "Starting vLLM server..."
# Start vLLM in a subshell with its own environment
(
    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    export NCCL_P2P_DISABLE=1
    
    # Run vLLM in background
    nohup vllm serve "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --task "generate" \
        --dtype "bfloat16" \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --host ${HOST} \
        --port ${PORT} > outputs/logs/vllm_run_all.log 2>&1 &
    
    # Save PID to a file to pass it back to parent shell
    echo $! > vllm_pid_tmp
)

# Read the PID from the file
sleep 1
if [ -f vllm_pid_tmp ]; then
    VLLM_PID=$(cat vllm_pid_tmp)
    rm vllm_pid_tmp
    echo "vLLM server PID: $VLLM_PID"
else
    echo "Failed to get vLLM PID."
    exit 1
fi

# Wait for server to be ready
echo "Waiting for vLLM server to be ready at http://${HOST}:${PORT}..."
MAX_RETRIES=120 # Wait up to 10 minutes
COUNT=0

while ! curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null; do
    sleep 5
    COUNT=$((COUNT+1))
    echo "Waiting for vLLM... ($COUNT/$MAX_RETRIES)"
    
    # Check if process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server failed to start (Process died). Check outputs/logs/vllm_run_all.log for details."
        cat outputs/logs/vllm_run_all.log
        exit 1
    fi

    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for vLLM server."
        exit 1
    fi
done
echo "vLLM server is ready!"

# Run Evaluation
echo "Starting evaluation..."
# Activate evalscope environment
source evalscope/.venv/bin/activate

evalscope eval \
 --model "${MODEL_NAME}" \
 --api-url "http://${HOST}:${PORT}/v1" \
 --api-key EMPTY \
 --datasets bigcodebench \
 --eval-type openai_api \
 --generation-config '{"max_tokens": 16384}' \
 --limit 2 \
 --repeats 2 \
 --generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'

echo "Evaluation finished successfully."
