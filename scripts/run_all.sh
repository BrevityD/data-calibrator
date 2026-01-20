#!/bin/bash
set -e

# Configuration
BASE_MODEL_DIR="/public/home/dzj/data-calibrator/samples/experiment1/code_domain/outputs-5e7-bs16-ep3-sgd"
PORT=13133
HOST="127.0.0.1"

# Clean up existing results
echo "Cleaning up existing results in outputs/"
# Remove contents of subdirectories but keep the directory structure if possible, 
# or just remove them. Evalscope usually creates them.
rm -rf outputs/logs/* outputs/predictions/* outputs/reports/* outputs/reviews/*
# Ensure directories exist (optional, but good practice)
mkdir -p outputs/logs outputs/predictions outputs/reports outputs/reviews

# Setup cleanup trap to kill vLLM when script exits (e.g. Ctrl+C)
cleanup() {
    echo "Stopping vLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Iterate over checkpoints
for CHECKPOINT_PATH in ${BASE_MODEL_DIR}/checkpoint-*; do
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        continue
    fi

    MODEL_NAME=$(basename "${CHECKPOINT_PATH}")
    echo "========================================================"
    echo "Processing checkpoint: ${MODEL_NAME}"
    echo "Path: ${CHECKPOINT_PATH}"
    echo "========================================================"

    echo "Starting vLLM server for ${MODEL_NAME}..."
    # Start vLLM in a subshell with its own environment
    (
        source .venv/bin/activate
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        export NCCL_P2P_DISABLE=1
        
        # Run vLLM in background
        nohup vllm serve "${CHECKPOINT_PATH}" \
            --served-model-name "${MODEL_NAME}" \
            --task "generate" \
            --dtype "bfloat16" \
            --trust-remote-code \
            --max-model-len 32768 \
            --gpu-memory-utilization 0.7 \
            --data-parallel-size 4 \
            --host ${HOST} \
            --port ${PORT} > outputs/logs/vllm_${MODEL_NAME}.log 2>&1 &
        
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

    # Reset ready flag
    SERVER_READY=0
    while [ $COUNT -lt $MAX_RETRIES ]; do
        if curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null; then
            SERVER_READY=1
            break
        fi
        
        sleep 5
        COUNT=$((COUNT+1))
        echo "Waiting for vLLM... ($COUNT/$MAX_RETRIES)"
        
        # Check if process is still alive
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM server failed to start (Process died). Check outputs/logs/vllm_${MODEL_NAME}.log for details."
            # We don't exit the whole script, maybe just skip this checkpoint? 
            # But the user probably wants to know. Let's try to skip to cleanup.
            break
        fi
    done

    if [ $SERVER_READY -eq 1 ]; then
        echo "vLLM server is ready!"
        
        # Run Evaluation
        echo "Starting evaluation for ${MODEL_NAME}..."
        # Activate evalscope environment inside a subshell to avoid polluting current shell
        (
            source evalscope/.venv/bin/activate

            evalscope eval \
             --model "${MODEL_NAME}" \
             --api-url "http://${HOST}:${PORT}/v1" \
             --api-key EMPTY \
             --datasets bigcodebench \
             --eval-type openai_api \
             --dataset-args '{"bigcodebench":{"subset_list":["out_domain"]}}' \
             --eval-batch-size 36 \
             --repeats 1 \
             --generation-config '{"max_tokens": 16384, "temperature":0.6, "top_p":0.95, "top_k":20}'
        )
        echo "Evaluation finished for ${MODEL_NAME}."
    else
        echo "Timeout waiting for vLLM server for ${MODEL_NAME}."
    fi

    # Stop vLLM for this iteration
    echo "Stopping vLLM server for ${MODEL_NAME}..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    
    # Clear PID variable so trap doesn't try to kill it again if we exit now
    VLLM_PID=""
    
    # Wait a bit for port to be freed
    sleep 5
done

echo "All checkpoints processed."