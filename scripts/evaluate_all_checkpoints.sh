#!/bin/bash

# Configuration
BASE_DIR="/public/home/dzj/data-calibrator/samples/experiment1/code_domain/outputs-5e7-bs16-ep3-sgd"
OUTPUT_BASE="outputs/code_domain/outputs-5e7-bs16-ep3-sgd"
HOST="127.0.0.1"
PORT="24444"
SERVER_URL="http://${HOST}:${PORT}"
SPLIT="${1:-all}" # Default to all if not provided

# Server Configuration (Matching original script)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Detected ${NUM_DEVICES} devices: ${CUDA_VISIBLE_DEVICES}"
echo "Checkpoints directory: ${BASE_DIR}"
echo "Output directory: ${OUTPUT_BASE}"

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Get list of checkpoints and sort strictly numerically (e.g., checkpoint-2 before checkpoint-10)
# We accept both 'checkpoint-N' formats.
# sort -V handles version sorting which works for checkpoint-1, checkpoint-2, checkpoint-10
CHECKPOINTS=$(ls "$BASE_DIR" | grep "checkpoint-" | sort -V)

# Function to kill server safely
cleanup_server() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping server process $pid..."
        kill -TERM "$pid"
        
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo "Server did not stop gracefully. Force killing..."
            kill -9 "$pid"
        fi
    fi
}

# Trap to ensure cleanup on script exit or interruption
trap 'if [ -n "$SERVER_PID" ]; then cleanup_server $SERVER_PID; fi; exit' INT TERM EXIT

for CHECKPOINT in $CHECKPOINTS; do
    MODEL_PATH="${BASE_DIR}/${CHECKPOINT}"
    MODEL_NAME="${CHECKPOINT}"
    # Use deterministic subfolder name for resuming capabilities
    CHECKPOINT_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/code_${SPLIT}"
    LOG_FILE="${OUTPUT_BASE}/${MODEL_NAME}_server.log"
    
    echo "=================================================="
    echo "Processing ${MODEL_NAME}..."
    echo "=================================================="
    
    # Check if this checkpoint was already evaluated (Resume Logic)
    if [ -f "${CHECKPOINT_OUTPUT_DIR}/eval_results.jsonl" ]; then
        echo "Found existing evaluation results at ${CHECKPOINT_OUTPUT_DIR}/eval_results.jsonl"
        echo "Skipping ${MODEL_NAME}."
        continue
    fi
    
    echo "Starting vLLM server for ${MODEL_NAME}..."
    # Start server in background, redirecting logs
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
        --port "${PORT}" > "${LOG_FILE}" 2>&1 &
        
    SERVER_PID=$!
    echo "Server started with PID: ${SERVER_PID}. Logs at ${LOG_FILE}"
    
    # Wait for server to be ready
    echo "Waiting for server health check..."
    MAX_RETRIES=600 # 10 minutes wait max (loading models can be slow)
    COUNT=0
    SERVER_READY=false
    
    while [ $COUNT -lt $MAX_RETRIES ]; do
        if curl -s "${SERVER_URL}/health" > /dev/null; then
            SERVER_READY=true
            break
        fi
        
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process crashed unexpectedly."
            tail -n 20 "${LOG_FILE}"
            break
        fi
        
        sleep 1
        COUNT=$((COUNT+1))
        
        if [ $((COUNT % 30)) -eq 0 ]; then
            echo "Still waiting... (${COUNT}s)"
        fi
    done
    
    if [ "$SERVER_READY" = true ]; then
        echo "Server is ready! Running generation..."
        
        # 1. Run Generation ONLY (Blocking, keeps GPU busy)
        PYTHONPATH=. uv run python -m evaluation \
            --action generate \
            --domain code \
            --server_url "${SERVER_URL}" \
            --split "${SPLIT}" \
            --output_dir "${CHECKPOINT_OUTPUT_DIR}" \
            --model "${MODEL_NAME}" \
            --max_tokens 16384 \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k 20 \
            --no_timestamp
            
        GEN_EXIT_CODE=$?
        
        # 2. Stop Server IMMEDIATELY to free GPU for next checkpoint
        echo "Generation finished. Stopping server ${SERVER_PID} to release GPU..."
        cleanup_server $SERVER_PID
        
        if [ $GEN_EXIT_CODE -eq 0 ]; then
            echo "Generation successful. Starting evaluation in background..."
            
            # 3. Run Evaluation in BACKGROUND (Non-blocking, uses CPU)
            # We redirect output to a log file to avoid cluttering the main terminal
            PYTHONPATH=. uv run python -m evaluation \
                --action evaluate \
                --domain code \
                --input_file "${CHECKPOINT_OUTPUT_DIR}/generated_responses.jsonl" \
                --output_dir "${CHECKPOINT_OUTPUT_DIR}" \
                --split "${SPLIT}" \
                --no_timestamp > "${CHECKPOINT_OUTPUT_DIR}/eval_process.log" 2>&1 &
                
            echo "Evaluation job for ${MODEL_NAME} submitted to background."
        else
            echo "Generation failed with exit code ${GEN_EXIT_CODE}. Skipping evaluation."
        fi
    else
        echo "Timeout or crash waiting for server for ${MODEL_NAME}. Skipping."
        # Ensure server is killed if it failed to start properly
        cleanup_server $SERVER_PID
    fi
    
    # Reset PID variable so trap doesn't try to kill it again later
    SERVER_PID=""
    
    # Extra safety: ensure port is free
    echo "Waiting for port release..."
    sleep 5
    
done

# Clear the trap before exiting normally to avoid double cleanup
trap - INT TERM EXIT
echo "All checkpoints processed (Evaluation may still be running in background)."
echo "Check individual eval_process.log files for status."
