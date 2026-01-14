#!/bin/bash

# Configuration
BASE_DIR="/public/home/dzj/data-calibrator/samples/experiment1/code_domain/outputs-5e7-bs16-ep3-sgd"
OUTPUT_BASE="outputs/code_domain/outputs-5e7-bs16-ep3-sgd"
SPLIT="${1:-all}" # Default to all

echo "Running Metric Computation ONLY (No generation)..."
echo "Split: ${SPLIT}"

CHECKPOINTS=$(ls "$BASE_DIR" | grep "checkpoint-" | sort -V)

for CHECKPOINT in $CHECKPOINTS; do
    MODEL_NAME="${CHECKPOINT}"
    # Path where generation script put the files
    CHECKPOINT_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}/code_${SPLIT}"
    GENERATED_FILE="${CHECKPOINT_OUTPUT_DIR}/generated_responses.jsonl"
    EVAL_RESULTS_FILE="${CHECKPOINT_OUTPUT_DIR}/eval_results.jsonl"
    
    echo "Processing ${MODEL_NAME}..."
    
    if [ ! -f "$GENERATED_FILE" ]; then
        echo "Skipping ${MODEL_NAME}: No generated responses found at ${GENERATED_FILE}"
        continue
    fi
    
    # Check if already evaluated (optional, comment out if you want to force re-eval)
    if [ -f "$EVAL_RESULTS_FILE" ]; then
        echo "Skipping ${MODEL_NAME}: Evaluation results already exist."
        continue
    fi

    echo "Evaluating ${GENERATED_FILE}..."
    
    PYTHONPATH=. uv run python -m evaluation \
        --action evaluate \
        --domain code \
        --input_file "${GENERATED_FILE}" \
        --output_dir "${CHECKPOINT_OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --no_timestamp
        
    echo "Done."
done

echo "All metrics computed."
