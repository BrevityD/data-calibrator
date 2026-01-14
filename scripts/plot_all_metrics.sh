#!/bin/bash

RESULTS_DIR="outputs/code_domain/outputs-5e7-bs16-ep3-sgd"
OUTPUT_DIR="${RESULTS_DIR}/plots"

mkdir -p "$OUTPUT_DIR"

echo "Generating plots for results in ${RESULTS_DIR}..."

# Plot Test Accuracy
echo "Plotting Test Accuracy..."
uv run --with matplotlib python tools/plot_metrics.py \
    --results_dir "${RESULTS_DIR}" \
    --split test \
    --output_plot "${OUTPUT_DIR}/accuracy_test.png"

# Plot Train Accuracy
echo "Plotting Train Accuracy..."
uv run --with matplotlib python tools/plot_metrics.py \
    --results_dir "${RESULTS_DIR}" \
    --split train \
    --output_plot "${OUTPUT_DIR}/accuracy_train.png"

# Plot Overall Accuracy
echo "Plotting Overall Accuracy..."
uv run --with matplotlib python tools/plot_metrics.py \
    --results_dir "${RESULTS_DIR}" \
    --split all \
    --output_plot "${OUTPUT_DIR}/accuracy_all.png"

echo "Done. Plots saved to ${OUTPUT_DIR}"
