# Scripts

This directory contains utility scripts for the data-calibrator project.

## Files

- **`check_dataset_columns.py`**:
  - Inspects the columns and content of the code dataset (`bigcodebench`) to ensure keys like `entry_point` and `test` are present.
  - Usage: `uv run python scripts/check_dataset_columns.py`

- **`run_code_eval.sh`**:
  - Shell script to launch the evaluation of a model on the code domain.
  - Configured to use vLLM with data parallelism on GPUs 4, 5, 6, and 7.
  - Usage: `bash scripts/run_code_eval.sh` (run from the project root to ensure `datacalibrator` module is found).

- **`verify_code_eval.py`**:
  - A verification script to test the code extraction and execution logic used in `datacalibrator/eval.py`.
  - Ensures that correct code passes, incorrect code fails, and syntax errors are handled gracefully.
  - Usage: `uv run python scripts/verify_code_eval.py`
