# Tools

Development and debugging utilities.

## Files

- **`inspect_train_data.py`**: Print raw and formatted samples from the datasets to verify pre-processing logic.
- **`verify_code_eval.py`**: Unit tests for the evaluation core (code extraction and execution isolation).
- **`check_dataset_columns.py`**: Minimal script to verify the existence of required keys in loaded datasets.

## Usage

Most tools require the project root to be in `PYTHONPATH`:

```bash
PYTHONPATH=. uv run python tools/inspect_train_data.py
```
