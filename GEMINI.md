# Data Calibrator Context

## Project Overview

`data-calibrator` is a research framework designed to calibrate the correlation between data domains and model alignment outcomes. It specifically focuses on evaluating Chain-of-Thought (CoT) models (like DeepSeek-R1) on reasoning tasks such as Code, Logic, and Math.

The system features an automated evaluation pipeline that handles:
1.  **Inference**: High-throughput async inference using `vLLM` and `AsyncOpenAI`.
2.  **Processing**: specialized handling for CoT models (stripping `<think>` tags).
3.  **Execution**: Secure, sandboxed code execution for verifying generated solutions.

## Architecture

### 1. Core Logic (`evaluation/`)
The heart of the evaluation engine.
*   **`core.py`**:
    *   `extract_code`: Regex-based extractor that removes `<think>...</think>` blocks to isolate the final solution.
    *   `evaluate_code_correctness`: Runs generated code + unit tests in a `tempfile.TemporaryDirectory` to ensure isolation.
    *   `run_eval`: Main loop managing the async generation and evaluation workflow.
*   **`__main__.py`**: CLI entry point supporting arguments for model path, domain (`code`, `math`, `logic`), and hyperparameters.

### 2. Data Adaptation (`datacalibrator/datasets/`)
Standardizes different benchmarks into a common format for the evaluator.
*   **`code_adaptor.py`**: Adapts `bigcodebench`. Formats prompts with "Write a python function..." and manages train/test splits.
*   **`logic_adaptor.py`**, **`math_adaptor.py`**: Analogous adaptors for other domains.

### 3. Execution Scripts (`scripts/`)
Shell scripts that automate the entire lifecycle (Server Start -> Eval -> Server Stop).
*   **`run_code_eval.sh`**:
    *   Detects GPUs and sets `data-parallel-size`.
    *   Starts a local `vLLM` server in the background.
    *   Uses `trap` to ensure the server is killed when the script exits.
    *   Runs `python -m evaluation`.

## Development & Usage

### Dependencies
The project uses `uv` for dependency management.
```bash
uv sync
source .venv/bin/activate
```

### Running Evaluation
The standard way to run an evaluation is via the shell scripts, which handle the vLLM server automatically.

```bash
# Edit scripts/run_code_eval.sh to set MODEL_PATH
bash scripts/run_code_eval.sh
```

### Manual/Debug Run
You can run the python module directly if you have a server running or for debugging adaptors:

```bash
# Run evaluation (requires active vLLM server at SERVER_URL)
python -m evaluation --action all --domain code --server_url http://127.0.0.1:8000 ...

# Inspect dataset
python datacalibrator/datasets/code_adaptor.py
```

## Key Conventions
*   **CoT Handling**: The system assumes models may produce `<think>` tags. The evaluator actively strips these before processing code.
*   **Sandboxing**: Code is never executed in the project root. Always use `tempfile` contexts.
*   **Async**: Inference is heavily async. Avoid blocking calls in the generation loop.
*   **Formatting**: Code follows standard Python practices. Configuration is largely done via CLI args or shell variables.
