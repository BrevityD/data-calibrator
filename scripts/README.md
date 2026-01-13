# Scripts

This directory contains shell script entry points for automating multi-step workflows like model serving and evaluation.

## Files

- **`run_code_eval.sh`**: 
  - Starts a vLLM server in the background.
  - Waits for health check to pass.
  - Triggers the `evaluation` module with specific sampling parameters.
  - Automatically cleans up the server process on exit.