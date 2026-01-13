# Datasets

This directory contains datasets organized by domain.

## Directory Structure

- `code_domain/` - Code-related datasets
- `general_domain/` - General-purpose datasets  
- `logic_domain/` - Logic and reasoning datasets
- `math_domain/` - Mathematics datasets

## Download Datasets

Configure your `hf-token` in `download_data.py`.

To obtain your `hf-token`, visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). To access restricted datasets like `livecodebench`, grant permission for "Read access to contents of all public gated repos you can access".

```console
foo@bar:~/data-calibrator$ source .venv/bin/activate
(.venv)foo@bar:~/data-calibrator$ cd datasets
(.venv)foo@bar:~/data-calibrator/datasets$ python download_data.py
```
