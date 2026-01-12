# data-calibrator

Experiments on calibrating the correlation between data domains and model alignment outcomes.

## Usage

### Installation

Install uv before proceeding.

```console
foo@bar:~$ git clone git@github.com:BrevityD/data-calibrator.git
foo@bar:~$ cd data-calibrator
foo@bar:~/data-calibrator$ uv venv .venv
foo@bar:~/data-calibrator$ uv sync
```

Activate your virtual environment:

```console
foo@bar:~/data-calibrator$ source .venv/bin/activate
```

Log in to `WandB` (optional):

Skip this step if not using WandB.

### Reproduce Samples

Samples:
1. [EXP-1](./samples/experiment1/README.md)

### Download Datasets

Configure your `hf-token` in `data-calibrator/datasets/download_data.py`.

To obtain your `hf-token`, visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). To access restricted datasets like `livecodebench`, grant permission for "Read access to contents of all public gated repos you can access".

```console
foo@bar:~/data-calibrator$ source .venv/bin/activate
(.venv)foo@bar:~/data-calibrator$ cd datasets
(.venv)foo@bar:~/data-calibrator/datasets$ python download_data.py
```
