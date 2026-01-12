import sys
import os
from pathlib import Path
import pytest

# Ensure the project root is in sys.path so we can import datacalibrator
# Assuming tests is at data-calibrator/tests
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from datacalibrator.datasets.code_adaptor import get_code_dataset

def test_get_code_dataset_structure():
    """Test if the dataset has the correct structure and size."""
    train_size = 10
    train_ds, test_ds = get_code_dataset(size=train_size)

    assert len(train_ds) == train_size
    assert len(test_ds) == 100

    example = train_ds[0]
    assert "prompt" in example
    assert "completion" in example
    assert "test" in example
    
    assert isinstance(example["prompt"], list)
    assert isinstance(example["completion"], list)
    assert example["prompt"][0]["role"] == "user"
    assert example["completion"][0]["role"] == "assistant"

def test_reproducibility():
    """Test if the dataset generation is reproducible with the fixed seed."""
    train_size = 10
    train_ds_1, _ = get_code_dataset(size=train_size)
    train_ds_2, _ = get_code_dataset(size=train_size)

    # Check if the first item is identical
    assert train_ds_1[0]["prompt"] == train_ds_2[0]["prompt"]
    assert train_ds_1[0]["completion"] == train_ds_2[0]["completion"]
    assert train_ds_1[0]["test"] == train_ds_2[0]["test"]
