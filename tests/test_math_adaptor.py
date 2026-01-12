import sys
import os
from pathlib import Path
import pytest

# Ensure the project root is in sys.path so we can import datacalibrator
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from datacalibrator.datasets.math_adaptor import get_math_dataset, find_box

def test_find_box():
    """Test the find_box function."""
    assert find_box("The answer is \\boxed{42}.") == "42"
    assert find_box("The answer is \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert find_box("No box here") == ""
    assert find_box("Nested \\boxed{a{b}c}") == "a{b}c"

def test_get_math_dataset_structure():
    """Test if the dataset has the correct structure and size."""
    train_size = 10
    train_ds, test_ds = get_math_dataset(size=train_size)

    assert len(train_ds) == train_size
    assert len(test_ds) == 100

    example = train_ds[0]
    assert "prompt" in example
    assert "completion" in example
    assert "ground_truth" in example
    
    assert isinstance(example["prompt"], list)
    assert isinstance(example["completion"], list)
    assert example["prompt"][0]["role"] == "user"
    assert example["completion"][0]["role"] == "assistant"
    
    # Check if prompt contains the instruction
    assert "Please enclose the final answer in \boxed{}." in example["prompt"][0]["content"]

def test_reproducibility():
    """Test if the dataset generation is reproducible with the fixed seed."""
    train_size = 10
    train_ds_1, _ = get_math_dataset(size=train_size)
    train_ds_2, _ = get_math_dataset(size=train_size)

    # Check if the first item is identical
    assert train_ds_1[0]["prompt"] == train_ds_2[0]["prompt"]
    assert train_ds_1[0]["completion"] == train_ds_2[0]["completion"]
    assert train_ds_1[0]["ground_truth"] == train_ds_2[0]["ground_truth"]
    assert train_ds_1[0]["ground_truth"] != ""
