import sys
import os
from pathlib import Path
import pytest

# Ensure the project root is in sys.path so we can import datacalibrator
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from datacalibrator.datasets.logic_adaptor import get_logic_dataset

def test_get_logic_dataset_structure():
    """Test if the dataset has the correct structure and size."""
    train_size = 10
    train_ds, test_ds = get_logic_dataset(size=train_size)

    assert len(train_ds) == train_size
    assert len(test_ds) == 100

    example = train_ds[0]
    assert "question" in example
    assert "query" in example
    assert "chain_of_thought" in example
    assert "prompt" in example
    assert "completion" in example
    
    assert isinstance(example["question"], str)
    assert isinstance(example["query"], str)
    assert isinstance(example["chain_of_thought"], list)
    assert isinstance(example["prompt"], list)
    assert isinstance(example["completion"], list)
    
    assert example["prompt"][0]["role"] == "user"
    assert example["completion"][0]["role"] == "assistant"
    
    # Check if prompt contains the required instructions
    prompt_content = example["prompt"][0]["content"]
    assert "logical reasoning problem" in prompt_content.lower()
    assert "step-by-step proof" in prompt_content.lower()
    assert "instructions:" in prompt_content.lower()
    
    # Check if completion is built from chain_of_thought
    completion_content = example["completion"][0]["content"]
    assert completion_content == "\n".join(example["chain_of_thought"])

def test_reproducibility():
    """Test if the dataset generation is reproducible with the fixed seed."""
    train_size = 10
    train_ds_1, _ = get_logic_dataset(size=train_size)
    train_ds_2, _ = get_logic_dataset(size=train_size)

    # Check if the first item is identical
    assert train_ds_1[0]["question"] == train_ds_2[0]["question"]
    assert train_ds_1[0]["query"] == train_ds_2[0]["query"]
    assert train_ds_1[0]["chain_of_thought"] == train_ds_2[0]["chain_of_thought"]
    assert train_ds_1[0]["prompt"] == train_ds_2[0]["prompt"]
    assert train_ds_1[0]["completion"] == train_ds_2[0]["completion"]

def test_dataset_content():
    """Test that the dataset contains valid content."""
    train_size = 10
    train_ds, test_ds = get_logic_dataset(size=train_size)
    
    # Check that all examples have required fields
    for example in train_ds:
        assert example["question"] != ""
        assert example["query"] != ""
        assert len(example["chain_of_thought"]) > 0
        assert example["prompt"][0]["content"] != ""
        assert example["completion"][0]["content"] != ""
        
        # Check that query starts with "Prove:"
        assert example["query"].startswith("Prove:")
        
        # Check that completion is not empty
        assert len(example["completion"][0]["content"]) > 0

if __name__ == "__main__":
    # Run tests manually if needed
    test_get_logic_dataset_structure()
    test_reproducibility()
    test_dataset_content()
    print("All tests passed!")