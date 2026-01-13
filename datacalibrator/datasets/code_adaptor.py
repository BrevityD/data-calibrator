from pathlib import Path
from datasets import load_dataset

# ## =======
# current_file_path = Path(__file__).resolve()
# project_root = current_file_path.parents[2]
# import sys
# sys.path.append(str(project_root))
# ## =======

from datacalibrator.seed import set_seed, SEED

def get_code_dataset(size: int=1000):
    # Set global seed
    set_seed(SEED)

    # Get the absolute path of the current file
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    dataset_path = project_root / "datasets" / "code_domain" / "bigcodebench"
    
    # Load the dataset
    # Using streaming=False to allow shuffling and selecting by index
    full_dataset = load_dataset(str(dataset_path), split="v0.1.4", streaming=False)
    
    # Shuffle the dataset with the fixed seed
    shuffled_dataset = full_dataset.shuffle(seed=SEED)
    
    # Select 1000 samples for training and 100 samples for testing
    # Ensure we have enough data
    if len(shuffled_dataset) < size + 100:
        raise ValueError(f"Dataset size ({len(shuffled_dataset)}) is smaller than requested size + test set ({size + 100})")
        
    train_dataset = shuffled_dataset.select(range(size))
    test_dataset = shuffled_dataset.select(range(size, size + 100))

    def preprocess_function(example):
        return {
            "prompt": [{
                "role": "user",
                "content": f"Write a python function to solve following problem. \n{example['instruct_prompt']}"
            }],
            "completion": [{
                "role": "assistant",
                "content": f"{example['code_prompt']}\n{example['canonical_solution']}"
            }],
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_ds, test_ds = get_code_dataset()

    print(train_ds)
    print(train_ds[0])

    print(test_ds)
    print(test_ds[0])
