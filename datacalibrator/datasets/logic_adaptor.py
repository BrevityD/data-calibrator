from pathlib import Path
from datasets import Dataset
import json
import glob

# ## =======
# current_file_path = Path(__file__).resolve()
# project_root = current_file_path.parents[2]
# import sys
# sys.path.append(str(project_root))
# ## =======

from datacalibrator.seed import set_seed, SEED

def get_logic_dataset(size: int=1000):
    # Set global seed
    set_seed(SEED)

    # Get the absolute path of the current file
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    dataset_path = project_root / "datasets" / "logic_domain" / "prontoqa"
    
    # Find all JSON files in the dataset directory
    json_files = glob.glob(str(dataset_path / "*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {dataset_path}")
    
    # Load and flatten all examples from all JSON files
    train_examples = []
    test_examples = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Flatten the structure: each example has 9 sub-examples (8 in_context + 1 test)
        for example_key, example_data in data.items():
            for sub_key, sub_data in example_data.items():
                if "test" not in sub_key:
                    train_examples.append(sub_data)
                else:
                    test_examples.append(sub_data)

    # Create dataset from flattened examples
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    # Shuffle the dataset with the fixed seed
    shuffled_train_dataset = train_dataset.shuffle(seed=SEED)
    shuffled_test_dataset = test_dataset.shuffle(seed=SEED)
    
    # Select samples for training and testing
    # Ensure we have enough data
    if len(shuffled_train_dataset) < size:
        raise ValueError(f"Dataset size ({len(shuffled_train_dataset)}) is smaller than \
                         requested train set size")
    if len(shuffled_test_dataset) < 100:
        raise ValueError(f"Dataset size ({len(shuffled_test_dataset)}) is smaller than \
                         requested test set size")

    train_dataset = shuffled_train_dataset.select(range(size))
    test_dataset = shuffled_test_dataset.select(range(100))

    def preprocess_function(example):
        # Build the prompt with instructions
        prompt = f"""You are given a logical reasoning problem.
Please provide a step-by-step proof.

Problem:
{example['question']}

Goal:
{example['query']}

Instructions:
1. Provide a step-by-step logical proof. and each step should be wrapped in <step></step>
2. Each step should be clearly explained.
3. End with the final conclusion that matches the goal.

Please write your proof below:"""

        completion = ""
        for cot in example['chain_of_thought']:
            if "since" not in cot:
                completion += "<step>" + cot.strip() + "</step>\n"

        return {
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [
                {"role": "assistant", "content": completion}
            ],
            "goal": example['query'].replace("Prove:", "").strip()
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_ds, test_ds = get_logic_dataset()

    print(train_ds)
    print(train_ds[0])

    print(test_ds)
    print(test_ds[0])
