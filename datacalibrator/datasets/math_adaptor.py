from pathlib import Path
from datasets import load_dataset

# ## =======
# current_file_path = Path(__file__).resolve()
# project_root = current_file_path.parents[2]
# import sys
# sys.path.append(str(project_root))
# ## =======

from datacalibrator.seed import set_seed, SEED

def find_box(pred_str: str) -> str:
    """
    从包含 "boxed" 的字符串中提取答案
    Args:
        pred_str: 预测字符串
    Returns:
        提取的答案字符串
    """
    if "boxed" not in pred_str:
        return ""
    
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    
    return a.strip()

def get_math_dataset(size: int=1000):
    # Set global seed
    set_seed(SEED)

    # Get the absolute path of the current file
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    dataset_path = project_root / "datasets" / "math_domain" / "MATH-Hard"
    
    # Load the train and test datasets
    train_full = load_dataset(str(dataset_path), split="train", streaming=False)
    test_full = load_dataset(str(dataset_path), split="test", streaming=False)
    
    # Shuffle the datasets with the fixed seed
    train_shuffled = train_full.shuffle(seed=SEED)
    test_shuffled = test_full.shuffle(seed=SEED)
    
    # Select samples
    if len(train_shuffled) < size:
        raise ValueError(f"Train dataset size ({len(train_shuffled)}) is smaller than \
                         requested size ({size})")
    
    test_size = 100
    if len(test_shuffled) < test_size:
        raise ValueError(f"Test dataset size ({len(test_shuffled)}) is smaller than \
                         requested size ({test_size})")
        
    train_dataset = train_shuffled.select(range(size))
    test_dataset = test_shuffled.select(range(test_size))

    def preprocess_function(example):
        ground_truth = find_box(example['solution'])
        return {
            "prompt": [{"role": "user", "content": f"{example['problem']}\n\
                        Please enclose the final answer in \boxed{{}}."}],
            "completion": [
                {"role": "assistant", "content": example['solution']}
            ],
            "ground_truth": ground_truth
        }

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_ds, test_ds = get_math_dataset()

    print(train_ds)
    print(train_ds[0])

    print(test_ds)
    print(test_ds[0])
