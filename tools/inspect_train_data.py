import sys
import os
from pathlib import Path
import json

# Add project root to sys.path so we can import datacalibrator
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
sys.path.append(str(project_root))

from datacalibrator.datasets.code_adaptor import get_code_dataset

def main():
    print("Loading dataset...")
    # This uses the same logic as your training and eval scripts
    train_ds, test_ds = get_code_dataset()
    
    print(f"\nTraining Dataset Size: {len(train_ds)}")
    print(f"Testing Dataset Size: {len(test_ds)}")
    
    print("\n" + "="*80)
    print("INSPECTING TRAINING SAMPLES (First 2 examples)")
    print("="*80)

    # Inspect the 5th sample ONLY
    for i in range(5, 6):
        sample = train_ds[i]
        print(f"\n>>> Sample Index: {i}")
        
        all_keys = list(sample.keys())
        print(f"\n[All Keys Available]: {all_keys}")
        
        for key in all_keys:
            print(f"\n--- FIELD: {key} ---")
            value = sample[key]
            
            # Pretty print complex types (dict, list)
            if isinstance(value, (dict, list)):
                print(json.dumps(value, indent=2, ensure_ascii=False))
            else:
                # For long strings, maybe truncation isn't desired if we want to see EVERYTHING,
                # but let's just print it raw.
                print(value)
        
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()
