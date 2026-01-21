from pathlib import Path
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

## =======
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
import sys
sys.path.append(str(project_root))
## =======

from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.evaluators.gradient_evaluator import GradientEvaluator

# Configuration
output_dir_name = "outputs-1_7B-code-ceiling-5e7-bs16-ep15-signsgd"
base_model_path = "/models/Qwen3-1.7B"
project_name = "data-calibrator-exp0"
run_name = "grad-eval-code-ceiling-signsgd"

# Initialize WandB
wandb.init(project=project_name, name=run_name)

# Load Dataset
print("Loading dataset...")
# Default size=1000
train_dataset, test_dataset = get_code_dataset() 

# Get checkpoints
ckpt_dir = Path(__file__).parent / output_dir_name
if not ckpt_dir.exists():
    print(f"Checkpoint directory not found: {ckpt_dir}")
    sys.exit(1)

checkpoints = sorted(list(ckpt_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
print(f"Found {len(checkpoints)} checkpoints.")

# Tokenizer
print(f"Loading tokenizer from {base_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Iterate through checkpoints
for ckpt_path in checkpoints:
    step = int(ckpt_path.name.split('-')[-1])
    print(f"\n=== Evaluating checkpoint: {ckpt_path.name} (Step {step}) ===")
    
    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        continue

    evaluator = GradientEvaluator(model, tokenizer)
    
    # Evaluate on Train
    print("  Computing gradients on Train set...")
    # Use a smaller subset if needed, but for now we try full 1000
    # Batch size 16 to fit in VRAM with gradients
    train_metrics = evaluator.evaluate(train_dataset, batch_size=16, description="Train Grads")
    
    # Evaluate on Test
    print("  Computing gradients on Test set...")
    test_metrics = evaluator.evaluate(test_dataset, batch_size=16, description="Test Grads")
    
    # Log to WandB
    log_data = {"step": step}
    for k, v in train_metrics.items():
        log_data[f"train/{k}"] = v
    for k, v in test_metrics.items():
        log_data[f"test/{k}"] = v
        
    wandb.log(log_data)
    print(f"  Logged metrics for step {step}")
    
    # Free memory
    del model
    del evaluator
    torch.cuda.empty_cache()

wandb.finish()
print("Evaluation complete.")
