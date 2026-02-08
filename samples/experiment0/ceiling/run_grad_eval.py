from pathlib import Path
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

## =======
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
import sys
sys.path.append(str(project_root))
## =======

from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.evaluators.gradient_evaluator import GradientEvaluator

# Configuration
output_dir_name = "outputs-1_7B-math-ceiling-5e7-bs16-ep15-signsgd"
base_model_path = "/data/pretrained_models/Qwen3-1.7B"
project_name = "data-calibrator-gradient-distance"
run_name = "math-ceiling-signsgd"

# Initialize WandB
wandb.init(
    project=project_name,
    name=run_name,
    id="idcmss11",  # id-ceiling-code-sign-sgd-1.7B-1st.
    resume="allow"
)

# Dataset Configuration
# Map dataset name to its loader function
dataset_loaders = {
    "math": get_math_dataset,
    "code": get_code_dataset
}

# Load Datasets
loaded_datasets = {}
for name, loader in dataset_loaders.items():
    logger.info(f"Loading {name} dataset...")
    # Default size=1000 typically
    train_ds, test_ds = loader()
    loaded_datasets[name] = {"train": train_ds, "test": test_ds}

# Get checkpoints
ckpt_dir = Path(__file__).parent / output_dir_name
if not ckpt_dir.exists():
    logger.error(f"Checkpoint directory not found: {ckpt_dir}")
    sys.exit(1)

checkpoints = sorted(list(ckpt_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
logger.info(f"Found {len(checkpoints)} checkpoints.")

# Tokenizer
logger.info(f"Loading tokenizer from {base_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Iterate through checkpoints
for ckpt_path in checkpoints:
    step = int(ckpt_path.name.split('-')[-1])
    logger.info(f"Evaluating checkpoint: {ckpt_path.name} (Step {step})")
    
    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, 
            # device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
        continue

    evaluator = GradientEvaluator(model, tokenizer)
    
    step_metrics = {}
    
    # Evaluate on each dataset
    for ds_name, ds_dict in loaded_datasets.items():
        # Evaluate on Train
        logger.info(f"Computing gradients on {ds_name} Train set...")
        # Batch size 8 to fit in VRAM with gradients
        train_metrics = evaluator.evaluate(
            ds_dict["train"], 
            batch_size=4, 
            max_length=16384, 
            description=f"{ds_name} Train Grads"
        )
        for k, v in train_metrics.items():
            step_metrics[f"train-{ds_name}/{k}"] = v
        
        # Evaluate on Test
        logger.info(f"Computing gradients on {ds_name} Test set...")
        test_metrics = evaluator.evaluate(
            ds_dict["test"], 
            batch_size=4, 
            max_length=16384, 
            description=f"{ds_name} Test Grads"
        )
        for k, v in test_metrics.items():
            step_metrics[f"test-{ds_name}/{k}"] = v
    
    # Log to WandB
    log_data = {"step": step}
    log_data.update(step_metrics)
        
    wandb.log(log_data)
    logger.info(f"Logged metrics for step {step}: {log_data}")
    
    # Free memory
    del model
    del evaluator
    torch.cuda.empty_cache()

wandb.finish()
logger.info("Evaluation complete.")
