import os
import sys
import json
import argparse
import re

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import wandb
from loguru import logger
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

# Setup path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
sys.path.append(str(project_root))

from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED

def parse_args():
    parser = argparse.ArgumentParser(description="Run Heatmap Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (json)")
    return parser.parse_args()

class ExperimentCallback(TrainerCallback):
    def __init__(self, target_loss, min_steps):
        # min_steps的作用：
        # 1. 保证训练至少进行这么多步，即使loss已经达标。
        # 2. 作为一个固定的观测点，记录该步数下的loss和grad_norm，用于构建Heatmap（即固定X轴为min_steps时的Y轴数值）。
        self.target_loss = target_loss
        self.min_steps = min_steps
        self.steps_at_target_loss = None
        self.loss_at_min_steps = None
        self.grad_norm_at_min_steps = None
        self.results = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # We care about eval_loss for stopping condition
        current_eval_loss = logs.get("eval_loss")
        current_grad_norm = logs.get("grad_norm") # This might be None in eval logs
        
        # We also want to track training loss if available
        current_train_loss = logs.get("loss")
        
        # Determine the loss we are checking against target
        # The user requested stopping when eval_loss reaches target
        
        if current_eval_loss is not None:
             # 1. Record steps when eval_loss <= target_loss (first time)
            if self.steps_at_target_loss is None and current_eval_loss <= self.target_loss:
                self.steps_at_target_loss = state.global_step
                logger.info(f"Target eval_loss {self.target_loss} reached at step {state.global_step}")

            # Stopping condition: Eval Loss <= Target AND Steps >= Min
            if state.global_step >= self.min_steps and current_eval_loss <= self.target_loss:
                logger.info(f"Stopping criteria met at step {state.global_step} (Eval Loss: {current_eval_loss} <= {self.target_loss})")
                control.should_training_stop = True

        # 2. Record loss/grad_norm at min_steps
        # We capture this when we pass the step. Ideally we want to capture eval loss at this point too?
        # But this callback might run on train log or eval log.
        if state.global_step >= self.min_steps and self.loss_at_min_steps is None:
             # If we have loss in this log (could be train or eval), record it.
             # Ideally we'd want specific metrics but let's grab what's available
            if current_eval_loss is not None:
                self.loss_at_min_steps = current_eval_loss
                logger.info(f"Reached min_steps {self.min_steps}. Eval Loss: {current_eval_loss}")
            elif current_train_loss is not None:
                 self.loss_at_min_steps = current_train_loss
                 logger.info(f"Reached min_steps {self.min_steps}. Train Loss: {current_train_loss}")
            
            if current_grad_norm is not None:
                self.grad_norm_at_min_steps = current_grad_norm

            
    def on_train_end(self, args, state, control, **kwargs):
        # Final cleanup if we stopped early or finished
        self.results = {
            "steps_to_reach_target_loss": self.steps_at_target_loss,
            "loss_at_min_steps": self.loss_at_min_steps,
            "grad_norm_at_min_steps": self.grad_norm_at_min_steps,
            "total_steps": state.global_step,
            # 为什么是空的
            "final_loss": state.log_history[-1].get("eval_loss", state.log_history[-1].get("loss")) if state.log_history else None
        }

def get_checkpoints(source_dir, start_idx, end_idx):
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist")
    
    checkpoints = []
    # Pattern to match checkpoint-N
    pattern = re.compile(r"checkpoint-(\d+)")
    
    for item in source_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                cp_idx = int(match.group(1))
                if start_idx <= cp_idx <= end_idx:
                    checkpoints.append((cp_idx, item))
    
    # Sort by index
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    source_model_dir = config["source_model_dir"]
    checkpoint_range = config["checkpoint_range"] # [start, end]
    target_dataset_name = config["target_dataset"] # "math" or "code"
    target_loss = config["target_loss"]
    min_steps = config["min_steps"]
    output_base_dir = config["output_dir"]
    wandb_project = config["wandb_project"]
    training_params = config.get("training_params", {})
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # add logger
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | \
            <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        f"{output_base_dir}/training.log",  # 文件名
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        encoding="utf-8",
        level="DEBUG"
    )

    # Setup global WANDB env
    os.environ["WANDB_PROJECT"] = wandb_project
    
    # Load Dataset
    test_dataset = {}
    if target_dataset_name == "math":
        train_dataset, test_dataset["math"] = get_math_dataset()
        _, test_dataset["code"] = get_code_dataset()
    elif target_dataset_name == "code":
        train_dataset, test_dataset["code"] = get_code_dataset()
        _, test_dataset["math"] = get_math_dataset()
    else:
        raise ValueError(f"Unknown dataset: {target_dataset_name}")
        
    # Get Checkpoints
    checkpoints = get_checkpoints(source_model_dir, checkpoint_range[0], checkpoint_range[1])
    logger.info(f"Found {len(checkpoints)} checkpoints to process in range {checkpoint_range}.")
    
    experiment_results = []
    
    for cp_idx, cp_path in checkpoints:
        logger.info(f"Processing Checkpoint: {cp_path}")
        
        # Define run name
        run_name = f"{target_dataset_name}-from-{cp_path.name}"
        output_dir = os.path.join(output_base_dir, run_name)
        
        # Prepare Callback
        callback = ExperimentCallback(target_loss=target_loss, min_steps=min_steps)
        
        # Default Training Args
        # Note: save_strategy and save_steps are now respected from config or defaults
        default_args = {
            "output_dir": output_dir,
            "do_eval": True,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "eval_on_start": True,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "logging_steps": 1,
            "learning_rate": 2e-7,
            "max_length": 16384,
            "num_train_epochs": 100,
            "max_steps": -1,
            "save_strategy": "steps", # Default to steps if not provided
            "save_steps": 500,        # Default save step
            "save_only_model": True,
            "seed": SEED,
            "report_to": "wandb",
            "run_name": run_name,
            "adam_beta1": 1e-12,
            "adam_beta2": 1e-12,
            "lr_scheduler_type": "constant",
        }
        
        # Merge defaults with config params
        merged_args = {**default_args, **training_params}
        
        # Ensure max_steps is set high enough if using steps
        if "max_steps" not in training_params:
            merged_args["max_steps"] = 2000
            merged_args["num_train_epochs"] = 0
        
        # Force run_name in args to ensure separate runs
        merged_args["run_name"] = run_name

        # Initialize Trainer
        trainer = SFTTrainer(
            model = str(cp_path),
            train_dataset = train_dataset,
            eval_dataset = test_dataset,
            args = SFTConfig(**merged_args),
            callbacks=[callback]
        )
        
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed for checkpoint {cp_path}: {e}")
        finally:
            # Ensure wandb run is closed so the next one can start cleanly
            wandb.finish()
        
        # Collect results
        result_entry = {
            "checkpoint_idx": cp_idx,
            "source_checkpoint": str(cp_path),
            "target_dataset": target_dataset_name,
            "results": callback.results,
            "wandb_run_name": run_name
        }
        experiment_results.append(result_entry)
        
        # Save incremental results to the base directory
        with open(os.path.join(output_base_dir, "experiment_results.json"), 'w') as f:
            json.dump(experiment_results, f, indent=2)

    # Log Summary to WandB (as a separate summary run or just a table)
    # We will start a final run just to log the summary table
    try:
        run = wandb.init(
            project="data-calibrator-exp0-heatmap",
            name=f"summary-{target_dataset_name}",
            reinit=True
        )

        for res in experiment_results:
            r = res.get("results", {})
            run.log({
                'step': res.get("checkpoint_idx", None),
                'distance_via_loss': r.get("steps_to_reach_target_loss", None),
                'min_step_loss': r.get("loss_at_min_steps", None),
                'min_step_gradient': r.get("grad_norm_at_min_steps", None)
            })
        
        # run.log({"heatmap_results": table})
        run.finish()
        logger.info("Summary logged to WandB")
        
    except Exception as e:
        logger.error(f"Failed to log summary to WandB: {e}")

if __name__ == "__main__":
    main()