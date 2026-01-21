import os
from pathlib import Path

## =======
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
import sys
sys.path.append(str(project_root))
## =======

from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED

from trl import SFTTrainer, SFTConfig

os.environ["WANDB_PROJECT"] = "data-calibrator-exp0"
max_seq_length = 16384

train_dataset, test_dataset = get_code_dataset()

trainer = SFTTrainer(
    model = "/models/Qwen3-1.7B",
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args = SFTConfig(
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = 1,
        eval_on_start = True,
        max_length = max_seq_length,
        learning_rate = 2e-7,
        per_device_train_batch_size = 4,  # 64 * `epoch` steps
        num_train_epochs = 15,
        gradient_accumulation_steps = 4,
        logging_steps = 1,
        output_dir = "outputs-1_7B-code-ceiling-5e7-bs16-ep15-signsgd",
        # optim = "sgd",
        adam_beta1 = 1e-12,
        adam_beta2 = 1e-12,
        lr_scheduler_type = "constant",
        seed = SEED,
        save_strategy = "steps",
        save_steps = 0.02,
        save_only_model = True,
        run_name = "code-1_7B-ceiling-5e7-bs16-ep15-signsgd",
        report_to = "wandb",
    ),
)
trainer.train()
