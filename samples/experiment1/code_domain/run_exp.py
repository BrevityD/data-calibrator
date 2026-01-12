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

os.environ["WANDB_PROJECT"] = "data-calibrator"
max_seq_length = 16384

train_dataset, test_dataset = get_code_dataset()

trainer = SFTTrainer(
    model = "~/models/Qwen3-4B",
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args = SFTConfig(
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = 1,
        max_length = max_seq_length,
        learning_rate = 1e-5,
        per_device_train_batch_size = 16,  # 64 * `epoch` steps
        num_train_epochs = 2,
        gradient_accumulation_steps = 1,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "sgd",
        seed = SEED,
        save_strategy = "steps",
        save_steps = 1,
        save_only_model = True,
        run_name = "exp1-code-domain",
        report_to = "wandb",
    ),
)
trainer.train()
