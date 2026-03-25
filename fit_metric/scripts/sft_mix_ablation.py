"""
消融实验：比较单领域 SFT 与混合配比 SFT 的 eval loss 曲线。

第一条线：1000 条纯 math 数据 SFT
第二条线：1000 条 9:1 (math:code) 混合数据 SFT

超参数与 train_with_geo.py 对齐。输出折线图，纵轴为指定 eval_set 的 loss，横轴为 global_step。
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from datasets import concatenate_datasets

_SCRIPT_DIR = Path(__file__).resolve().parent
_FIT_METRIC_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _FIT_METRIC_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED, set_seed


# =========================================================
# 配置
# =========================================================
@dataclass
class AblationConfig:
    base_model_path: str = "~/models/Qwen3-4B"

    # 数据
    dataset_pool_size: int = 1000
    train_size: int = 1000
    math_test_size: int = 100
    code_test_size: int = 100
    mix_ratio: float = 0.9  # math 占比（第二条线）

    # 训练超参（与 train_with_geo 对齐）
    max_steps: int = 100
    learning_rate: float = 2e-7
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 16384
    adam_beta1: float = 1e-12
    adam_beta2: float = 1e-12
    lr_scheduler_type: str = "constant"
    train_device: str = "cuda:0"

    # eval
    eval_domain: str = "math"  # 画图用的 eval 领域: "math" 或 "code"
    eval_steps: int = 1

    # 输出
    seed: int = SEED
    wandb_project: str = "data-calibrator"
    report_to: str = "wandb"


# =========================================================
# 工具函数
# =========================================================
def pre_tokenize_pool(dataset, tokenizer, max_length: int):
    def _tokenize(example):
        prompt_ids = tokenizer.apply_chat_template(example["prompt"])
        full_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"]
        )
        completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            completion_mask = completion_mask[:max_length]
        return {"input_ids": full_ids, "completion_mask": completion_mask}

    return dataset.map(_tokenize)


def build_dataset(math_train, code_train, math_ratio: float, total_size: int):
    """按比例构建混合数据集。math_ratio=1.0 时为纯 math。"""
    n_math = max(1, int(round(total_size * math_ratio)))
    n_code = total_size - n_math
    n_math = min(n_math, len(math_train))

    math_subset = math_train.shuffle(seed=SEED).select(range(n_math))

    if n_code <= 0:
        return math_subset

    n_code = min(n_code, len(code_train))
    code_subset = code_train.shuffle(seed=SEED).select(range(n_code))
    mixed = concatenate_datasets([math_subset, code_subset]).shuffle(seed=SEED)
    return mixed


def run_sft(label, model_init_fn, tokenizer, train_dataset, eval_dataset,
            cfg: AblationConfig, output_dir: str):
    """运行一次 SFT 训练，返回每步的 eval loss 列表。"""
    from trl import SFTTrainer, SFTConfig

    model = model_init_fn()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            do_eval=True,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            eval_on_start=True,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_length=cfg.max_seq_length,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            lr_scheduler_type=cfg.lr_scheduler_type,
            num_train_epochs=9999,
            logging_steps=1,
            output_dir=output_dir,
            remove_unused_columns=False,
            seed=cfg.seed,
            save_strategy="no",
            run_name=label,
            report_to=cfg.report_to,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )

    trainer.train()

    # 从 trainer log_history 提取 eval loss
    eval_key = f"eval_{cfg.eval_domain}_loss"
    steps, losses = [], []
    for entry in trainer.state.log_history:
        if eval_key in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry[eval_key])

    del model, trainer
    torch.cuda.empty_cache()

    return steps, losses


# =========================================================
# 主函数
# =========================================================
def main(cfg: AblationConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_idx = int(cfg.train_device.split(":")[-1]) if cfg.train_device != "cpu" else -1
    if train_idx >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(train_idx)
        cfg.train_device = "cuda:0"

    os.environ["WANDB_PROJECT"] = cfg.wandb_project

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = str(_FIT_METRIC_DIR / "result" / "sft_mix_ablation" / timestamp)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(cfg.seed)

    # 保存配置
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # 加载数据
    print("Loading datasets ...")
    math_train, math_test = get_math_dataset(size=cfg.dataset_pool_size)
    code_train, code_test = get_code_dataset(size=cfg.dataset_pool_size)

    print(f"Loading tokenizer from {cfg.base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)

    print("Pre-tokenizing ...")
    math_train = pre_tokenize_pool(math_train, tokenizer, cfg.max_seq_length)
    code_train = pre_tokenize_pool(code_train, tokenizer, cfg.max_seq_length)
    math_test = pre_tokenize_pool(math_test, tokenizer, cfg.max_seq_length)
    code_test = pre_tokenize_pool(code_test, tokenizer, cfg.max_seq_length)

    eval_dataset = {"math": math_test, "code": code_test}

    def model_init_fn():
        return AutoModelForCausalLM.from_pretrained(cfg.base_model_path)

    # --- 实验 1: 纯 math ---
    print("\n=== Run 1: pure math (1000 math) ===")
    ds_pure = build_dataset(math_train, code_train, 1.0, cfg.train_size)
    steps1, losses1 = run_sft(
        "pure-math", model_init_fn, tokenizer, ds_pure, eval_dataset,
        cfg, os.path.join(output_dir, "pure_math"),
    )

    # --- 实验 2: 混合 ---
    ratio_label = f"{cfg.mix_ratio:.0%}math-{1-cfg.mix_ratio:.0%}code"
    print(f"\n=== Run 2: mixed ({ratio_label}, 1000 total) ===")
    ds_mixed = build_dataset(math_train, code_train, cfg.mix_ratio, cfg.train_size)
    steps2, losses2 = run_sft(
        f"mixed-{ratio_label}", model_init_fn, tokenizer, ds_mixed, eval_dataset,
        cfg, os.path.join(output_dir, "mixed"),
    )

    # --- 保存数据 ---
    results = {
        "eval_domain": cfg.eval_domain,
        "pure_math": {"steps": steps1, "losses": losses1},
        f"mixed_{ratio_label}": {"steps": steps2, "losses": losses2},
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # --- 画图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(steps1, losses1, marker="o", markersize=3, label="pure math (1000)")
    plt.plot(steps2, losses2, marker="s", markersize=3, label=f"mixed {ratio_label} (1000)")
    plt.xlabel("global_step")
    plt.ylabel(f"eval_{cfg.eval_domain}_loss")
    plt.title(f"SFT Ablation: pure math vs mixed ({ratio_label})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "ablation_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


# =========================================================
# CLI
# =========================================================
def parse_args() -> AblationConfig:
    defaults = AblationConfig()
    p = argparse.ArgumentParser(description="消融实验：单领域 vs 混合配比 SFT")

    p.add_argument("--base_model_path", type=str, default=defaults.base_model_path)
    p.add_argument("--dataset_pool_size", type=int, default=defaults.dataset_pool_size)
    p.add_argument("--train_size", type=int, default=defaults.train_size)
    p.add_argument("--math_test_size", type=int, default=defaults.math_test_size)
    p.add_argument("--code_test_size", type=int, default=defaults.code_test_size)
    p.add_argument("--mix_ratio", type=float, default=defaults.mix_ratio)
    p.add_argument("--max_steps", type=int, default=defaults.max_steps)
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    p.add_argument("--adam_beta1", type=float, default=defaults.adam_beta1)
    p.add_argument("--adam_beta2", type=float, default=defaults.adam_beta2)
    p.add_argument("--lr_scheduler_type", type=str, default=defaults.lr_scheduler_type)
    p.add_argument("--train_device", type=str, default=defaults.train_device)
    p.add_argument("--eval_domain", type=str, default=defaults.eval_domain, choices=["math", "code"])
    p.add_argument("--eval_steps", type=int, default=defaults.eval_steps)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    p.add_argument("--report_to", type=str, default=defaults.report_to)

    args = p.parse_args()
    return AblationConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
