"""
验证 geo_common 初始化 + CUDA_VISIBLE_DEVICES 缩减对 trainer.train() 速度的影响。

对比：
  A) 干净环境 (单卡, CUDA_VISIBLE_DEVICES=3)
  B) 模拟 sft_via_geoguide 环境:
     先在两张卡上初始化 geo_common，再缩减 CUDA_VISIBLE_DEVICES
"""

import os
# 先设两张卡，模拟 sft_via_geoguide 的初始状态
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import concatenate_datasets
from copy import deepcopy

from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED, set_seed
import geo_common


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


def build_mixed_dataset(math_train, code_train, math_ratio, total_size):
    n_math = max(1, int(round(total_size * math_ratio)))
    n_code = max(1, total_size - n_math)
    n_math = min(n_math, len(math_train))
    n_code = min(n_code, len(code_train))
    math_subset = math_train.shuffle(seed=SEED).select(range(n_math))
    code_subset = code_train.shuffle(seed=SEED).select(range(n_code))
    mixed = concatenate_datasets([math_subset, code_subset])
    mixed = mixed.shuffle(seed=SEED)
    return mixed, n_math, n_code


def run_train(model, tokenizer, mixed_train, steps, tag):
    out_dir = f"/tmp/verify_speed_{tag}"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=mixed_train,
        args=SFTConfig(
            max_steps=steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-7,
            max_length=16384,
            num_train_epochs=9999,
            logging_steps=1,
            output_dir=out_dir,
            remove_unused_columns=False,
            seed=SEED,
            save_strategy="no",
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    del trainer
    torch.cuda.empty_cache()
    return elapsed


def main():
    set_seed(SEED)
    STEPS = 19
    RATIO = 0.6731

    model_path = "/public/home/jza/share_model/Qwen/Qwen3-1.7B"

    # ---- Phase 1: 初始化 geo_common (在 cuda:1 即物理卡4) ----
    print("Initializing geo_common on cuda:1 ...")
    geo_common.init(device_str="cuda:1", seed=SEED)
    print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")

    # ---- Phase 2: 缩减 CUDA_VISIBLE_DEVICES (模拟 sft_via_geoguide) ----
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["WORLD_SIZE"] = "1"
    print(f"  Shrunk CUDA_VISIBLE_DEVICES to 3")
    print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")

    # ---- 加载模型和数据 ----
    print(f"\nLoading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading datasets ...")
    math_train, _ = get_math_dataset(size=1000)
    code_train, _ = get_code_dataset(size=1000)

    print("Pre-tokenizing ...")
    math_train = pre_tokenize_pool(math_train, tokenizer, 16384)
    code_train = pre_tokenize_pool(code_train, tokenizer, 16384)

    mixed, n_math, n_code = build_mixed_dataset(
        math_train, code_train, RATIO, 1000,
    )
    print(f"Mixed: {n_math} math + {n_code} code")

    init_state = deepcopy(model.state_dict())

    # ---- B: 带 geo_common 环境训练 ----
    print(f"\n{'=' * 60}")
    print("B) With geo_common initialized (simulating sft_via_geoguide)")
    print(f"{'=' * 60}")
    model.load_state_dict(init_state)
    elapsed_B = run_train(model, tokenizer, mixed, STEPS, "B")
    print(f"  elapsed: {elapsed_B:.2f}s  ({elapsed_B/STEPS:.2f}s/step)")

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"  With geo_common: {elapsed_B:.2f}s  ({elapsed_B/STEPS:.2f}s/step)")
    print(f"  Previous clean run was ~20s (~1.05s/step)")
    print(f"  If this is >>20s, geo_common env is the cause.")


if __name__ == "__main__":
    main()
