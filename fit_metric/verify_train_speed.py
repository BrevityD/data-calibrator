"""
验证 trainer.train() 速度差异：
  A) train_with_geo 风格 — 无 eval 配置
  B) sft_via_geoguide 风格 — do_eval=True, eval_strategy="steps", eval_steps=999

用完全相同的数据集和模型，排除数据差异。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WORLD_SIZE"] = "1"

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


def run_train_A(model, tokenizer, mixed_train, steps, tag):
    """train_with_geo 风格: 无 eval"""
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


def run_train_B(model, tokenizer, mixed_train, eval_dataset, steps, tag):
    """sft_via_geoguide 风格: do_eval=True, eval_strategy=steps, eval_steps=999"""
    out_dir = f"/tmp/verify_speed_{tag}"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=mixed_train,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            do_eval=True,
            eval_strategy="steps",
            eval_steps=999,
            eval_on_start=False,
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
    RATIO = 0.6731  # sft_via_geoguide segment 0 的 ratio
    STEPS = 19

    model_path = "/public/home/jza/share_model/Qwen/Qwen3-1.7B"
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading datasets ...")
    math_train, math_test = get_math_dataset(size=1000)
    code_train, code_test = get_code_dataset(size=1000)

    print("Pre-tokenizing ...")
    math_train = pre_tokenize_pool(math_train, tokenizer, 16384)
    code_train = pre_tokenize_pool(code_train, tokenizer, 16384)
    math_test = pre_tokenize_pool(math_test, tokenizer, 16384)
    code_test = pre_tokenize_pool(code_test, tokenizer, 16384)

    mixed_train, n_math, n_code = build_mixed_dataset(
        math_train, code_train, RATIO, 1000,
    )
    print(f"Mixed dataset: {n_math} math + {n_code} code = {len(mixed_train)}")

    eval_dataset = {"math": math_test, "code": code_test}

    # 保存初始权重
    init_state = deepcopy(model.state_dict())

    # --- A: train_with_geo 风格 ---
    print("\n" + "=" * 60)
    print("A) train_with_geo style (no eval config)")
    print("=" * 60)
    model.load_state_dict(init_state)
    elapsed_A = run_train_A(model, tokenizer, mixed_train, STEPS, "A")
    print(f"  elapsed: {elapsed_A:.2f}s  ({elapsed_A/STEPS:.2f}s/step)")

    # --- B: sft_via_geoguide 风格 ---
    print("\n" + "=" * 60)
    print("B) sft_via_geoguide style (do_eval=True, eval_steps=999)")
    print("=" * 60)
    model.load_state_dict(init_state)
    elapsed_B = run_train_B(model, tokenizer, mixed_train, eval_dataset, STEPS, "B")
    print(f"  elapsed: {elapsed_B:.2f}s  ({elapsed_B/STEPS:.2f}s/step)")

    # --- 对比 ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  A (no eval):   {elapsed_A:.2f}s  ({elapsed_A/STEPS:.2f}s/step)")
    print(f"  B (with eval):  {elapsed_B:.2f}s  ({elapsed_B/STEPS:.2f}s/step)")
    print(f"  ratio B/A:      {elapsed_B/elapsed_A:.2f}x")


if __name__ == "__main__":
    main()
