"""
验证 token 数量对 trainer.train() 速度的影响。

用两种 ratio 构建数据集，对比每步 token 数和训练耗时：
  A) ratio=0.7676 (train_with_geo epoch 0)
  B) ratio=0.6731 (sft_via_geoguide segment 0)
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

    init_state = deepcopy(model.state_dict())

    ratios = {
        "A (train_with_geo epoch0)": 0.7676,
        "B (sft_via_geoguide seg0)": 0.6731,
    }

    results = {}
    for label, ratio in ratios.items():
        mixed, n_math, n_code = build_mixed_dataset(
            math_train, code_train, ratio, 1000,
        )
        total_tokens = sum(len(x) for x in mixed["input_ids"])
        avg_tokens = total_tokens / len(mixed)
        print(f"\n{'=' * 60}")
        print(f"{label}: ratio={ratio}")
        print(f"  {n_math} math + {n_code} code = {len(mixed)}")
        print(f"  total_tokens={total_tokens}, avg={avg_tokens:.1f} tokens/sample")
        print(f"{'=' * 60}")

        model.load_state_dict(init_state)
        elapsed = run_train(model, tokenizer, mixed, STEPS, label[:1])
        print(f"  elapsed: {elapsed:.2f}s  ({elapsed/STEPS:.2f}s/step)")
        results[label] = {
            "ratio": ratio, "n_math": n_math, "n_code": n_code,
            "total_tokens": total_tokens, "avg_tokens": avg_tokens,
            "elapsed": elapsed, "per_step": elapsed / STEPS,
        }

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    for label, r in results.items():
        print(f"  {label}:")
        print(f"    {r['n_math']}m+{r['n_code']}c  "
              f"avg={r['avg_tokens']:.0f}tok/sample  "
              f"{r['elapsed']:.2f}s  ({r['per_step']:.2f}s/step)")


if __name__ == "__main__":
    main()
