"""
基于有限差分的测地线方向 SFT 训练。

与 sft_via_geoguide.py 结构对齐，区别仅在于方向计算方式：
- sft_via_geoguide.py: 预训练向量场模型计算 Jacobian
- train_with_geo.py: 有限差分（tentative SFTTrainer N步）测量 v1/v2

每个 epoch：
1. eval → normalize → early stop
2. 纯 math 数据 → SFTTrainer tentative N步 → v1 → 恢复
3. 纯 code 数据 → SFTTrainer tentative N步 → v2 → 恢复
4. compute_metric_and_ratio → ratio
5. build_mixed_dataset(ratio, total_train_size)
6. SFTTrainer.train(max_steps=rebalance_steps)
"""

import os
import sys
import json
import logging
import shutil
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets

# 项目根目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED, set_seed


# =========================================================
# 0. 配置
# =========================================================
@dataclass
class GeoGradConfig:
    """训练配置 — 与 sft_via_geoguide.py 的 GeoGuideConfig 结构对齐。"""
    # --- 模型 ---
    base_model_path: str = "~/models/Qwen3-4B"

    # --- 数据 ---
    dataset_pool_size: int = 1000
    total_train_size: int = 1000       # 每轮混合训练集总量
    math_test_size: int = 100
    code_test_size: int = 100

    # --- 训练超参 ---
    max_epochs: int = 1000
    rebalance_steps: int = 19          # 每轮 tentative / 正式训练步数
    learning_rate: float = 2e-7
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 16384
    adam_beta1: float = 1e-12
    adam_beta2: float = 1e-12
    lr_scheduler_type: str = "constant"
    max_steps: int = -1                # 全局累计 effective step 上限（-1=不限）
    target_loss: float = -1.0          # early stop 阈值（-1=不启用）
    save_steps: int = 19
    save_total_limit: int = 3
    train_device: str = "cuda:0"
    eps: float = 1e-8                  # 正则化常数，A = J J^T + eps * I

    # --- 测地线目标 ---
    geo_target_domain: str = "math"
    geo_target_value: float = 0.2

    # --- 归一化数据源 ---
    m2c_json: str = str(_SCRIPT_DIR / "m2c.json")
    c2m_json: str = str(_SCRIPT_DIR / "c2m.json")

    # --- 输出 ---
    output_dir: str = str(_SCRIPT_DIR / "result" / "train_with_geo")
    seed: int = SEED

    # --- wandb ---
    wandb_project: str = "data-calibrator"
    report_to: str = "wandb"


# =========================================================
# 1. 归一化参数
# =========================================================
def compute_normalization_params(m2c_path: str, c2m_path: str):
    """从 m2c.json / c2m.json 提取全局 min-max 归一化参数。"""
    def _load_matrix(path):
        with open(path, "r") as f:
            data = json.load(f)
        row_keys = sorted(data.keys(), key=int)
        col_keys = sorted(data[row_keys[0]].keys(), key=int)
        matrix = [[data[r][c] for c in col_keys] for r in row_keys]
        matrix = [row[::19] for row in matrix]
        return matrix

    mat_m2c = _load_matrix(m2c_path)
    mat_c2m = _load_matrix(c2m_path)

    all_math = [item["eval_math_loss"] for mat in (mat_m2c, mat_c2m) for row in mat for item in row]
    all_code = [item["eval_code_loss"] for mat in (mat_m2c, mat_c2m) for row in mat for item in row]

    return {
        "math_min": min(all_math), "math_max": max(all_math),
        "code_min": min(all_code), "code_max": max(all_code),
    }


def normalize_losses(math_loss: float, code_loss: float, norm_params: dict):
    """原始 loss → [0,1]² 归一化坐标。"""
    math_range = norm_params["math_max"] - norm_params["math_min"]
    code_range = norm_params["code_max"] - norm_params["code_min"]
    x = (math_loss - norm_params["math_min"]) / math_range if math_range > 0 else 0.0
    y = (code_loss - norm_params["code_min"]) / code_range if code_range > 0 else 0.0
    return float(x), float(y)


# =========================================================
# 1b. 预 tokenize
# =========================================================
def pre_tokenize_pool(dataset, tokenizer, max_length: int):
    """对 prompt/completion 格式的数据集做一次性 tokenize + truncate。"""
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


# =========================================================
# 2. 评估（SFTTrainer-based，与 sft_via_geoguide.py 对齐）
# =========================================================
def evaluate_losses(model, tokenizer, math_train, math_test, code_test, cfg):
    """用 SFTTrainer.evaluate() 评估 math/code loss，与 sft_via_geoguide.py 对齐。"""
    from trl import SFTTrainer, SFTConfig

    dummy_train, _, _ = build_mixed_dataset(
        math_train, math_train, 0.5, min(10, cfg.total_train_size),
    )
    ckpt_dir = os.path.join(cfg.output_dir, "_eval_tmp")
    eval_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dummy_train,
        eval_dataset={"math": math_test, "code": code_test},
        args=SFTConfig(
            do_eval=True,
            eval_strategy="no",
            max_length=cfg.max_seq_length,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_train_batch_size,
            num_train_epochs=1,
            output_dir=ckpt_dir,
            remove_unused_columns=False,
            seed=cfg.seed,
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    metrics = eval_trainer.evaluate()
    del eval_trainer
    torch.cuda.empty_cache()
    return metrics["eval_math_loss"], metrics["eval_code_loss"]


# =========================================================
# 3. 数据集构建
# =========================================================
def build_mixed_dataset(math_train, code_train, math_ratio: float, total_size: int):
    """按比例从 math/code 池采样并拼接。"""
    n_math = max(1, int(round(total_size * math_ratio)))
    n_code = max(1, total_size - n_math)
    n_math = min(n_math, len(math_train))
    n_code = min(n_code, len(code_train))

    math_subset = math_train.shuffle(seed=SEED).select(range(n_math))
    code_subset = code_train.shuffle(seed=SEED).select(range(n_code))

    mixed = concatenate_datasets([math_subset, code_subset])
    mixed = mixed.shuffle(seed=SEED)
    return mixed, n_math, n_code


# =========================================================
# 4. 椭圆最近点 & 度规计算
# =========================================================
def ellipse_closest_to_line(G, target_domain, target_value, cx, cy):
    """在度规椭圆 x^T G x = 1 上找到距离目标直线最近的点。"""
    G_np = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
    G_inv = np.linalg.inv(G_np)

    if target_domain == "math":
        n = np.array([1.0, 0.0])
        sign_ref = target_value - cx
    else:
        n = np.array([0.0, 1.0])
        sign_ref = target_value - cy

    G_inv_n = G_inv @ n
    denom = np.sqrt(n @ G_inv_n)
    if denom < 1e-15:
        return torch.tensor(n, dtype=torch.float64)

    a = G_inv_n / denom
    if sign_ref < 0:
        a = -a

    return torch.tensor(a, dtype=torch.float64)


def compute_metric_and_ratio(v1, v2, target_domain, target_value, nx, ny, eps):
    """从有限差分方向 v1/v2 构建 Jacobian，计算度规张量 G，求椭圆最优方向和配比。"""
    J = np.column_stack([v1, v2])
    A = J @ J.T + eps * np.eye(2)
    G = np.linalg.inv(A)

    a = ellipse_closest_to_line(G, target_domain, target_value, nx, ny)
    a_np = a.numpy()

    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("[WARN] Jacobian singular, using pseudo-inverse")
        J_inv = np.linalg.pinv(J)

    delta_ratio = J_inv @ a_np
    abs_delta = np.abs(delta_ratio)
    s = abs_delta.sum()
    if s < 1e-12:
        ratio = np.array([0.5, 0.5])
    else:
        ratio = abs_delta / s

    return ratio, a_np, G


# =========================================================
# 5. 有限差分：tentative SFTTrainer N步 → re-eval → restore
# =========================================================
def finite_diff_direction(model, tokenizer, domain_train_dataset,
                          math_train, math_test, code_test,
                          norm_params, nx_before, ny_before,
                          cfg):
    """用纯 domain 数据跑 rebalance_steps 步 tentative 训练，测量归一化 loss 空间位移。"""
    from trl import SFTTrainer, SFTConfig

    model_state = deepcopy(model.state_dict())

    ckpt_dir = os.path.join(cfg.output_dir, "_tentative_tmp")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=domain_train_dataset,
        args=SFTConfig(
            max_steps=cfg.rebalance_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            lr_scheduler_type=cfg.lr_scheduler_type,
            max_length=cfg.max_seq_length,
            num_train_epochs=9999,
            output_dir=ckpt_dir,
            remove_unused_columns=False,
            seed=cfg.seed,
            save_strategy="no",
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    # re-eval after tentative training
    raw_math, raw_code = evaluate_losses(model, tokenizer, math_train, math_test, code_test, cfg)
    nx_after, ny_after = normalize_losses(raw_math, raw_code, norm_params)

    # restore model
    model.load_state_dict(model_state)

    return np.array([nx_after - nx_before, ny_after - ny_before])


# =========================================================
# 6. stdout/stderr tee
# =========================================================
class TeeStream:
    """同时写入原始流和日志文件的流包装器。"""
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()


def _parse_cuda_index(device_str: str) -> int:
    if device_str == "cpu":
        return -1
    return int(device_str.split(":")[-1])


# =========================================================
# 7. 主入口
# =========================================================
def main(cfg: GeoGradConfig | None = None):
    if cfg is None:
        cfg = GeoGradConfig()

    train_idx = _parse_cuda_index(cfg.train_device)
    if train_idx >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(train_idx)
        cfg.train_device = "cuda:0"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.makedirs(cfg.output_dir, exist_ok=True)

    log_file = open(os.path.join(cfg.output_dir, "train_with_geo.log"), "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    file_handler = logging.FileHandler(
        os.path.join(cfg.output_dir, "train_with_geo.log"), mode="a", encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)

    try:
        _run(cfg)
    finally:
        logging.root.removeHandler(file_handler)
        file_handler.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


def _run(cfg: GeoGradConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    set_seed(cfg.seed)
    device = torch.device(cfg.train_device)

    # --- wandb ---
    use_wandb = cfg.report_to == "wandb"
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), name="geo-grad")

    # --- 归一化参数 ---
    print("Computing normalization params ...")
    norm_params = compute_normalization_params(cfg.m2c_json, cfg.c2m_json)
    print(f"  math: [{norm_params['math_min']:.6f}, {norm_params['math_max']:.6f}]")
    print(f"  code: [{norm_params['code_min']:.6f}, {norm_params['code_max']:.6f}]")

    # --- 加载模型 ---
    print(f"Loading model from {cfg.base_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    model.to(device)
    model.train()

    # --- 加载数据集 ---
    print("Loading datasets ...")
    math_train, math_test = get_math_dataset(size=cfg.dataset_pool_size)
    code_train, code_test = get_code_dataset(size=cfg.dataset_pool_size)

    # --- 预 tokenize ---
    print("Pre-tokenizing data pools ...")
    math_train = pre_tokenize_pool(math_train, tokenizer, cfg.max_seq_length)
    code_train = pre_tokenize_pool(code_train, tokenizer, cfg.max_seq_length)
    math_test = pre_tokenize_pool(math_test, tokenizer, cfg.max_seq_length)
    code_test = pre_tokenize_pool(code_test, tokenizer, cfg.max_seq_length)
    print("  Pre-tokenization done.")

    # --- 主循环 ---
    epoch_logs = []
    saved_checkpoints = []
    global_step = 0

    for epoch in range(cfg.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.max_epochs}  (global_step={global_step})")
        print(f"{'='*60}")

        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            print(f"  [STOP] global step budget exhausted: {global_step} >= {cfg.max_steps}")
            break

        if cfg.max_steps > 0:
            remaining = cfg.max_steps - global_step
            epoch_steps = min(cfg.rebalance_steps, remaining)
        else:
            epoch_steps = cfg.rebalance_steps

        # 1. Evaluate
        print("  Evaluating ...")
        raw_math, raw_code = evaluate_losses(model, tokenizer, math_train, math_test, code_test, cfg)
        nx, ny = normalize_losses(raw_math, raw_code, norm_params)
        print(f"  raw: math={raw_math:.6f}, code={raw_code:.6f}")
        print(f"  normalized: ({nx:.6f}, {ny:.6f})")

        check_loss = nx if cfg.geo_target_domain == "math" else ny
        if cfg.target_loss > 0 and check_loss <= cfg.target_loss:
            print(f"  [STOP] target reached: {check_loss:.6f} <= {cfg.target_loss:.6f}")
            break

        # 2. Tentative: 纯 math → v1
        print("  Computing v1 (math tentative) ...")
        v1 = finite_diff_direction(model, tokenizer, math_train,
                                   math_train, math_test, code_test,
                                   norm_params, nx, ny, cfg)
        print(f"  v1 = ({v1[0]:.8f}, {v1[1]:.8f})")

        # 3. Tentative: 纯 code → v2
        print("  Computing v2 (code tentative) ...")
        v2 = finite_diff_direction(model, tokenizer, code_train,
                                   math_train, math_test, code_test,
                                   norm_params, nx, ny, cfg)
        print(f"  v2 = ({v2[0]:.8f}, {v2[1]:.8f})")

        # 4. 度规 → ratio
        ratio, a, G = compute_metric_and_ratio(
            v1, v2, cfg.geo_target_domain, cfg.geo_target_value, nx, ny, cfg.eps)
        print(f"  ratio: math={ratio[0]:.4f}, code={ratio[1]:.4f}")
        print(f"  ellipse direction a = ({a[0]:.6f}, {a[1]:.6f})")

        # 5. 构建混合数据集
        mixed_train, n_math, n_code = build_mixed_dataset(
            math_train, code_train, ratio[0], cfg.total_train_size)
        print(f"  mixed dataset: {n_math} math + {n_code} code = {len(mixed_train)}")

        # 6. SFTTrainer 正式训练
        ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=mixed_train,
            args=SFTConfig(
                max_steps=epoch_steps,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.learning_rate,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                lr_scheduler_type=cfg.lr_scheduler_type,
                max_length=cfg.max_seq_length,
                num_train_epochs=9999,
                logging_steps=1,
                output_dir=ckpt_dir,
                remove_unused_columns=False,
                seed=cfg.seed,
                save_strategy="no",
                run_name=f"geo-grad-epoch{epoch}",
                report_to=cfg.report_to,
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )
        print(f"  Training {epoch_steps} steps with ratio math={ratio[0]:.4f}, code={ratio[1]:.4f} ...")
        trainer.train()

        # 7. warm start
        seg_steps_actual = trainer.state.global_step
        global_step += seg_steps_actual
        model = trainer.model

        del trainer
        torch.cuda.empty_cache()

        # 8. 日志
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_steps": seg_steps_actual,
            "raw_math_loss": raw_math, "raw_code_loss": raw_code,
            "normalized_xy": [nx, ny],
            "v1": v1.tolist(), "v2": v2.tolist(),
            "ratio": {"math": float(ratio[0]), "code": float(ratio[1])},
            "ellipse_direction": a.tolist(),
            "G": G.tolist() if isinstance(G, np.ndarray) else G,
            "n_math": n_math, "n_code": n_code,
        }
        epoch_logs.append(log_entry)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "raw_math_loss": raw_math, "raw_code_loss": raw_code,
                "norm_math": nx, "norm_code": ny,
                "ratio_math": ratio[0], "ratio_code": ratio[1],
            })

        full_log = {"config": asdict(cfg), "epochs": epoch_logs}
        with open(os.path.join(cfg.output_dir, "geo_grad_log.json"), "w") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)

        summary_path = os.path.join(cfg.output_dir, "epoch_summary.json")
        summary_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "raw_loss": {"math": raw_math, "code": raw_code},
            "normalized": {"math": nx, "code": ny},
            "ratio": {"math": float(ratio[0]), "code": float(ratio[1])},
        }
        if epoch == 0:
            summary_list = []
        else:
            with open(summary_path, "r") as f:
                summary_list = json.load(f)
        summary_list.append(summary_entry)
        with open(summary_path, "w") as f:
            json.dump(summary_list, f, indent=2, ensure_ascii=False)

        # 9. checkpoint
        if cfg.save_steps > 0 and epoch % cfg.save_steps == 0:
            save_dir = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}")
            print(f"  Saving checkpoint to {save_dir} ...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            saved_checkpoints.append(save_dir)
            while len(saved_checkpoints) > cfg.save_total_limit:
                old_dir = saved_checkpoints.pop(0)
                if os.path.isdir(old_dir):
                    shutil.rmtree(old_dir)
                    print(f"  Removed old checkpoint: {old_dir}")

    if use_wandb:
        wandb.finish()

    print(f"\n{'='*60}")
    print("Geo-grad training complete.")
    print(f"  Total epochs: {len(epoch_logs)}, global_step: {global_step}")
    print(f"{'='*60}")


# =========================================================
# 8. CLI
# =========================================================
def parse_args() -> GeoGradConfig:
    defaults = GeoGradConfig()
    p = argparse.ArgumentParser(description="基于有限差分的测地线方向 SFT 训练")

    p.add_argument("--base_model_path", type=str, default=defaults.base_model_path)
    p.add_argument("--dataset_pool_size", type=int, default=defaults.dataset_pool_size)
    p.add_argument("--total_train_size", type=int, default=defaults.total_train_size)
    p.add_argument("--math_test_size", type=int, default=defaults.math_test_size)
    p.add_argument("--code_test_size", type=int, default=defaults.code_test_size)
    p.add_argument("--max_epochs", type=int, default=defaults.max_epochs)
    p.add_argument("--rebalance_steps", type=int, default=defaults.rebalance_steps)
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    p.add_argument("--adam_beta1", type=float, default=defaults.adam_beta1)
    p.add_argument("--adam_beta2", type=float, default=defaults.adam_beta2)
    p.add_argument("--lr_scheduler_type", type=str, default=defaults.lr_scheduler_type)
    p.add_argument("--max_steps", type=int, default=defaults.max_steps)
    p.add_argument("--target_loss", type=float, default=defaults.target_loss)
    p.add_argument("--save_steps", type=int, default=defaults.save_steps)
    p.add_argument("--save_total_limit", type=int, default=defaults.save_total_limit)
    p.add_argument("--train_device", type=str, default=defaults.train_device)
    p.add_argument("--eps", type=float, default=defaults.eps)
    p.add_argument("--geo_target_domain", type=str, default=defaults.geo_target_domain, choices=["math", "code"])
    p.add_argument("--geo_target_value", type=float, default=defaults.geo_target_value)
    p.add_argument("--m2c_json", type=str, default=defaults.m2c_json)
    p.add_argument("--c2m_json", type=str, default=defaults.c2m_json)
    p.add_argument("--output_dir", type=str, default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    p.add_argument("--report_to", type=str, default=defaults.report_to)

    args = p.parse_args()
    return GeoGradConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
