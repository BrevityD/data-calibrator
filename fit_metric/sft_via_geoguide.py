"""
测地线指导的 SFT 数据配比迭代训练。

每个 segment（N 个训练步）：评估 math/code loss → 归一化到 [0,1]² → 计算测地线方向 → 调整配比 → 训练 N 步。
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datacalibrator.datasets.math_adaptor import get_math_dataset
from datacalibrator.datasets.code_adaptor import get_code_dataset
from datacalibrator.seed import SEED, set_seed

# fit_metric 内部模块（延迟初始化，import 无副作用）
sys.path.insert(0, str(_SCRIPT_DIR))
import geo_common
from variational_geodesic import multi_start_variational_geodesic_to_line


# =========================================================
# 0. 配置
# =========================================================
@dataclass
class GeoGuideConfig:
    # --- 模型 ---
    base_model_path: str = "~/models/Qwen3-4B"

    # --- 数据 ---
    dataset_pool_size: int = 1000      # 数据集加载大小（固定，保证 test split 一致）
    total_train_size: int = 1000       # 每轮混合训练集总量（<= dataset_pool_size）
    math_test_size: int = 100
    code_test_size: int = 100

    # --- 训练超参 ---
    max_segments: int = 1000         # 外层循环安全上限
    rebalance_steps: int = 20        # 每隔多少训练步重新配比
    learning_rate: float = 2e-7
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 16384
    adam_beta1: float = 1e-12
    adam_beta2: float = 1e-12
    lr_scheduler_type: str = "constant"
    max_steps: int = -1              # 全局累计步数上限（-1=不限）
    target_loss: float = -1.0        # early stop 阈值（-1=不启用），检查 geo_target_domain 的 loss
    eval_steps: int = 1
    eval_on_start: bool = True
    save_steps: int = 19
    train_device: str = "cuda:0"       # SFT 训练设备

    # --- 测地线 ---
    geo_device: str = "cuda:4"         # 测地线计算设备（与 train_device 分开避免显存冲突）
    geo_target_domain: str = "math"    # 优化目标领域: "math" → x=const, "code" → y=const
    geo_target_value: float = 0.2      # 目标直线的坐标值
    geo_K: int = 12                    # 多起点候选数
    geo_N: int = 50                    # 粗搜索离散段数
    geo_refine_top_k: int = 3
    geo_refine_N: int = 100
    math_model_path: str = "/public/home/jza/data_calibrate/data_mixture/metric_fit/v1.pth"
    code_model_path: str = "/public/home/jza/data_calibrate/data_mixture/metric_fit/v2.pth"

    # --- 归一化数据源 ---
    m2c_json: str = str(_SCRIPT_DIR / "m2c.json")
    c2m_json: str = str(_SCRIPT_DIR / "c2m.json")

    # --- 输出 ---
    output_dir: str = str(_SCRIPT_DIR / "result" / "sft_via_geoguide")
    seed: int = SEED

    # --- wandb ---
    wandb_project: str = "data-calibrator"
    report_to: str = "wandb"


# =========================================================
# 1. 归一化参数
# =========================================================
def compute_normalization_params(m2c_path: str, c2m_path: str):
    """从 m2c.json / c2m.json 提取全局 min-max 归一化参数。

    直接读取 JSON 而不 import data_process（避免模块级副作用）。
    """
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
        "math_min": min(all_math),
        "math_max": max(all_math),
        "code_min": min(all_code),
        "code_max": max(all_code),
    }


def normalize_losses(math_loss: float, code_loss: float, norm_params: dict):
    """原始 loss → [0,1]² 归一化坐标。"""
    math_range = norm_params["math_max"] - norm_params["math_min"]
    code_range = norm_params["code_max"] - norm_params["code_min"]
    x = (math_loss - norm_params["math_min"]) / math_range if math_range > 0 else 0.0
    y = (code_loss - norm_params["code_min"]) / code_range if code_range > 0 else 0.0
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


# =========================================================
# 1b. 预 tokenize
# =========================================================
def pre_tokenize_pool(dataset, tokenizer, max_length: int):
    """对 prompt/completion 格式的数据集做一次性 tokenize + truncate。

    复用 SFTTrainer 内部逻辑：apply_chat_template → input_ids + completion_mask。
    之后每段训练用 skip_prepare_dataset=True 跳过重复 tokenize。
    """
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
# 2. 测地线方向 & 配比映射
# =========================================================
def compute_geodesic_tangent(start_xy, cfg: GeoGuideConfig):
    """调用 multi_start_variational_geodesic_to_line，返回起点处切向量。

    geo_target_domain="math" → target_axis=0, 到 x=geo_target_value 竖直线
    geo_target_domain="code" → target_axis=1, 到 y=geo_target_value 水平线
    """
    target_axis = 0 if cfg.geo_target_domain == "math" else 1

    path, energy, arc_length, info, candidates = multi_start_variational_geodesic_to_line(
        start=start_xy,
        x_target=cfg.geo_target_value,
        K=cfg.geo_K,
        N=cfg.geo_N,
        refine_top_k=cfg.geo_refine_top_k,
        refine_N=cfg.geo_refine_N,
        verbose=True,
        target_axis=target_axis,
    )
    # 起点处切向量：path[1] - path[0]
    tangent = path[1] - path[0]
    return tangent, path, energy, arc_length


def compute_jacobian(xy):
    """在归一化坐标 xy 处计算 J = [v1 | v2]（2×2）。

    v1 = _math_model(xy)  → ∂loss/∂math_ratio 方向
    v2 = _code_model(xy)  → ∂loss/∂code_ratio 方向
    J 列向量分别是两个方向的 loss 变化。
    """
    device = geo_common.get_device()
    dtype = geo_common.get_dtype()
    loc = torch.tensor(xy, dtype=dtype, device=device).unsqueeze(0)
    with torch.no_grad():
        v1 = geo_common._math_model(loc).squeeze(0).cpu().numpy()  # (2,)
        v2 = geo_common._code_model(loc).squeeze(0).cpu().numpy()  # (2,)
    J = np.column_stack([v1, v2])  # (2, 2)
    return J


def tangent_to_ratio(tangent, xy):
    """通过 J^{-1} 将 loss 空间切向量还原为 ratio 空间变化，再归一化为配比。

    delta_ratio = J^{-1} @ tangent
    ratio = |delta_ratio| / sum(|delta_ratio|)

    返回 (math_ratio, code_ratio)，和为 1。
    """
    J = compute_jacobian(xy)
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("[WARN] Jacobian singular, falling back to pseudo-inverse")
        J_inv = np.linalg.pinv(J)

    delta_ratio = J_inv @ tangent  # a = J^{-1} γ̇(0), Proposal §8.1
    # Proposal §8.2: p_math = a1/(a1+a2) when both >= 0
    # Proposal §8.5: negative coefficient → truncate & renormalize
    if delta_ratio[0] < 0 or delta_ratio[1] < 0:
        print(f"  [WARN] negative opt-basis coefficient: a=({delta_ratio[0]:.6f}, {delta_ratio[1]:.6f}), "
              f"truncating to abs (Proposal §8.5)")
    abs_delta = np.abs(delta_ratio)
    s = abs_delta.sum()
    if s < 1e-12:
        return 0.5, 0.5
    ratio = abs_delta / s
    return float(ratio[0]), float(ratio[1])


# =========================================================
# 3. 数据集构建
# =========================================================
def build_mixed_dataset(math_train, code_train, math_ratio: float, total_size: int):
    """按比例从 math/code 池采样并拼接。"""
    n_math = max(1, int(round(total_size * math_ratio)))
    n_code = max(1, total_size - n_math)

    # 确保不超过可用数据量
    n_math = min(n_math, len(math_train))
    n_code = min(n_code, len(code_train))

    math_subset = math_train.shuffle(seed=SEED).select(range(n_math))
    code_subset = code_train.shuffle(seed=SEED).select(range(n_code))

    mixed = concatenate_datasets([math_subset, code_subset])
    mixed = mixed.shuffle(seed=SEED)
    return mixed, n_math, n_code


# =========================================================
# 4. stdout/stderr tee
# =========================================================
class TeeStream:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()


# =========================================================
# 5. 主循环
# =========================================================
def _parse_cuda_index(device_str: str) -> int:
    """从 'cuda:N' 提取 N，'cpu' 返回 -1。"""
    if device_str == "cpu":
        return -1
    return int(device_str.split(":")[-1])


def main(cfg: GeoGuideConfig | None = None):
    if cfg is None:
        cfg = GeoGuideConfig()

    # 设置 CUDA_VISIBLE_DEVICES，包含 train 和 geo 两张卡
    # 必须在任何 CUDA 操作之前设置
    train_idx = _parse_cuda_index(cfg.train_device)
    geo_idx = _parse_cuda_index(cfg.geo_device)
    if train_idx >= 0 and geo_idx >= 0:
        if train_idx == geo_idx:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(train_idx)
            # 两者共用同一张卡，重映射为 cuda:0
            cfg = GeoGuideConfig(**{
                **{k: v for k, v in cfg.__dict__.items()},
                "train_device": "cuda:0",
                "geo_device": "cuda:0",
            })
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{train_idx},{geo_idx}"
            # 重映射: train → cuda:0, geo → cuda:1
            cfg = GeoGuideConfig(**{
                **{k: v for k, v in cfg.__dict__.items()},
                "train_device": "cuda:0",
                "geo_device": "cuda:1",
            })
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  train_device={cfg.train_device}, geo_device={cfg.geo_device}")

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.makedirs(cfg.output_dir, exist_ok=True)

    # tee log
    log_file = open(os.path.join(cfg.output_dir, "sft_via_geoguide.log"), "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    # 把 transformers/trl 等库的 logging 也写入日志文件
    file_handler = logging.FileHandler(
        os.path.join(cfg.output_dir, "sft_via_geoguide.log"), mode="a", encoding="utf-8",
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


def _run(cfg: GeoGuideConfig):
    from trl import SFTTrainer, SFTConfig

    set_seed(cfg.seed)

    # --- 归一化参数 ---
    print("Computing normalization params ...")
    norm_params = compute_normalization_params(cfg.m2c_json, cfg.c2m_json)
    print(f"  math: [{norm_params['math_min']:.6f}, {norm_params['math_max']:.6f}]")
    print(f"  code: [{norm_params['code_min']:.6f}, {norm_params['code_max']:.6f}]")

    # --- 初始化 geo_common ---
    print("Initializing geo_common ...")
    geo_common.init(
        device_str=cfg.geo_device,
        seed=cfg.seed,
        math_model_path=cfg.math_model_path,
        code_model_path=cfg.code_model_path,
    )

    # --- 限制 Trainer 只用训练卡 ---
    # geo_common 已在 geo_device 上初始化完毕，CUDA context 已建立。
    # 缩减 CUDA_VISIBLE_DEVICES 在 runtime 初始化后不影响 device_count()，
    # 所以额外设置 WORLD_SIZE=1 阻止 Trainer 做 DataParallel。
    if cfg.train_device != cfg.geo_device:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        train_vis_idx = cfg.train_device.replace("cuda:", "")  # "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = vis.split(",")[int(train_vis_idx)]
        os.environ["WORLD_SIZE"] = "1"
        print(f"  Shrunk CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} "
              f"(Trainer single-GPU, geo_common stays on {cfg.geo_device})")

    # --- 加载数据集 ---
    print("Loading datasets ...")
    math_train, math_test = get_math_dataset(size=cfg.dataset_pool_size)
    code_train, code_test = get_code_dataset(size=cfg.dataset_pool_size)

    # --- 加载模型和 tokenizer（只从磁盘加载一次） ---
    print(f"Loading model from {cfg.base_model_path} ...")
    model_obj = AutoModelForCausalLM.from_pretrained(cfg.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)

    # --- 预 tokenize 数据池（一次性开销） ---
    print("Pre-tokenizing data pools ...")
    math_train = pre_tokenize_pool(math_train, tokenizer, cfg.max_seq_length)
    code_train = pre_tokenize_pool(code_train, tokenizer, cfg.max_seq_length)
    math_test = pre_tokenize_pool(math_test, tokenizer, cfg.max_seq_length)
    code_test = pre_tokenize_pool(code_test, tokenizer, cfg.max_seq_length)
    print("  Pre-tokenization done.")

    # --- 主循环 ---
    segment_logs = []
    global_step = 0
    segment = 0

    while segment < cfg.max_segments:
        print(f"\n{'='*60}")
        print(f"Segment {segment}/{cfg.max_segments}  (global_step={global_step})")
        print(f"{'='*60}")

        # 计算本段训练步数
        if cfg.max_steps > 0:
            remaining = cfg.max_steps - global_step
            if remaining <= 0:
                print(f"  [STOP] global step budget exhausted: {global_step} >= {cfg.max_steps}")
                break
            segment_steps = min(cfg.rebalance_steps, remaining)
        else:
            segment_steps = cfg.rebalance_steps

        # a. 创建临时 trainer 用于评估当前模型
        dummy_train, _, _ = build_mixed_dataset(
            math_train, code_train, 0.5, min(10, cfg.total_train_size),
        )
        ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint_step{global_step}")
        eval_test = {"math": math_test, "code": code_test}
        eval_trainer = SFTTrainer(
            model=model_obj,
            processing_class=tokenizer,
            train_dataset=dummy_train,
            eval_dataset=eval_test,
            args=SFTConfig(
                do_eval=True,
                eval_strategy="no",
                max_length=cfg.max_seq_length,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                num_train_epochs=1,
                output_dir=ckpt_dir,
                remove_unused_columns=False,
                seed=cfg.seed,
                report_to="none",
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )

        # b. 评估当前模型的 math/code loss
        print("  Evaluating domain losses ...")
        metrics = eval_trainer.evaluate()
        raw_math_loss = metrics["eval_math_loss"]
        raw_code_loss = metrics["eval_code_loss"]
        print(f"  raw losses: math={raw_math_loss:.6f}, code={raw_code_loss:.6f}")

        # c. 归一化
        nx, ny = normalize_losses(raw_math_loss, raw_code_loss, norm_params)
        print(f"  normalized: ({nx:.6f}, {ny:.6f})")

        # c2. early stopping 检查（归一化后 loss）
        check_loss = nx if cfg.geo_target_domain == "math" else ny
        if cfg.target_loss > 0 and check_loss <= cfg.target_loss:
            print(f"  [STOP] normalized target loss reached: {check_loss:.6f} <= {cfg.target_loss:.6f}")
            del eval_trainer
            torch.cuda.empty_cache()
            break

        # d. 测地线方向
        print("  Computing geodesic tangent ...")
        tangent, geo_path, geo_energy, geo_arc = compute_geodesic_tangent((nx, ny), cfg)
        print(f"  tangent = ({tangent[0]:.6f}, {tangent[1]:.6f})")

        # e. 切向量 → 配比
        math_ratio, code_ratio = tangent_to_ratio(tangent, (nx, ny))
        print(f"  ratio: math={math_ratio:.4f}, code={code_ratio:.4f}")

        # 释放 eval_trainer（不释放 model_obj，它被共享）
        del eval_trainer
        torch.cuda.empty_cache()

        # f. 构建混合数据集
        mixed_train, n_math, n_code = build_mixed_dataset(
            math_train, code_train, math_ratio, cfg.total_train_size,
        )
        print(f"  mixed dataset: {n_math} math + {n_code} code = {len(mixed_train)}")

        # g. 创建 SFTTrainer 训练（用 max_steps 控制段长）
        trainer = SFTTrainer(
            model=model_obj,
            processing_class=tokenizer,
            train_dataset=mixed_train,
            eval_dataset=eval_test,
            args=SFTConfig(
                do_eval=True,
                eval_strategy="steps",
                eval_steps=cfg.eval_steps,
                eval_on_start=cfg.eval_on_start,
                max_length=cfg.max_seq_length,
                learning_rate=cfg.learning_rate,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                num_train_epochs=9999,
                max_steps=segment_steps,
                logging_steps=1,
                output_dir=ckpt_dir,
                remove_unused_columns=False,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                lr_scheduler_type=cfg.lr_scheduler_type,
                seed=cfg.seed,
                save_strategy="steps",
                save_steps=cfg.save_steps,
                save_only_model=True,
                save_total_limit=3,
                run_name=f"geoguide-seg{segment}",
                report_to=cfg.report_to,
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )

        # h. 训练
        print(f"  Training segment {segment} for {segment_steps} steps ...")
        trainer.train()

        # h2. 累计步数
        seg_steps_actual = trainer.state.global_step
        global_step += seg_steps_actual

        # i. warm start: 保留训练后的模型对象供下一轮使用
        model_obj = trainer.model

        # j. 保存 segment 日志
        seg_log = {
            "segment": segment,
            "math_ratio": math_ratio,
            "n_math": n_math,
            "n_code": n_code,
            "raw_math_loss": raw_math_loss,
            "raw_code_loss": raw_code_loss,
            "normalized_xy": [nx, ny],
            "tangent": [float(tangent[0]), float(tangent[1])],
            "geodesic_energy": float(geo_energy),
            "geodesic_arc_length": float(geo_arc),
            "geodesic_path": geo_path.tolist(),
            "segment_steps": seg_steps_actual,
            "global_step": global_step,
            "checkpoint_dir": ckpt_dir,
        }
        segment_logs.append(seg_log)

        # 写入完整日志
        full_log = {"config": asdict(cfg), "epochs": segment_logs}
        log_path = os.path.join(cfg.output_dir, "geoguide_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)
        print(f"  log saved to {log_path}")

        # 写入 segment 摘要
        summary_path = os.path.join(cfg.output_dir, "epoch_summary.json")
        summary_entry = {
            "segment": segment,
            "raw_loss": {"math": raw_math_loss, "code": raw_code_loss},
            "normalized": {"math": nx, "code": ny},
            "ratio": {"math": math_ratio, "code": code_ratio},
            "segment_steps": seg_steps_actual,
            "global_step": global_step,
        }
        if segment == 0:
            summary_list = []
        else:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_list = json.load(f)
        summary_list.append(summary_entry)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_list, f, indent=2, ensure_ascii=False)
        print(f"  summary saved to {summary_path}")

        # k. 释放 trainer（model_obj 已单独保留）
        del trainer
        torch.cuda.empty_cache()

        # l. 检查全局步数预算
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            print(f"  [STOP] global step budget: {global_step} >= {cfg.max_steps}")
            break

        segment += 1

    print(f"\n{'='*60}")
    print("GeoGuide training complete.")
    print(f"{'='*60}")


def parse_args() -> GeoGuideConfig:
    defaults = GeoGuideConfig()
    p = argparse.ArgumentParser(description="测地线指导的 SFT 数据配比迭代训练")

    # 模型
    p.add_argument("--base_model_path", type=str, default=defaults.base_model_path)

    # 数据
    p.add_argument("--dataset_pool_size", type=int, default=defaults.dataset_pool_size)
    p.add_argument("--total_train_size", type=int, default=defaults.total_train_size)
    p.add_argument("--math_test_size", type=int, default=defaults.math_test_size)
    p.add_argument("--code_test_size", type=int, default=defaults.code_test_size)

    # 训练超参
    p.add_argument("--max_segments", type=int, default=defaults.max_segments)
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
    p.add_argument("--eval_steps", type=int, default=defaults.eval_steps)
    p.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction, default=defaults.eval_on_start)
    p.add_argument("--save_steps", type=int, default=defaults.save_steps)
    p.add_argument("--train_device", type=str, default=defaults.train_device)

    # 测地线
    p.add_argument("--geo_device", type=str, default=defaults.geo_device)
    p.add_argument("--geo_target_domain", type=str, default=defaults.geo_target_domain, choices=["math", "code"])
    p.add_argument("--geo_target_value", type=float, default=defaults.geo_target_value)
    p.add_argument("--geo_K", type=int, default=defaults.geo_K)
    p.add_argument("--geo_N", type=int, default=defaults.geo_N)
    p.add_argument("--geo_refine_top_k", type=int, default=defaults.geo_refine_top_k)
    p.add_argument("--geo_refine_N", type=int, default=defaults.geo_refine_N)
    p.add_argument("--math_model_path", type=str, default=defaults.math_model_path)
    p.add_argument("--code_model_path", type=str, default=defaults.code_model_path)

    # 归一化数据源
    p.add_argument("--m2c_json", type=str, default=defaults.m2c_json)
    p.add_argument("--c2m_json", type=str, default=defaults.c2m_json)

    # 输出
    p.add_argument("--output_dir", type=str, default=defaults.output_dir)
    p.add_argument("--seed", type=int, default=defaults.seed)

    # wandb
    p.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    p.add_argument("--report_to", type=str, default=defaults.report_to)

    args = p.parse_args()
    return GeoGuideConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
