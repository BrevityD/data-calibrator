"""
测地线指导的 SFT 数据配比迭代训练。

每个 epoch：评估 math/code loss → 归一化到 [0,1]² → 计算测地线方向 → 调整配比 → 重新训练。
"""

import os
import sys
import json
import logging
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
    total_train_size: int = 1000       # math + code 训练集总量
    math_test_size: int = 100
    code_test_size: int = 100

    # --- 训练超参 ---
    num_epochs: int = 5
    learning_rate: float = 2e-7
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 16384
    optim: str = "adamw_torch"
    adam_beta1: float = 1e-12
    adam_beta2: float = 1e-12
    lr_scheduler_type: str = "constant"
    max_steps: int = -1                # -1 表示由 num_train_epochs 控制
    inner_num_train_epochs: int = 100  # 每轮 SFTTrainer 内部 epoch 数（受 max_steps 截断）
    eval_steps: int = 1
    eval_on_start: bool = True
    save_steps: int = 500
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

    # --- 初始配比 ---
    init_math_ratio: float = 0.5

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
# 4. 评估
# =========================================================
def evaluate_domain_losses(trainer, math_test, code_test):
    """分别在 math_test / code_test 上 evaluate，返回两个 eval_loss。

    需要先用 trainer 内部的 _prepare_dataset 对数据集做 tokenization，
    否则 collator 会因缺少 input_ids 报错。
    """
    prepared_math = trainer._prepare_dataset(
        math_test, trainer.processing_class, trainer.args,
        packing=False, formatting_func=None, dataset_name="eval_math",
    )
    prepared_code = trainer._prepare_dataset(
        code_test, trainer.processing_class, trainer.args,
        packing=False, formatting_func=None, dataset_name="eval_code",
    )
    math_metrics = trainer.evaluate(eval_dataset=prepared_math, metric_key_prefix="eval_math")
    code_metrics = trainer.evaluate(eval_dataset=prepared_code, metric_key_prefix="eval_code")
    return math_metrics["eval_math_loss"], code_metrics["eval_code_loss"]


# =========================================================
# 5. stdout/stderr tee
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
# 6. 主循环
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

    # --- 加载数据集 ---
    print("Loading datasets ...")
    math_train, math_test = get_math_dataset(size=cfg.total_train_size)
    code_train, code_test = get_code_dataset(size=cfg.total_train_size)

    # --- 主循环 ---
    model_path = cfg.base_model_path
    math_ratio = cfg.init_math_ratio
    epoch_logs = []

    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.num_epochs}  |  math_ratio = {math_ratio:.4f}")
        print(f"{'='*60}")

        # a. 构建混合数据集
        mixed_train, n_math, n_code = build_mixed_dataset(
            math_train, code_train, math_ratio, cfg.total_train_size,
        )
        print(f"  mixed dataset: {n_math} math + {n_code} code = {len(mixed_train)}")

        # b. checkpoint 目录
        ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}")

        # c. 创建 SFTTrainer（warm start: model_path 指向上一轮 checkpoint）
        trainer = SFTTrainer(
            model=model_path,
            train_dataset=mixed_train,
            eval_dataset=math_test,  # 默认 eval 用 math_test
            args=SFTConfig(
                do_eval=True,
                eval_strategy="steps",
                eval_steps=cfg.eval_steps,
                eval_on_start=cfg.eval_on_start,
                max_length=cfg.max_seq_length,
                learning_rate=cfg.learning_rate,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                num_train_epochs=cfg.inner_num_train_epochs,
                max_steps=cfg.max_steps,
                logging_steps=1,
                output_dir=ckpt_dir,
                optim=cfg.optim,
                remove_unused_columns=False,
                adam_beta1=cfg.adam_beta1,
                adam_beta2=cfg.adam_beta2,
                lr_scheduler_type=cfg.lr_scheduler_type,
                seed=cfg.seed,
                save_strategy="steps",
                save_steps=cfg.save_steps,
                save_only_model=True,
                run_name=f"geoguide-epoch{epoch}",
                report_to=cfg.report_to,
            ),
        )

        # d. 训练
        print(f"  Training (model_path={model_path}) ...")
        trainer.train()

        # e. 评估
        print("  Evaluating domain losses ...")
        raw_math_loss, raw_code_loss = evaluate_domain_losses(trainer, math_test, code_test)
        print(f"  raw losses: math={raw_math_loss:.6f}, code={raw_code_loss:.6f}")

        # f. 归一化
        nx, ny = normalize_losses(raw_math_loss, raw_code_loss, norm_params)
        print(f"  normalized: ({nx:.6f}, {ny:.6f})")

        # g. 测地线方向
        print("  Computing geodesic tangent ...")
        tangent, geo_path, geo_energy, geo_arc = compute_geodesic_tangent((nx, ny), cfg)
        print(f"  tangent = ({tangent[0]:.6f}, {tangent[1]:.6f})")

        # h. 切向量 → 配比
        new_math_ratio, new_code_ratio = tangent_to_ratio(tangent, (nx, ny))
        print(f"  new ratio: math={new_math_ratio:.4f}, code={new_code_ratio:.4f}")

        # i. 保存 epoch 日志
        epoch_log = {
            "epoch": epoch,
            "model_path": model_path,
            "math_ratio": math_ratio,
            "n_math": n_math,
            "n_code": n_code,
            "raw_math_loss": raw_math_loss,
            "raw_code_loss": raw_code_loss,
            "normalized_xy": [nx, ny],
            "tangent": [float(tangent[0]), float(tangent[1])],
            "geodesic_energy": float(geo_energy),
            "geodesic_arc_length": float(geo_arc),
            "new_math_ratio": new_math_ratio,
            "new_code_ratio": new_code_ratio,
            "checkpoint_dir": ckpt_dir,
        }
        epoch_logs.append(epoch_log)

        # 写入完整日志
        full_log = {"config": asdict(cfg), "epochs": epoch_logs}
        log_path = os.path.join(cfg.output_dir, "geoguide_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)
        print(f"  log saved to {log_path}")

        # j. 释放显存
        del trainer
        torch.cuda.empty_cache()

        # k. warm start: 下一轮用本轮 checkpoint
        # SFTTrainer save_strategy="epoch" 会保存到 ckpt_dir/checkpoint-{step}
        # 找到最新的 checkpoint 子目录
        ckpt_subdirs = sorted(
            [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if ckpt_subdirs:
            model_path = os.path.join(ckpt_dir, ckpt_subdirs[-1])
        else:
            model_path = ckpt_dir
        print(f"  next model_path = {model_path}")

        # 更新配比
        math_ratio = new_math_ratio

    print(f"\n{'='*60}")
    print("GeoGuide training complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    cfg = GeoGuideConfig(
        base_model_path="/public/home/jza/share_model/Qwen/Qwen3-1.7B",
        total_train_size=100,
        num_epochs=2,
        max_steps=10,
        save_steps=10,
        eval_steps=10,
        eval_on_start=False,
        train_device="cuda:4",
        geo_device="cuda:5",
        report_to="none",
    )
    main(cfg)
