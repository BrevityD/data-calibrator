"""
基于实际训练梯度的测地线方向 SFT 训练。

与 sft_via_geoguide.py 不同，本脚本不依赖预训练的 VectorField 模型，
而是通过有限差分直接从训练梯度计算 Jacobian 和度规张量 G，
在度规椭圆上找到最优方向后，直接组合参数梯度更新模型。
"""

import os
import sys
import json
import copy
import time
import logging
import shutil
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    # --- 模型 ---
    base_model_path: str = "~/models/Qwen3-4B"

    # --- 数据 ---
    dataset_pool_size: int = 1000
    math_test_size: int = 100
    code_test_size: int = 100

    # --- 训练超参 ---
    max_epochs: int = 1000
    learning_rate: float = 2e-7
    per_device_train_batch_size: int = 4
    max_seq_length: int = 16384
    adam_beta1: float = 1e-12
    adam_beta2: float = 1e-12
    lr_scheduler_type: str = "constant"
    target_loss: float = -1.0
    save_steps: int = 19
    save_total_limit: int = 3
    train_device: str = "cuda:0"
    eps: float = 1e-8

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
# 1. 归一化参数（复用 sft_via_geoguide.py）
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
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


# =========================================================
# 1b. 预 tokenize（复用 sft_via_geoguide.py）
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
# 2. DataLoader collate
# =========================================================
def sft_collate_fn(batch, pad_token_id: int):
    """Collate for DataLoader: pad input_ids, create attention_mask and labels."""
    input_ids_list = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    completion_masks = [torch.tensor(ex["completion_mask"], dtype=torch.long) for ex in batch]

    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_ids = []
    attn_masks = []
    labels_list = []
    for ids, cmask in zip(input_ids_list, completion_masks):
        pad_len = max_len - ids.size(0)
        padded = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        attn = torch.cat([torch.ones_like(ids), torch.zeros(pad_len, dtype=torch.long)])
        lab = padded.clone()
        # prompt tokens → -100, pad tokens → -100
        lab[:cmask.size(0)][cmask == 0] = -100
        if pad_len > 0:
            lab[-pad_len:] = -100
        padded_ids.append(padded)
        attn_masks.append(attn)
        labels_list.append(lab)

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attn_masks),
        "labels": torch.stack(labels_list),
    }


# =========================================================
# 3. 评估 & 梯度计算
# =========================================================
@torch.no_grad()
def evaluate_losses(model, math_loader, code_loader, device):
    """Forward-only eval on test sets, return average loss per domain."""
    model.eval()
    losses = {}
    for name, loader in [("math", math_loader), ("code", code_loader)]:
        total_loss, total_tokens = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # count non-ignored tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
        losses[name] = total_loss / max(total_tokens, 1)
    model.train()
    return losses["math"], losses["code"]


def compute_domain_gradient(model, dataloader, device):
    """Single batch forward+backward, return cloned per-param gradients."""
    model.train()
    model.zero_grad()
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.clone())
        else:
            grads.append(torch.zeros_like(p))
    model.zero_grad()
    return grads


# =========================================================
# 4. 椭圆最近点 & 度规计算
# =========================================================
def ellipse_closest_to_line(G, target_domain, target_value, cx, cy):
    """Find point on ellipse x^T G x = 1 closest to target line.

    target_domain="math" → vertical line x=target_value (minimize |px - target_value|)
    target_domain="code" → horizontal line y=target_value (minimize |py - target_value|)

    Uses Cholesky: G = L L^T, transform to unit circle, find closest point, transform back.
    """
    G_np = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
    L = np.linalg.cholesky(G_np)  # G = L @ L.T

    # target direction in original space: unit vector toward the target line
    if target_domain == "math":
        d = np.array([target_value - cx, 0.0])
    else:
        d = np.array([0.0, target_value - cy])

    norm_d = np.linalg.norm(d)
    if norm_d < 1e-15:
        # already on target line, pick arbitrary direction
        d = np.array([1.0, 0.0]) if target_domain == "math" else np.array([0.0, 1.0])
        norm_d = 1.0

    # transform to unit-circle space: w = L^T @ d
    w = L.T @ d
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-15:
        return torch.tensor([d[0], d[1]], dtype=torch.float64)

    # closest point on unit circle to direction w
    w_hat = w / w_norm

    # transform back: a = L^{-T} @ w_hat (point on ellipse)
    L_inv_T = np.linalg.inv(L.T)
    a = L_inv_T @ w_hat

    return torch.tensor(a, dtype=torch.float64)


def compute_metric_and_ratio(v1, v2, target_domain, target_value, nx, ny, eps):
    """Build J from v1/v2, compute G, find ellipse direction, compute ratio.

    Returns: (ratio, a, G) where ratio is [r_math, r_code] summing to 1.
    """
    J = np.column_stack([v1, v2])  # (2, 2)
    A = J @ J.T + eps * np.eye(2)
    G = np.linalg.inv(A)

    # find optimal direction on ellipse
    a = ellipse_closest_to_line(G, target_domain, target_value, nx, ny)
    a_np = a.numpy()

    # ratio = |J^{-1} @ a| normalized
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
# 5. 梯度组合 & 更新
# =========================================================
def manual_update_grad(model, grad_math, grad_code, ratio):
    """Set param.grad = ratio[0]*gm + ratio[1]*gc for optimizer.step()."""
    for p, gm, gc in zip(model.parameters(), grad_math, grad_code):
        p.grad = ratio[0] * gm + ratio[1] * gc


# =========================================================
# 6. stdout/stderr tee
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


def _parse_cuda_index(device_str: str) -> int:
    if device_str == "cpu":
        return -1
    return int(device_str.split(":")[-1])


# =========================================================
# 7. 有限差分：tentative step → re-eval → restore
# =========================================================
def finite_diff_direction(model, optimizer, grads, math_eval_loader, code_eval_loader,
                          device, norm_params, nx_before, ny_before):
    """Apply tentative gradient step, re-evaluate, measure Δloss, restore state.

    Returns: v = (Δnx, Δny) — the direction in normalized loss space.
    """
    # backup model + optimizer state
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())

    # apply grads and step
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()
    optimizer.step()
    optimizer.zero_grad()

    # re-evaluate
    raw_math, raw_code = evaluate_losses(model, math_eval_loader, code_eval_loader, device)
    nx_after, ny_after = normalize_losses(raw_math, raw_code, norm_params)

    # restore
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)

    return np.array([nx_after - nx_before, ny_after - ny_before])


# =========================================================
# 8. 主入口
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
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

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

    collate = partial(sft_collate_fn, pad_token_id=pad_token_id)
    math_train_loader = DataLoader(math_train, batch_size=cfg.per_device_train_batch_size,
                                   shuffle=True, collate_fn=collate)
    code_train_loader = DataLoader(code_train, batch_size=cfg.per_device_train_batch_size,
                                   shuffle=True, collate_fn=collate)
    math_eval_loader = DataLoader(math_test, batch_size=cfg.per_device_train_batch_size,
                                  shuffle=False, collate_fn=collate)
    code_eval_loader = DataLoader(code_test, batch_size=cfg.per_device_train_batch_size,
                                  shuffle=False, collate_fn=collate)

    # --- Adam optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2))

    # --- 主循环 ---
    epoch_logs = []
    saved_checkpoints = []
    cost = {"forward_passes": 0, "backward_passes": 0, "eval_passes": 0,
            "wall_time_seconds": 0.0}
    t_start = time.time()

    for epoch in range(cfg.max_epochs):
        t_epoch = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.max_epochs}")
        print(f"{'='*60}")

        # 1. Evaluate
        print("  Evaluating ...")
        raw_math, raw_code = evaluate_losses(model, math_eval_loader, code_eval_loader, device)
        cost["eval_passes"] += 2  # math + code eval
        nx, ny = normalize_losses(raw_math, raw_code, norm_params)
        print(f"  raw: math={raw_math:.6f}, code={raw_code:.6f}")
        print(f"  normalized: ({nx:.6f}, {ny:.6f})")

        # early stopping
        check_loss = nx if cfg.geo_target_domain == "math" else ny
        if cfg.target_loss > 0 and check_loss <= cfg.target_loss:
            print(f"  [STOP] target reached: {check_loss:.6f} <= {cfg.target_loss:.6f}")
            break

        # 2. grad_math + v1 (finite diff)
        print("  Computing math gradient + v1 ...")
        grad_math = compute_domain_gradient(model, math_train_loader, device)
        cost["forward_passes"] += 1; cost["backward_passes"] += 1
        v1 = finite_diff_direction(model, optimizer, grad_math,
                                   math_eval_loader, code_eval_loader,
                                   device, norm_params, nx, ny)
        cost["forward_passes"] += 1; cost["eval_passes"] += 2
        print(f"  v1 = ({v1[0]:.8f}, {v1[1]:.8f})")

        # 3. grad_code + v2 (finite diff)
        print("  Computing code gradient + v2 ...")
        grad_code = compute_domain_gradient(model, code_train_loader, device)
        cost["forward_passes"] += 1; cost["backward_passes"] += 1
        v2 = finite_diff_direction(model, optimizer, grad_code,
                                   math_eval_loader, code_eval_loader,
                                   device, norm_params, nx, ny)
        cost["forward_passes"] += 1; cost["eval_passes"] += 2
        print(f"  v2 = ({v2[0]:.8f}, {v2[1]:.8f})")

        # 4. Metric G, ellipse direction, ratio
        ratio, a, G = compute_metric_and_ratio(
            v1, v2, cfg.geo_target_domain, cfg.geo_target_value, nx, ny, cfg.eps)
        print(f"  ratio: math={ratio[0]:.4f}, code={ratio[1]:.4f}")
        print(f"  ellipse direction a = ({a[0]:.6f}, {a[1]:.6f})")

        # 5. Combined gradient → optimizer step
        manual_update_grad(model, grad_math, grad_code, ratio)
        optimizer.step()
        optimizer.zero_grad()

        epoch_time = time.time() - t_epoch
        cost["wall_time_seconds"] = time.time() - t_start

        # 6. Logging
        log_entry = {
            "epoch": epoch,
            "raw_math_loss": raw_math, "raw_code_loss": raw_code,
            "normalized_xy": [nx, ny],
            "v1": v1.tolist(), "v2": v2.tolist(),
            "ratio": {"math": float(ratio[0]), "code": float(ratio[1])},
            "ellipse_direction": a.tolist(),
            "G": G.tolist() if isinstance(G, np.ndarray) else G,
            "epoch_time_seconds": epoch_time,
        }
        epoch_logs.append(log_entry)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "raw_math_loss": raw_math, "raw_code_loss": raw_code,
                "norm_math": nx, "norm_code": ny,
                "ratio_math": ratio[0], "ratio_code": ratio[1],
                "epoch_time": epoch_time,
            })

        # full log
        full_log = {"config": asdict(cfg), "epochs": epoch_logs}
        with open(os.path.join(cfg.output_dir, "geo_grad_log.json"), "w") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)

        # epoch summary (incremental)
        summary_path = os.path.join(cfg.output_dir, "epoch_summary.json")
        summary_entry = {
            "epoch": epoch,
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

        # cost summary
        with open(os.path.join(cfg.output_dir, "cost_summary.json"), "w") as f:
            json.dump(cost, f, indent=2)

        # checkpoint
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

    # final cost
    cost["wall_time_seconds"] = time.time() - t_start
    with open(os.path.join(cfg.output_dir, "cost_summary.json"), "w") as f:
        json.dump(cost, f, indent=2)

    if use_wandb:
        wandb.finish()

    print(f"\n{'='*60}")
    print("Geo-grad training complete.")
    print(f"  Total epochs: {len(epoch_logs)}")
    print(f"  Cost: {cost}")
    print(f"{'='*60}")


# =========================================================
# 9. CLI
# =========================================================
def parse_args() -> GeoGradConfig:
    defaults = GeoGradConfig()
    p = argparse.ArgumentParser(description="基于实际梯度的测地线方向 SFT 训练")

    p.add_argument("--base_model_path", type=str, default=defaults.base_model_path)
    p.add_argument("--dataset_pool_size", type=int, default=defaults.dataset_pool_size)
    p.add_argument("--math_test_size", type=int, default=defaults.math_test_size)
    p.add_argument("--code_test_size", type=int, default=defaults.code_test_size)
    p.add_argument("--max_epochs", type=int, default=defaults.max_epochs)
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--per_device_train_batch_size", type=int, default=defaults.per_device_train_batch_size)
    p.add_argument("--max_seq_length", type=int, default=defaults.max_seq_length)
    p.add_argument("--adam_beta1", type=float, default=defaults.adam_beta1)
    p.add_argument("--adam_beta2", type=float, default=defaults.adam_beta2)
    p.add_argument("--lr_scheduler_type", type=str, default=defaults.lr_scheduler_type)
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
