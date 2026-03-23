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
    """训练配置。

    与 sft_via_geoguide.py 的 GeoGuideConfig 相比，移除了：
    - geo_device / geo_K / geo_N / geo_refine_* （不需要预训练向量场模型）
    - math_model_path / code_model_path （不需要预训练向量场模型）
    新增：
    - eps: 正则化常数，防止 J J^T 奇异

    结构对齐 sft_via_geoguide.py：
    - 外层循环 (epoch): eval → 有限差分计算度规 → 确定 ratio
    - 内层循环 (rebalance_steps): 用固定 ratio 做多步 combined gradient update
    """
    # --- 模型 ---
    base_model_path: str = "~/models/Qwen3-4B"

    # --- 数据 ---
    dataset_pool_size: int = 1000      # 从 math/code 数据集各加载多少条
    math_test_size: int = 100          # math 测试集大小（由 get_math_dataset 内部控制）
    code_test_size: int = 100          # code 测试集大小（由 get_code_dataset 内部控制）

    # --- 训练超参 ---
    max_epochs: int = 1000             # 外层循环上限（每轮重新计算度规和 ratio）
    rebalance_steps: int = 19          # 内层循环步数：每轮用固定 ratio 跑多少个 effective step
    learning_rate: float = 2e-7        # Adam 学习率
    per_device_train_batch_size: int = 4  # 每次前向/反向的 micro batch 大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数，effective batch = batch_size * accum
    max_seq_length: int = 16384        # tokenize 时的最大序列长度
    adam_beta1: float = 1e-12          # Adam β1 ≈ 0，使动量几乎不累积（等效 SGD）
    adam_beta2: float = 1e-12          # Adam β2 ≈ 0，同上
    lr_scheduler_type: str = "constant"  # 恒定学习率（不衰减）
    max_steps: int = -1                # 全局累计 effective step 上限（-1=不限）
    target_loss: float = -1.0          # early stop 阈值（-1=不启用），检查 geo_target_domain 的归一化 loss
    save_steps: int = 19               # 每隔多少个 epoch 保存一次 checkpoint
    save_total_limit: int = 3          # 最多保留几个 checkpoint（FIFO 淘汰）
    train_device: str = "cuda:0"       # 训练设备
    eps: float = 1e-8                  # 正则化常数，A = J J^T + eps * I 防止奇异

    # --- 测地线目标 ---
    geo_target_domain: str = "math"    # 优化目标领域: "math" → 目标线 x=const, "code" → y=const
    geo_target_value: float = 0.2      # 目标线在归一化坐标中的位置

    # --- 归一化数据源 ---
    m2c_json: str = str(_SCRIPT_DIR / "m2c.json")  # math→code 方向的 loss 矩阵
    c2m_json: str = str(_SCRIPT_DIR / "c2m.json")  # code→math 方向的 loss 矩阵

    # --- 输出 ---
    output_dir: str = str(_SCRIPT_DIR / "result" / "train_with_geo")
    seed: int = SEED

    # --- wandb ---
    wandb_project: str = "data-calibrator"
    report_to: str = "wandb"           # "wandb" 或 "none"


# =========================================================
# 1. 归一化参数（复用 sft_via_geoguide.py）
# =========================================================
def compute_normalization_params(m2c_path: str, c2m_path: str):
    """从 m2c.json / c2m.json 提取全局 min-max 归一化参数。

    两个 JSON 文件记录了不同 math:code 配比下训练后的 eval loss 矩阵。
    从中提取所有 math_loss 和 code_loss 的全局最小/最大值，
    用于将原始 loss 映射到 [0,1]² 归一化坐标空间。
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
        "math_min": min(all_math), "math_max": max(all_math),
        "code_min": min(all_code), "code_max": max(all_code),
    }


def normalize_losses(math_loss: float, code_loss: float, norm_params: dict):
    """原始 loss → [0,1]² 归一化坐标。

    使用 min-max 归一化: x = (loss - min) / (max - min)，并 clip 到 [0, 1]。
    归一化后 (x, y) 分别对应 math 和 code 维度在标准化空间中的位置。
    """
    math_range = norm_params["math_max"] - norm_params["math_min"]
    code_range = norm_params["code_max"] - norm_params["code_min"]
    x = (math_loss - norm_params["math_min"]) / math_range if math_range > 0 else 0.0
    y = (code_loss - norm_params["code_min"]) / code_range if code_range > 0 else 0.0
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


# =========================================================
# 1b. 预 tokenize（复用 sft_via_geoguide.py）
# =========================================================
def pre_tokenize_pool(dataset, tokenizer, max_length: int):
    """对 prompt/completion 格式的数据集做一次性 tokenize + truncate。

    使用 apply_chat_template 将 prompt 和 completion 转为 token ids，
    同时生成 completion_mask 标记哪些 token 属于 completion 部分（用于计算 loss）。
    超过 max_length 的序列会被截断。

    返回的数据集包含 input_ids 和 completion_mask 两个字段。
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
# 2. DataLoader collate
# =========================================================
def sft_collate_fn(batch, pad_token_id: int):
    """DataLoader 的 collate 函数：将变长样本 pad 到同一长度。

    对每个样本：
    1. input_ids 右侧填充 pad_token_id 到 batch 内最大长度
    2. attention_mask: 真实 token 为 1，pad 为 0
    3. labels: 从 input_ids 复制，但 prompt 部分（completion_mask=0）和 pad 部分设为 -100
       （CrossEntropyLoss 会忽略 -100，只在 completion token 上计算 loss）

    返回 dict: {input_ids, attention_mask, labels}，每个都是 (batch_size, max_len) 的 tensor。
    """
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
    """在测试集上做 forward-only 评估，返回 math 和 code 的平均 loss。

    对每个 domain 的 DataLoader 遍历所有 batch，累加 loss * n_tokens，
    最后除以总 token 数得到 token 级平均 loss。
    使用 (labels != -100) 统计有效 token 数（排除 prompt 和 pad）。

    注意：评估前切换到 eval 模式（关闭 dropout 等），评估后恢复 train 模式。
    """
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
    """对单个 domain 的一个 batch 做 forward+backward，返回克隆的逐参数梯度。

    流程：
    1. 取 DataLoader 的第一个 batch
    2. 前向传播计算 loss
    3. 反向传播计算梯度
    4. 克隆每个参数的 .grad（深拷贝，与计算图解耦）
    5. 清零模型梯度，避免影响后续计算

    返回 list[Tensor]，与 model.parameters() 一一对应。
    无梯度的参数（如 frozen 层）返回零张量。
    """
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
    """在度规椭圆 x^T G x = 1 上找到距离目标直线最近的点。

    目标直线：
    - target_domain="math" → 竖直线 x = target_value，法向量 n = [1, 0]
    - target_domain="code" → 水平线 y = target_value，法向量 n = [0, 1]

    算法：
    椭圆 x^T G x = 1 上沿方向 n 的极值点为：
        a = ± G^{-1} n / sqrt(n^T G^{-1} n)
    这是 Lagrange 乘子法的解析解（最大化/最小化 n^T x subject to x^T G x = 1）。
    取符号使 a 朝向目标直线（即 n^T a 与 target_value - current_value 同号）。
    """
    G_np = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
    G_inv = np.linalg.inv(G_np)  # G^{-1} = J J^T + εI（即协方差矩阵 A）

    # 目标直线的法向量
    if target_domain == "math":
        n = np.array([1.0, 0.0])
        sign_ref = target_value - cx  # 需要朝 x 增大还是减小的方向
    else:
        n = np.array([0.0, 1.0])
        sign_ref = target_value - cy

    # 椭圆上沿 n 方向的极值点: a = G^{-1} n / sqrt(n^T G^{-1} n)
    G_inv_n = G_inv @ n
    denom = np.sqrt(n @ G_inv_n)
    if denom < 1e-15:
        return torch.tensor(n, dtype=torch.float64)

    a = G_inv_n / denom

    # 取符号使 a 朝向目标直线
    if sign_ref < 0:
        a = -a

    return torch.tensor(a, dtype=torch.float64)


def compute_metric_and_ratio(v1, v2, target_domain, target_value, nx, ny, eps):
    """从有限差分方向 v1/v2 构建 Jacobian，计算度规张量 G，求椭圆最优方向和配比。

    算法步骤：
    1. 构建 Jacobian: J = [v1 | v2]，列向量分别是 math/code 梯度在归一化 loss 空间的效果
    2. 计算度规张量: A = J @ J^T + eps*I, G = A^{-1}
       - A 是 loss 空间的协方差矩阵，G 是其逆（度规张量）
       - eps 正则化防止 J 秩不足时 A 奇异
    3. 在度规椭圆 x^T G x = 1 上找距目标线最近的点 a
    4. 通过 J^{-1} @ a 将 loss 空间方向映射回 ratio 空间
       - delta_ratio[0] 对应 math 梯度的权重，delta_ratio[1] 对应 code
    5. 取绝对值并归一化为和为 1 的配比

    返回: (ratio, a, G)
    - ratio: [r_math, r_code]，用于组合 grad_math 和 grad_code
    - a: 椭圆上的最优方向向量
    - G: 2×2 度规张量
    """
    J = np.column_stack([v1, v2])  # (2, 2): 列向量是各 domain 梯度的 loss 空间效果
    A = J @ J.T + eps * np.eye(2)  # 正则化协方差矩阵
    G = np.linalg.inv(A)           # 度规张量 G = (J J^T + εI)^{-1}

    # 在度规椭圆上找距目标线最近的点
    a = ellipse_closest_to_line(G, target_domain, target_value, nx, ny)
    a_np = a.numpy()

    # 将 loss 空间方向 a 映射回 ratio 空间: delta_ratio = J^{-1} @ a
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("[WARN] Jacobian singular, using pseudo-inverse")
        J_inv = np.linalg.pinv(J)

    delta_ratio = J_inv @ a_np
    # 取绝对值并归一化（负系数表示该 domain 梯度方向需要反转，取 abs 后仍保留其贡献）
    abs_delta = np.abs(delta_ratio)
    s = abs_delta.sum()
    if s < 1e-12:
        ratio = np.array([0.5, 0.5])  # 退化情况：均匀配比
    else:
        ratio = abs_delta / s

    return ratio, a_np, G


# =========================================================
# 5. 梯度组合 & 更新
# =========================================================
def manual_update_grad(model, grad_math, grad_code, ratio):
    """将组合梯度写入模型参数的 .grad 字段，供 optimizer.step() 使用。

    combined_grad = ratio[0] * grad_math + ratio[1] * grad_code

    这样 Adam optimizer 会用组合后的梯度做一步更新，
    等效于按 ratio 配比混合两个 domain 的训练信号。
    """
    for p, gm, gc in zip(model.parameters(), grad_math, grad_code):
        p.grad = ratio[0] * gm + ratio[1] * gc


# =========================================================
# 6. stdout/stderr tee
# =========================================================
class TeeStream:
    """同时写入原始流和日志文件的流包装器，用于捕获 stdout/stderr 到日志。"""
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
    """从 'cuda:N' 提取 N，'cpu' 返回 -1。"""
    if device_str == "cpu":
        return -1
    return int(device_str.split(":")[-1])


# =========================================================
# 7. 有限差分：tentative step → re-eval → restore
# =========================================================
def finite_diff_direction(model, optimizer, grads, math_eval_loader, code_eval_loader,
                          device, norm_params, nx_before, ny_before):
    """有限差分法测量单个 domain 梯度在归一化 loss 空间中的效果方向。

    流程：
    1. 深拷贝模型权重和优化器状态（备份）
    2. 将 grads 写入 param.grad，执行一步 optimizer.step()（tentative step）
    3. 重新评估 math/code loss 并归一化
    4. 计算 Δ(nx, ny) = (nx_after - nx_before, ny_after - ny_before)
    5. 恢复模型和优化器到备份状态

    返回: v = (Δnx, Δny)，表示该 domain 梯度一步更新后在归一化 loss 空间的位移方向。
    这个方向向量将作为 Jacobian J 的列向量。

    计算开销：1 次 optimizer step + 2 次 eval（math + code）+ 状态备份/恢复。
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
    # beta1 ≈ 0, beta2 ≈ 0 使 Adam 退化为近似 SGD（不累积动量和二阶矩）
    # 配合 constant LR，与 sft_via_geoguide.py 的训练设置一致
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2))

    # --- 主循环 ---
    # 外层 (epoch): eval → 有限差分算 v1/v2 → 算 ratio
    # 内层 (rebalance_steps): 用固定 ratio 顺序遍历 math/code batch，
    #   每 gradient_accumulation_steps 个 micro batch 做一次 optimizer.step()
    # batch 遍历方式与 vanilla SFT 对齐：0-18, 19-38, 39-57, 58-63+0-13, ...
    epoch_logs = []
    saved_checkpoints = []
    cost = {"forward_passes": 0, "backward_passes": 0, "eval_passes": 0,
            "wall_time_seconds": 0.0}
    t_start = time.time()
    global_step = 0
    # 持久化 DataLoader 迭代器，跨 epoch 连续遍历（模拟 vanilla SFT 的数据遍历顺序）
    math_train_iter = iter(math_train_loader)
    code_train_iter = iter(code_train_loader)

    def _next_batch(loader_iter, loader):
        """从迭代器取下一个 batch，耗尽则重建迭代器（新 epoch shuffle）。"""
        try:
            return next(loader_iter), loader_iter
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter), loader_iter

    for epoch in range(cfg.max_epochs):
        t_epoch = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.max_epochs}  (global_step={global_step})")
        print(f"{'='*60}")

        # 检查全局步数预算
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            print(f"  [STOP] global step budget exhausted: {global_step} >= {cfg.max_steps}")
            break

        # 计算本 epoch 内层步数（受 max_steps 约束）
        if cfg.max_steps > 0:
            remaining = cfg.max_steps - global_step
            epoch_steps = min(cfg.rebalance_steps, remaining)
        else:
            epoch_steps = cfg.rebalance_steps

        # 1. Evaluate: 在测试集上评估当前模型的 math/code loss
        print("  Evaluating ...")
        raw_math, raw_code = evaluate_losses(model, math_eval_loader, code_eval_loader, device)
        cost["eval_passes"] += 2  # math eval + code eval 各一次
        nx, ny = normalize_losses(raw_math, raw_code, norm_params)
        print(f"  raw: math={raw_math:.6f}, code={raw_code:.6f}")
        print(f"  normalized: ({nx:.6f}, {ny:.6f})")

        # early stopping: 检查目标 domain 的归一化 loss 是否已达标
        check_loss = nx if cfg.geo_target_domain == "math" else ny
        if cfg.target_loss > 0 and check_loss <= cfg.target_loss:
            print(f"  [STOP] target reached: {check_loss:.6f} <= {cfg.target_loss:.6f}")
            break

        # 2. 计算 math 梯度 + 有限差分方向 v1
        #    grad_math: 在 math batch 上的参数梯度（1 forward + 1 backward）
        #    v1: tentative step 后在归一化 loss 空间的位移（1 forward + 2 eval）
        print("  Computing math gradient + v1 ...")
        grad_math = compute_domain_gradient(model, math_train_loader, device)
        cost["forward_passes"] += 1; cost["backward_passes"] += 1
        v1 = finite_diff_direction(model, optimizer, grad_math,
                                   math_eval_loader, code_eval_loader,
                                   device, norm_params, nx, ny)
        cost["forward_passes"] += 1; cost["eval_passes"] += 2  # tentative step 中的 eval
        print(f"  v1 = ({v1[0]:.8f}, {v1[1]:.8f})")

        # 3. 计算 code 梯度 + 有限差分方向 v2（同上）
        print("  Computing code gradient + v2 ...")
        grad_code = compute_domain_gradient(model, code_train_loader, device)
        cost["forward_passes"] += 1; cost["backward_passes"] += 1
        v2 = finite_diff_direction(model, optimizer, grad_code,
                                   math_eval_loader, code_eval_loader,
                                   device, norm_params, nx, ny)
        cost["forward_passes"] += 1; cost["eval_passes"] += 2  # tentative step 中的 eval
        print(f"  v2 = ({v2[0]:.8f}, {v2[1]:.8f})")

        # 4. 构建 Jacobian J=[v1|v2]，计算度规 G=(JJ^T+εI)^{-1}，
        #    在度规椭圆上找最优方向 a，通过 J^{-1} 映射为 ratio
        ratio, a, G = compute_metric_and_ratio(
            v1, v2, cfg.geo_target_domain, cfg.geo_target_value, nx, ny, cfg.eps)
        print(f"  ratio: math={ratio[0]:.4f}, code={ratio[1]:.4f}")
        print(f"  ellipse direction a = ({a[0]:.6f}, {a[1]:.6f})")

        # 5. 内层循环：用固定 ratio 跑 epoch_steps 个 effective step
        #    每个 effective step = gradient_accumulation_steps 个 micro batch
        #    顺序遍历 DataLoader，耗尽后重建（与 vanilla SFT 数据遍历对齐）
        print(f"  Training {epoch_steps} steps with ratio math={ratio[0]:.4f}, code={ratio[1]:.4f} ...")
        accum = cfg.gradient_accumulation_steps
        model.train()
        for step_i in range(epoch_steps):
            optimizer.zero_grad()
            for micro in range(accum):
                # 取 math micro batch
                math_batch, math_train_iter = _next_batch(math_train_iter, math_train_loader)
                m_ids = math_batch["input_ids"].to(device)
                m_attn = math_batch["attention_mask"].to(device)
                m_lab = math_batch["labels"].to(device)
                m_out = model(input_ids=m_ids, attention_mask=m_attn, labels=m_lab)
                (m_out.loss / accum * ratio[0]).backward()
                cost["forward_passes"] += 1; cost["backward_passes"] += 1

                # 取 code micro batch
                code_batch, code_train_iter = _next_batch(code_train_iter, code_train_loader)
                c_ids = code_batch["input_ids"].to(device)
                c_attn = code_batch["attention_mask"].to(device)
                c_lab = code_batch["labels"].to(device)
                c_out = model(input_ids=c_ids, attention_mask=c_attn, labels=c_lab)
                (c_out.loss / accum * ratio[1]).backward()
                cost["forward_passes"] += 1; cost["backward_passes"] += 1

            optimizer.step()
            global_step += 1

        epoch_time = time.time() - t_epoch
        cost["wall_time_seconds"] = time.time() - t_start

        # 6. 日志记录
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_steps": epoch_steps,
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

        # 完整日志（每次覆盖写入，包含全部 config + 所有 epoch 数据）
        full_log = {"config": asdict(cfg), "epochs": epoch_logs}
        with open(os.path.join(cfg.output_dir, "geo_grad_log.json"), "w") as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False)

        # epoch 摘要（增量追加，便于快速查看训练进展）
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

        # 计算开销摘要（每次覆盖，便于随时查看当前累计开销）
        with open(os.path.join(cfg.output_dir, "cost_summary.json"), "w") as f:
            json.dump(cost, f, indent=2)

        # checkpoint 保存（FIFO 淘汰旧 checkpoint）
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
    print(f"  Total epochs: {len(epoch_logs)}, global_step: {global_step}")
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
