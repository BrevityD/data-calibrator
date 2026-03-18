"""
离散变分法求解任意点到直线 x = x_target 的最短测地线。

终点 x 固定为 x_target，y 坐标由优化自动确定。
最小化离散能量自动满足横截性条件（终点切向量与直线度规正交）。
"""

import os
import sys
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# 从 geo_common 复用基础设施
from geo_common import (
    init,
    get_device,
    get_dtype,
    metric_tensor_xy,
    metric_tensor_xy_batch,
    metric_mat,
    compute_logdet_grid,
    draw_metric_ellipses,
)


# =========================================================
# 1. 核心优化：离散变分测地线到竖线
# =========================================================
def variational_geodesic_to_line(
    start,
    x_target=0.2,
    y_init=None,
    init_path=None,
    N=50,
    lr=1.0,
    max_iter=300,
    tol=1e-8,
    grad_tol=1e-6,
    verbose=True,
    target_axis=0,
):
    """
    用 L-BFGS 最小化离散能量，求从 start 到目标直线的最短测地线。

    使用 sigmoid 重参数化：优化无约束的 raw_params，通过 sigmoid 映射到 (eps, 1-eps)，
    使 L-BFGS 始终在光滑无约束空间工作。

    参数:
        start: (x0, y0) 起点坐标
        x_target: 目标直线的坐标值
        y_init: 终点自由坐标的初始猜测，默认 = start 中对应的自由坐标
        init_path: 可选 numpy array (N+1, 2)，提供时跳过直线插值
        N: 离散段数（路径有 N+1 个点）
        lr: L-BFGS 学习率
        max_iter: 最大迭代次数
        tol: 收敛容差（能量相对变化）
        grad_tol: 梯度范数收敛容差
        verbose: 是否打印优化过程
        target_axis: 0 → 固定 x（到竖直线 x=x_target），1 → 固定 y（到水平线 y=x_target）

    返回:
        path: numpy array (N+1, 2) 最优路径
        energy: 最终离散能量
        arc_length: 度规弧长
        info: dict 包含优化细节
    """
    fixed_axis = target_axis      # 终点固定的坐标轴
    free_axis = 1 - target_axis   # 终点自由的坐标轴

    x0, y0 = start
    if y_init is None:
        y_init = start[free_axis]

    eps = 1e-4  # 坐标边界余量

    def to_raw(x):
        """inverse sigmoid: 将 (eps, 1-eps) 映射到无约束空间 ℝ"""
        x_c = np.clip(x, eps * 2, 1 - eps * 2)
        return np.log((x_c - eps) / (1 - eps - x_c))

    # 初始路径
    if init_path is not None:
        assert init_path.shape == (N + 1, 2), \
            f"init_path shape {init_path.shape} != ({N + 1}, 2)"
        ip = init_path.copy()
        y_init = ip[-1, free_axis]
    else:
        # start 到终点的直线等分
        ip = np.zeros((N + 1, 2))
        end_pt = [0.0, 0.0]
        end_pt[fixed_axis] = x_target
        end_pt[free_axis] = y_init
        for i in range(N + 1):
            t = i / N
            ip[i, 0] = x0 * (1 - t) + end_pt[0] * t
            ip[i, 1] = y0 * (1 - t) + end_pt[1] * t

    # 自由变量：内部节点 (q_1 ... q_{N-1}) 的 xy + 终点自由坐标
    # 共 2*(N-1) + 1 个标量，在无约束空间 ℝ 中优化
    n_free = 2 * (N - 1) + 1
    params = torch.zeros(n_free, dtype=get_dtype(), device=get_device(), requires_grad=True)

    # 用 inverse sigmoid 填入初始值
    with torch.no_grad():
        for i in range(1, N):
            params[2 * (i - 1)] = to_raw(ip[i, 0])
            params[2 * (i - 1) + 1] = to_raw(ip[i, 1])
        params[-1] = to_raw(y_init)  # 终点自由坐标

    q0 = torch.tensor([x0, y0], dtype=get_dtype(), device=get_device())  # 固定起点

    def build_path(p):
        """从无约束自由变量经 sigmoid 映射构建完整路径 (N+1, 2)"""
        coords = torch.sigmoid(p) * (1 - 2 * eps) + eps  # 映射到 (eps, 1-eps)
        inner = coords[:-1].reshape(N - 1, 2)                        # (N-1, 2)
        free_end = coords[-1:]
        fixed_end = torch.full_like(free_end, x_target)
        # 按 axis 组装终点: endpoint[fixed_axis]=x_target, endpoint[free_axis]=free_end
        endpoint = torch.zeros(2, dtype=free_end.dtype, device=free_end.device)
        endpoint[fixed_axis] = fixed_end[0]
        endpoint[free_axis] = free_end[0]
        return torch.cat([q0.unsqueeze(0), inner, endpoint.unsqueeze(0)], dim=0)

    def compute_energy(p):
        """离散能量 E = Σ (dq^T G(mid) dq)"""
        path = build_path(p)                                    # (N+1, 2)
        dq = path[1:] - path[:-1]                               # (N, 2)
        mids = (0.5 * (path[:-1] + path[1:])).clamp(eps, 1 - eps)  # (N, 2)
        G = metric_tensor_xy_batch(mids)                        # (N, 2, 2)
        Gdq = torch.einsum('bij,bj->bi', G, dq)                # (N, 2)
        E = torch.einsum('bi,bi->', dq, Gdq)                    # scalar
        return E

    # L-BFGS 优化
    optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=20,
                                   line_search_fn='strong_wolfe')

    energy_history = []
    prev_energy = float('inf')
    rel_change = float('inf')

    t_start = time.time()

    for iteration in range(max_iter):
        def closure():
            optimizer.zero_grad()
            E = compute_energy(params)
            E.backward()
            torch.nn.utils.clip_grad_norm_([params], max_norm=1e3)
            return E

        E_val = optimizer.step(closure)
        e = E_val.item()
        energy_history.append(e)

        grad_norm = params.grad.norm().item() if params.grad is not None else float('inf')

        if verbose and (iteration % 20 == 0 or iteration == max_iter - 1):
            print(f"  iter {iteration:4d}  energy = {e:.8f}  |grad| = {grad_norm:.2e}")

        # 收敛检查：同时要求能量变化小 + 梯度范数小
        rel_change = abs(prev_energy - e) / (abs(e) + 1e-12)
        if rel_change < tol and grad_norm < grad_tol and iteration > 10:
            if verbose:
                print(f"  converged at iter {iteration}, rel_change = {rel_change:.2e}, |grad| = {grad_norm:.2e}")
            break
        prev_energy = e

    elapsed = time.time() - t_start

    # 提取最优路径
    with torch.no_grad():
        final_path = build_path(params).detach().cpu().numpy()
        final_energy = compute_energy(params).item()

    arc_length = compute_variational_arc_length(final_path)

    info = {
        'energy_history': energy_history,
        'n_iter': len(energy_history),
        'elapsed_s': elapsed,
        'converged': rel_change < tol,
    }

    if verbose:
        print(f"  final energy   = {final_energy:.8f}")
        print(f"  arc length     = {arc_length:.6f}")
        print(f"  endpoint       = ({final_path[-1, 0]:.6f}, {final_path[-1, 1]:.6f})")
        print(f"  iterations     = {info['n_iter']}")
        print(f"  elapsed        = {elapsed:.3f} s")

    return final_path, final_energy, arc_length, info


# =========================================================
# 2. 弧长计算
# =========================================================
def compute_variational_arc_length(path):
    """
    对离散路径计算度规弧长 L = Σ sqrt(dq^T G(mid) dq)

    参数:
        path: numpy array (M, 2)
    返回:
        arc_length: float
    """
    arc_length = 0.0
    M = len(path)
    # 批量计算所有中点的度规张量
    mids = 0.5 * (path[:-1] + path[1:])                        # (M-1, 2)
    mids = np.clip(mids, 0.0, 1.0)
    mids_t = torch.tensor(mids, dtype=get_dtype(), device=get_device())
    with torch.no_grad():
        G_all = metric_tensor_xy_batch(mids_t).cpu().numpy()    # (M-1, 2, 2)
    dq = path[1:] - path[:-1]                                   # (M-1, 2)
    # ds2_i = dq_i^T G_i dq_i
    Gdq = np.einsum('bij,bj->bi', G_all, dq)                   # (M-1, 2)
    ds2 = np.einsum('bi,bi->b', dq, Gdq)                       # (M-1,)
    mask = ds2 > 0
    arc_length = float(np.sum(np.sqrt(ds2[mask])))
    return arc_length


# =========================================================
# 2b. 批量粗搜索（Adam 并行优化 K 条路径）
# =========================================================
def batch_coarse_search(
    start,
    x_target=0.2,
    y_candidates=None,
    K=12,
    N=50,
    max_iter=2000,
    lr=0.01,
    tol=1e-7,
    verbose=True,
    target_axis=0,
):
    """用 Adam 并行优化 K 条从 start 到目标直线的离散路径。

    所有路径共享同一次 NN forward pass 以提高效率。
    使用直接坐标 + clamp 约束路径点在 (eps, 1-eps) 内，
    避免 sigmoid 在边界附近梯度消失。

    参数:
        start: (x0, y0) 起点坐标
        x_target: 目标直线的坐标值
        y_candidates: 长度为 K 的终点自由坐标初始猜测数组
        K: 并行路径数（y_candidates 为 None 时使用）
        N: 每条路径的离散段数
        max_iter: Adam 最大迭代次数
        lr: Adam 学习率
        tol: 收敛容差（所有路径能量相对变化）
        verbose: 是否打印优化过程
        target_axis: 0 → 固定 x（竖直线），1 → 固定 y（水平线）

    返回:
        candidates: list of dict，按 energy 升序排列
    """
    fixed_axis = target_axis
    free_axis = 1 - target_axis

    if y_candidates is None:
        y_candidates = np.linspace(0.05, 0.95, K)
    else:
        K = len(y_candidates)

    x0, y0 = start
    eps = 1e-4
    n_free = 2 * (N - 1) + 1

    # 初始化 (K, n_free) 参数矩阵（直接坐标空间，Adam + clamp）
    all_params = torch.zeros(K, n_free, dtype=get_dtype(), device=get_device())
    for k, y_c in enumerate(y_candidates):
        end_pt = [0.0, 0.0]
        end_pt[fixed_axis] = x_target
        end_pt[free_axis] = y_c
        for i in range(1, N):
            t = i / N
            all_params[k, 2 * (i - 1)] = x0 * (1 - t) + end_pt[0] * t
            all_params[k, 2 * (i - 1) + 1] = y0 * (1 - t) + end_pt[1] * t
        all_params[k, -1] = y_c  # 终点自由坐标
    all_params = all_params.detach().requires_grad_(True)

    q0 = torch.tensor([x0, y0], dtype=get_dtype(), device=get_device())  # 固定起点

    def build_all_paths(params):
        """从 (K, n_free) 经 clamp 构建 (K, N+1, 2) 路径"""
        coords = params.clamp(eps, 1 - eps)  # (K, n_free)
        inner = coords[:, :-1].reshape(K, N - 1, 2)           # (K, N-1, 2)
        free_end = coords[:, -1:]                               # (K, 1)
        fixed_end = torch.full_like(free_end, x_target)
        # 按 axis 组装终点
        endpoint = torch.zeros(K, 1, 2, dtype=free_end.dtype, device=free_end.device)
        endpoint[:, 0, fixed_axis] = fixed_end[:, 0]
        endpoint[:, 0, free_axis] = free_end[:, 0]
        start_pt = q0.unsqueeze(0).unsqueeze(0).expand(K, 1, 2)   # (K, 1, 2)
        return torch.cat([start_pt, inner, endpoint], dim=1)       # (K, N+1, 2)

    optimizer = torch.optim.Adam([all_params], lr=lr)

    t_start = time.time()
    prev_energies = None
    iteration = 0

    for iteration in range(max_iter):
        optimizer.zero_grad()
        paths = build_all_paths(all_params)                    # (K, N+1, 2)
        dqs = paths[:, 1:] - paths[:, :-1]                    # (K, N, 2)
        mids = (0.5 * (paths[:, :-1] + paths[:, 1:])).clamp(eps, 1 - eps)  # (K, N, 2)

        # 共享 NN forward: 一次处理 K*N 个点
        mids_flat = mids.reshape(K * N, 2)
        G_flat = metric_tensor_xy_batch(mids_flat)             # (K*N, 2, 2)
        G = G_flat.reshape(K, N, 2, 2)

        Gdq = torch.einsum('kbij,kbj->kbi', G, dqs)          # (K, N, 2)
        E = torch.einsum('kbi,kbi->k', dqs, Gdq)             # (K,)

        loss = E.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            all_params.clamp_(eps, 1 - eps)

        if verbose and (iteration % 200 == 0 or iteration == max_iter - 1):
            print(f"  iter {iteration:4d}  energies: min={E.min().item():.6f} max={E.max().item():.6f}")

        # 收敛检查
        if iteration > 20 and prev_energies is not None:
            rel_change = (prev_energies - E).abs() / (E.abs() + 1e-12)
            if rel_change.max().item() < tol:
                if verbose:
                    print(f"  all paths converged at iter {iteration}")
                break
        prev_energies = E.detach().clone()

    elapsed = time.time() - t_start

    # 提取结果
    candidates = []
    with torch.no_grad():
        paths_final = build_all_paths(all_params)              # (K, N+1, 2)
        # 计算能量
        dqs_t = paths_final[:, 1:] - paths_final[:, :-1]
        mids_t = (0.5 * (paths_final[:, :-1] + paths_final[:, 1:])).clamp(eps, 1 - eps)
        mids_flat_t = mids_t.reshape(K * N, 2)
        G_flat_t = metric_tensor_xy_batch(mids_flat_t)
        G_t = G_flat_t.reshape(K, N, 2, 2)
        Gdq_t = torch.einsum('kbij,kbj->kbi', G_t, dqs_t)
        E_pure = torch.einsum('kbi,kbi->k', dqs_t, Gdq_t).cpu().numpy()
        paths_final_np = paths_final.cpu().numpy()

    for k in range(K):
        p = paths_final_np[k]
        arc_len = compute_variational_arc_length(p)
        candidates.append({
            'y_init': float(y_candidates[k]),
            'path': p,
            'energy': float(E_pure[k]),
            'arc_length': float(arc_len),
            'endpoint': [float(p[-1, 0]), float(p[-1, 1])],
            'n_iter': iteration + 1,
            'elapsed_s': elapsed,
            'phase': 'coarse',
        })

    candidates.sort(key=lambda c: c['energy'] if not np.isnan(c['energy']) else float('inf'))

    if verbose:
        print(f"  batch coarse search: {elapsed:.3f}s, {iteration+1} iters")

    return candidates


# =========================================================
# 2c. 多起点搜索 + 两阶段精化
# =========================================================
def multi_start_variational_geodesic_to_line(
    start, x_target=0.2, y_candidates=None, K=12,
    N=50, lr=1.0, max_iter=300, tol=1e-8, grad_tol=1e-5,
    refine_top_k=3, refine_N=100, refine_max_iter=300,
    verbose=True, target_axis=0,
):
    """
    多起点变分测地线搜索，避免局部最优。

    阶段1（粗搜索）：用 batch_coarse_search 对 K 个终点自由坐标候选并行 Adam 优化
    阶段2（精化）：取 top_k 个最优结果，用更多离散点 (refine_N) 进行 L-BFGS 精化

    参数:
        start: (x0, y0) 起点坐标
        x_target: 目标直线的坐标值
        y_candidates: 终点自由坐标候选数组，None 时自动生成
        K: 候选数量
        N: 粗搜索离散段数
        refine_top_k: 精化阶段保留的最优候选数
        refine_N: 精化阶段离散段数
        refine_max_iter: 精化阶段最大迭代次数
        target_axis: 0 → 固定 x（竖直线），1 → 固定 y（水平线）

    返回:
        best_path: numpy array (refine_N+1, 2) 最优路径
        best_energy: 最终离散能量
        best_arc_length: 度规弧长
        best_info: dict 优化细节
        candidates: list of dict，粗搜索阶段所有候选摘要
    """
    if y_candidates is None:
        y_candidates = np.linspace(0.05, 0.95, K)
    else:
        K = len(y_candidates)

    # --- 阶段 1：批量粗搜索（Adam 并行） ---
    t_coarse_start = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-start batch search: {K} candidates, N={N}")
        print(f"{'='*60}")

    candidates = batch_coarse_search(
        start=start, x_target=x_target, y_candidates=y_candidates, K=K,
        N=N, lr=0.01,
        verbose=verbose, target_axis=target_axis,
    )
    t_coarse = time.time() - t_coarse_start

    if verbose:
        print(f"\n{'='*60}")
        print("Coarse results (sorted by arc_length):")
        for i, c in enumerate(candidates):
            tag = " <-- top" if i < refine_top_k else ""
            print(f"  #{i+1} y_init={c['y_init']:.4f}  "
                  f"end=({c['endpoint'][0]:.4f},{c['endpoint'][1]:.4f})  "
                  f"arc={c['arc_length']:.6f}{tag}")

    # --- 阶段 2：精化 top_k ---
    t_refine_start = time.time()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Refining top {refine_top_k} with N={refine_N}")
        print(f"{'='*60}")

    refined = []
    for i in range(min(refine_top_k, len(candidates))):
        c = candidates[i]
        if verbose:
            print(f"\n--- Refining #{i+1}: y_init={c['y_init']:.4f}, "
                  f"coarse arc={c['arc_length']:.6f} ---")

        # 将粗路径重采样到 refine_N+1 个点
        coarse_path = c['path']
        t_old = np.linspace(0, 1, len(coarse_path))
        t_new = np.linspace(0, 1, refine_N + 1)
        resampled = np.column_stack([
            np.interp(t_new, t_old, coarse_path[:, 0]),
            np.interp(t_new, t_old, coarse_path[:, 1]),
        ])

        path, energy, arc_len, info = variational_geodesic_to_line(
            start=start, x_target=x_target, init_path=resampled,
            N=refine_N, lr=lr, max_iter=refine_max_iter,
            verbose=verbose, target_axis=target_axis,
        )
        refined.append({
            'y_init': c['y_init'],
            'path': path,
            'energy': float(energy),
            'arc_length': float(arc_len),
            'endpoint': [float(path[-1, 0]), float(path[-1, 1])],
            'n_iter': info['n_iter'],
            'info': info,
            'phase': 'refined',
        })

    refined.sort(key=lambda c: c['energy'] if not np.isnan(c['energy']) else float('inf'))
    best = refined[0]
    t_refine = time.time() - t_refine_start
    t_total = t_coarse + t_refine

    if verbose:
        print(f"\n{'='*60}")
        print(f"Best after refinement:")
        print(f"  arc_length = {best['arc_length']:.6f}")
        print(f"  endpoint   = ({best['endpoint'][0]:.6f}, {best['endpoint'][1]:.6f})")
        print(f"\n  Timing:")
        print(f"    coarse search : {t_coarse:7.2f}s  ({t_coarse/t_total*100:5.1f}%)")
        print(f"    refinement    : {t_refine:7.2f}s  ({t_refine/t_total*100:5.1f}%)")
        print(f"    total         : {t_total:7.2f}s")
        print(f"{'='*60}")

    return best['path'], best['energy'], best['arc_length'], best['info'], candidates


# =========================================================
# 3. 横截性检验
# =========================================================
def check_transversality(path, target_axis=0):
    """
    检验终点切向量与目标直线的度规正交性。
    target_axis=0: 直线 x=const，切向量为 e_y = (0, 1)
    target_axis=1: 直线 y=const，切向量为 e_x = (1, 0)
    横截性条件：cos<tangent, e_line>_G = 0

    返回:
        cos_angle: float, 归一化度规内积 cos(θ)（越接近 0 越正交）
        tangent: 终点切向量 (unnormalized)
    """
    # 终点切向量：最后一段的差分
    tangent = path[-1] - path[-2]
    endpoint = path[-1]

    x_end = np.clip(float(endpoint[0]), 0.0, 1.0)
    y_end = np.clip(float(endpoint[1]), 0.0, 1.0)
    G = metric_mat(x_end, y_end).cpu().numpy()

    # 目标直线的切方向
    if target_axis == 0:
        e_line = np.array([0.0, 1.0])  # x=const 的切方向是 e_y
    else:
        e_line = np.array([1.0, 0.0])  # y=const 的切方向是 e_x

    # G-范数归一化，使内积 = cos(θ)
    norm_t = np.sqrt(tangent @ G @ tangent)
    norm_e = np.sqrt(e_line @ G @ e_line)
    if norm_t > 0 and norm_e > 0:
        cos_angle = (tangent @ G @ e_line) / (norm_t * norm_e)
    else:
        cos_angle = float('nan')

    return cos_angle, tangent


# =========================================================
# 4. 可视化
# =========================================================
def plot_variational_geodesic_to_line(
    start,
    x_target=0.2,
    y_init=None,
    y_candidates=None,
    K=12,
    N=50,
    lr=1.0,
    max_iter=300,
    tol=1e-8,
    grad_tol=1e-5,
    refine_top_k=3,
    refine_N=200,
    refine_max_iter=500,
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    save_path=None,
    log_path=None,
    target_axis=0,
):
    """求解并可视化从 start 到直线 x=x_target 的变分测地线。

    当提供 y_candidates 时启用多起点搜索（batch_coarse_search + L-BFGS 精化）；
    否则从单一初始路径求解。绘制 log(det G) 热力图、度规椭圆、候选路径和最优路径，
    并进行横截性检验。

    参数:
        start: (x0, y0) 起点坐标
        x_target: 目标直线 x = x_target
        y_init: 单起点模式下终点 y 的初始猜测
        y_candidates: 多起点模式下终点 y 候选数组
        save_path: 图片保存路径（None 则不保存）
        log_path: JSON 日志保存路径

    返回:
        path: numpy array 最优路径
        energy: 最终离散能量
        arc_length: 度规弧长
        info: dict 优化细节
    """
    use_multi = y_candidates is not None

    print("=" * 60)
    print(f"Variational geodesic: {start} -> x = {x_target}")
    print(f"N = {N}, max_iter = {max_iter}, lr = {lr}")
    if use_multi:
        print(f"Multi-start: K={K}, refine_top_k={refine_top_k}, refine_N={refine_N}")
    print("=" * 60)

    candidates_summary = None

    if use_multi:
        path, energy, arc_length, info, candidates_raw = \
            multi_start_variational_geodesic_to_line(
                start=start, x_target=x_target,
                y_candidates=y_candidates, K=K,
                N=N, lr=lr, max_iter=max_iter, tol=tol,
                grad_tol=grad_tol,
                refine_top_k=refine_top_k, refine_N=refine_N,
                refine_max_iter=refine_max_iter, verbose=True,
                target_axis=target_axis,
            )
        candidates_summary = [
            {k: v for k, v in c.items() if k != 'path'}
            for c in candidates_raw
        ]
        candidate_paths = [c['path'] for c in candidates_raw]
    else:
        path, energy, arc_length, info = variational_geodesic_to_line(
            start=start, x_target=x_target, y_init=y_init,
            N=N, lr=lr, max_iter=max_iter, tol=tol,
            grad_tol=grad_tol, verbose=True,
            target_axis=target_axis,
        )
        candidate_paths = []

    # 横截性检验
    trans_ip, tangent = check_transversality(path, target_axis=target_axis)

    line_dir_name = "e_y" if target_axis == 0 else "e_x"
    print("\n--- Transversality check ---")
    print(f"  terminal tangent   = ({tangent[0]:.6f}, {tangent[1]:.6f})")
    print(f"  cos<tangent, {line_dir_name}>_G = {trans_ip:.6e}")
    print(f"  (should be ≈ 0 for orthogonality)")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        print("Computing heatmap ...")
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
        plt.colorbar(im, ax=ax, label='log det G')

    print("Drawing metric ellipses ...")
    draw_metric_ellipses(
        ax=ax,
        n_grid=ellipse_grid,
        scale=ellipse_scale,
        color='white' if show_heatmap else 'black',
        alpha=0.9,
        linewidth=1.0,
    )

    # 目标直线
    line_color = 'cyan' if show_heatmap else 'blue'
    if target_axis == 0:
        ax.axvline(x=x_target, linestyle='--', linewidth=1.5,
                   color=line_color, alpha=0.9, label=f'x = {x_target}')
    else:
        ax.axhline(y=x_target, linestyle='--', linewidth=1.5,
                   color=line_color, alpha=0.9, label=f'y = {x_target}')

    # 候选路径（浅灰色）
    for i, cp in enumerate(candidate_paths):
        ax.plot(cp[:, 0], cp[:, 1], color='gray', linewidth=0.8,
                alpha=0.4, label='candidates' if i == 0 else None)

    # 最优测地线路径（红色）
    ax.plot(path[:, 0], path[:, 1], color='red', linewidth=2.5,
            label='best geodesic')
    # 标注离散节点（内部点），检查点分布是否均匀
    ax.scatter(path[1:-1, 0], path[1:-1, 1], color='red', s=8,
               zorder=5, alpha=0.7, label=f'nodes ({len(path)-2})')

    # 起点 / 终点
    ax.scatter([start[0]], [start[1]], color='cyan', s=80,
               edgecolors='black', zorder=6, label='start')
    ax.scatter([path[-1, 0]], [path[-1, 1]], color='yellow', s=80,
               edgecolors='black', zorder=6,
               label=f'end ({path[-1,0]:.3f}, {path[-1,1]:.3f})')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Variational geodesic to x={x_target}\n"
        f"energy={energy:.6f}  arc_length={arc_length:.6f}  "
        f"transversality={trans_ip:.2e}"
    )
    ax.legend(fontsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()

    # JSON 日志
    if log_path is None and save_path is not None:
        log_path = save_path.rsplit('.', 1)[0] + '_log.json'

    if log_path is not None:
        log_data = {
            "start": list(start),
            "x_target": x_target,
            "N": N,
            "endpoint": [float(path[-1, 0]), float(path[-1, 1])],
            "energy": float(energy),
            "arc_length": float(arc_length),
            "transversality_cos_angle": float(trans_ip),
            "terminal_tangent": [float(tangent[0]), float(tangent[1])],
            "n_iter": info['n_iter'],
            "elapsed_s": info['elapsed_s'],
            "converged": info['converged'],
            "energy_history": [float(e) for e in info['energy_history']],
        }
        if candidates_summary is not None:
            log_data["candidates"] = candidates_summary
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Log saved to: {log_path}")

    return path, energy, arc_length, info


# =========================================================
# 5. stdout/stderr 同时写入日志文件
# =========================================================
class TeeStream:
    """将写入同时转发到原始流和文件。"""

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
# 6. 主程序
# =========================================================
if __name__ == '__main__':
    init(device_str="cuda:4")

    out_dir = 'result/variational_geodesic'
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, 'variational_geodesic.log'), 'w', encoding='utf-8')
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    try:
        start = (1.0, 1.0)
        plot_variational_geodesic_to_line(
            start=start,
            x_target=0.2,
            N=100,
            max_iter=300,
            y_candidates=np.linspace(0.05, 0.95, 20),
            save_path=os.path.join(out_dir, 'variational_geodesic.png'),
        )
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {os.path.join(out_dir, 'variational_geodesic.log')}")

