import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib.patches import Ellipse


# =========================================================
# 0. 随机种子
# =========================================================
def seed_everything(seed: int = 42) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


# =========================================================
# 1. 网络结构
# =========================================================
class SmoothResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return x + self.block(x)


class VectorField(nn.Module):
    def __init__(self):
        super(VectorField, self).__init__()
        hidden_dim = 48

        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU()
        )

        self.res_blocks = nn.Sequential(
            SmoothResidualBlock(hidden_dim),
            SmoothResidualBlock(hidden_dim)
        )

        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


# =========================================================
# 2. 加载模型
# =========================================================
seed_everything(42)

math_model_path = '/public/home/jza/data_calibrate/data_mixture/metric_fit/v1.pth'
code_model_path = '/public/home/jza/data_calibrate/data_mixture/metric_fit/v2.pth'

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

math_model = VectorField().to(device).double()
code_model = VectorField().to(device).double()

math_model.load_state_dict(torch.load(math_model_path, map_location=device))
code_model.load_state_dict(torch.load(code_model_path, map_location=device))

math_model.eval()
code_model.eval()


# =========================================================
# 3. 度规张量 G(x,y) = (J J^T)^(-1)
# =========================================================
def metric_tensor_xy(xy, eps=1e-8):
    """
    xy: torch tensor shape (2,)
    return: G shape (2,2)
    """
    loc = xy.unsqueeze(0)             # (1,2)
    v1 = math_model(loc)              # (1,2)
    v2 = code_model(loc)              # (1,2)
    J = torch.cat([v1, v2], dim=0).T  # (2,2)

    A = J @ J.T
    A = A + eps * torch.eye(2, dtype=dtype, device=device)
    G = torch.linalg.inv(A)
    return G


def metric_tensor_xy_batch(xy_batch, eps=1e-8):
    """
    批量计算度规张量。
    xy_batch: (B, 2) tensor
    返回: (B, 2, 2) tensor
    """
    v1 = math_model(xy_batch)          # (B, 2)
    v2 = code_model(xy_batch)          # (B, 2)
    J = torch.stack([v1, v2], dim=-1)  # (B, 2, 2)
    A = J @ J.transpose(-1, -2)       # (B, 2, 2)
    A = A + eps * torch.eye(2, dtype=dtype, device=device)
    G = torch.linalg.inv(A)           # (B, 2, 2)
    return G


def metric_mat(x, y, eps=1e-8):
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "输入坐标必须在 [0,1]"
    xy = torch.tensor([x, y], dtype=dtype, device=device)
    G = metric_tensor_xy(xy, eps=eps)
    return G.detach()


# =========================================================
# 4. Christoffel symbols
#    Gamma[k,i,j] = Γ^k_{ij}
# =========================================================
jac_metric_tensor_xy = torch.func.jacrev(metric_tensor_xy)

def christoffel_symbols(x, y):
    xy = torch.tensor([x, y], dtype=dtype, device=device, requires_grad=True)

    G = metric_tensor_xy(xy)          # (2,2)
    Ginv = torch.linalg.inv(G)        # (2,2)

    # J[j,l,a] = ∂ g_{jl} / ∂ x^a
    J = jac_metric_tensor_xy(xy)      # (2,2,2)

    # dg[a,j,l]
    dg = J.permute(2, 0, 1).contiguous()

    term = torch.zeros((2, 2, 2), dtype=dtype, device=device)
    for i in range(2):
        for j in range(2):
            for l in range(2):
                term[i, j, l] = dg[i, j, l] + dg[j, i, l] - dg[l, i, j]

    Gamma = 0.5 * torch.einsum('kl,ijl->kij', Ginv, term)
    return Gamma


# =========================================================
# 5. 测地线 ODE
#    Y = [x, y, u, v]
# =========================================================
def geodesic_rhs(t, Y):
    x, y, u, v = Y

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return [0.0, 0.0, 0.0, 0.0]

    Gamma = christoffel_symbols(x, y).detach().cpu().numpy()

    dxdt = u
    dydt = v
    dudt = -(Gamma[0, 0, 0] * u * u + 2.0 * Gamma[0, 0, 1] * u * v + Gamma[0, 1, 1] * v * v)
    dvdt = -(Gamma[1, 0, 0] * u * u + 2.0 * Gamma[1, 0, 1] * u * v + Gamma[1, 1, 1] * v * v)

    return [dxdt, dydt, dudt, dvdt]


# =========================================================
# 6. 边界停止事件
# =========================================================
def hit_boundary(t, Y):
    x, y, u, v = Y
    return min(x, y, 1.0 - x, 1.0 - y)

hit_boundary.terminal = True
hit_boundary.direction = -1


# =========================================================
# 7. 初值测地线
# =========================================================
def solve_geodesic_ivp(x0, y0, u0, v0, T=1.0, max_step=0.01, rtol=1e-6, atol=1e-8):
    sol = solve_ivp(
        geodesic_rhs,
        t_span=(0.0, T),
        y0=[x0, y0, u0, v0],
        method="RK45",
        events=hit_boundary,
        max_step=max_step,
        rtol=rtol,
        atol=atol
    )
    return sol


# =========================================================
# 8. shooting method
# =========================================================
def endpoint_error(vel, start, target, T=1.0, penalty=10.0):
    x0, y0 = start
    x1, y1 = target
    u0, v0 = vel

    sol = solve_geodesic_ivp(x0, y0, u0, v0, T=T)

    # 如果提前撞边界，施加罚项
    if sol.status == 1 and (sol.t_events is not None) and len(sol.t_events[0]) > 0:
        xe = sol.y[0, -1]
        ye = sol.y[1, -1]
        return np.array([
            (xe - x1) * penalty,
            (ye - y1) * penalty
        ], dtype=np.float64)

    xe = sol.y[0, -1]
    ye = sol.y[1, -1]
    return np.array([xe - x1, ye - y1], dtype=np.float64)


def shooting_geodesic(start, target, T=1.0, init_vel=None, method='hybr', maxiter=75):
    x0, y0 = start
    x1, y1 = target

    if init_vel is None:
        init_vel = np.array([x1 - x0, y1 - y0], dtype=np.float64)

    fun = lambda vel: endpoint_error(vel, start, target, T=T)
    options = {'maxfev': maxiter}
    result = root(fun, init_vel, method=method, options=options)
    return result


def robust_shooting_geodesic(start, target, T=1.0, method='hybr',
                              prev_vel=None, err_tol=1e-3):
    """
    多策略 shooting：
      - 若 prev_vel 不为 None，直接用 prev_vel 做热启动
      - 否则依次尝试: 直线方向, 0.5x 缩放, 2x 缩放
    返回误差最小的结果。
    """
    x0, y0 = start
    x1, y1 = target
    straight = np.array([x1 - x0, y1 - y0], dtype=np.float64)

    if prev_vel is not None:
        candidates = [np.array(prev_vel, dtype=np.float64)]
    else:
        candidates = [
            straight.copy(),
            straight * 0.5,
            straight * 2.0,
        ]

    best_result = None
    best_err = np.inf

    for vel in candidates:
        try:
            result = shooting_geodesic(start, target, T=T, init_vel=vel, method=method)
            err = np.linalg.norm(endpoint_error(result.x, start, target, T=T, penalty=1.0))
            if err < best_err:
                best_err = err
                best_result = result
            if err < err_tol:
                break
        except Exception:
            continue

    return best_result


def solve_two_point_geodesic(start, target, T=1.0, init_vel=None, method='hybr'):
    result = shooting_geodesic(start, target, T=T, init_vel=init_vel, method=method)
    x0, y0 = start
    u0, v0 = result.x
    sol = solve_geodesic_ivp(x0, y0, u0, v0, T=T)
    return result, sol


# =========================================================
# 9. 度规椭圆
#    椭圆由 v^T G v = 1 给出
# =========================================================
def metric_ellipse_data(x, y, scale=0.06):
    """
    返回画椭圆所需参数:
        center=(x,y)
        width, height, angle(deg)

    对于 v^T G v = 1:
      若 G 的特征值为 λ1, λ2, 特征向量为 e1, e2
      则椭圆半轴长度为 1/sqrt(λ1), 1/sqrt(λ2)

    scale 只是把椭圆整体缩放到图上合适大小
    """
    G = metric_mat(x, y).cpu().numpy()

    eigvals, eigvecs = np.linalg.eigh(G)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = scale / np.sqrt(eigvals[0])   # 长轴
    b = scale / np.sqrt(eigvals[1])   # 短轴

    vec = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))

    width = 2.0 * a
    height = 2.0 * b
    return width, height, angle, eigvals, eigvecs


def draw_metric_ellipses(ax, n_grid=9, scale=0.06, color='white', alpha=0.85, linewidth=1.0):
    xs = np.linspace(0.08, 0.92, n_grid)
    ys = np.linspace(0.08, 0.92, n_grid)

    for x in xs:
        for y in ys:
            try:
                width, height, angle, eigvals, eigvecs = metric_ellipse_data(x, y, scale=scale)

                e = Ellipse(
                    xy=(x, y),
                    width=width,
                    height=height,
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    alpha=alpha
                )
                ax.add_patch(e)
            except Exception as ex:
                print(f"skip ellipse at ({x:.3f}, {y:.3f}): {ex}")


# =========================================================
# 10. 画 log(det G) heatmap
# =========================================================
def compute_logdet_grid(n=120):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    Z = np.zeros((n, n), dtype=np.float64)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            G = metric_mat(float(x), float(y))
            sign, logabsdet = torch.linalg.slogdet(G)
            if sign.item() <= 0:
                Z[j, i] = np.nan
            else:
                Z[j, i] = logabsdet.item()

    return xs, ys, Z


# =========================================================
# 11. 测地线弧长 (度规意义下)
#     L = ∫ sqrt( dx^T G dx ) 沿曲线
# =========================================================
def compute_geodesic_arc_length(sol):
    """
    给定 solve_ivp 的解 sol，计算度规意义下的弧长。
    sol.y: shape (4, N)  ->  [x, y, u, v]

    弧长 = sum_i sqrt( dq_i^T @ G(q_i) @ dq_i )
    其中 dq_i = q_{i+1} - q_i, q_i = (x_i, y_i)
    """
    xs = sol.y[0]
    ys = sol.y[1]
    n_pts = len(xs)

    arc_length = 0.0
    for i in range(n_pts - 1):
        x_mid = 0.5 * (xs[i] + xs[i + 1])
        y_mid = 0.5 * (ys[i] + ys[i + 1])

        # 确保中点在 [0,1] 范围内
        x_mid = np.clip(x_mid, 0.0, 1.0)
        y_mid = np.clip(y_mid, 0.0, 1.0)

        G = metric_mat(float(x_mid), float(y_mid)).cpu().numpy()

        dq = np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]], dtype=np.float64)
        ds2 = dq @ G @ dq
        if ds2 > 0:
            arc_length += np.sqrt(ds2)

    return arc_length


def compute_euclidean_arc_length(sol):
    """欧氏弧长，用于对比。"""
    xs = sol.y[0]
    ys = sol.y[1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    return np.sum(np.sqrt(dx**2 + dy**2))


# =========================================================
# 12. 单条测地线 + 度规椭圆 + 可选 heatmap
# =========================================================
def plot_geodesic_with_metric_ellipses(
    start,
    target,
    ode_T=1.0,
    init_vel=None,
    method='hybr',
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    save_path=None
):
    result, sol = solve_two_point_geodesic(start, target, T=ode_T, init_vel=init_vel, method=method)

    xs_curve = sol.y[0]
    ys_curve = sol.y[1]

    x0, y0 = start
    x1, y1 = target

    end_err = np.array([xs_curve[-1] - x1, ys_curve[-1] - y1])
    err_norm = np.linalg.norm(end_err)

    print("=" * 60)
    print("shooting result.success =", result.success)
    print("message =", result.message)
    print("initial velocity found =", result.x)
    print("endpoint error =", end_err)
    print("endpoint error norm =", err_norm)
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(
            Z,
            origin='lower',
            extent=[0, 1, 0, 1],
            aspect='equal'
        )
        plt.colorbar(im, ax=ax, label='log det G')

    draw_metric_ellipses(
        ax=ax,
        n_grid=ellipse_grid,
        scale=ellipse_scale,
        color='white' if show_heatmap else 'black',
        alpha=0.9,
        linewidth=1.0
    )

    ax.plot(xs_curve, ys_curve, color='red', linewidth=2.5, label='geodesic')
    ax.scatter([x0], [y0], color='cyan', s=70, label='start')
    ax.scatter([x1], [y1], color='yellow', s=70, label='target')
    ax.scatter([xs_curve[-1]], [ys_curve[-1]], color='lime', s=45, label='reached')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Geodesic with local metric ellipses")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()

    return result, sol


# =========================================================
# 13. 沿测地线抽样，画局部切向方向与主方向
# =========================================================
def draw_tangent_and_easy_direction(ax, sol, n_samples=12, tangent_scale=0.04, easy_dir_scale=0.05):
    """
    在测地线上每隔一些点:
      - 画蓝色箭头: 测地线切向方向
      - 画绿色箭头: 椭圆长轴方向(最便宜方向)
    """
    xs = sol.y[0]
    ys = sol.y[1]
    us = sol.y[2]
    vs = sol.y[3]

    m = len(xs)
    idxs = np.linspace(0, m - 1, n_samples, dtype=int)

    for idx in idxs:
        x = xs[idx]
        y = ys[idx]
        u = us[idx]
        v = vs[idx]

        tang = np.array([u, v], dtype=np.float64)
        tang_norm = np.linalg.norm(tang)
        if tang_norm > 1e-12:
            tang = tang / tang_norm
            ax.arrow(
                x, y,
                tangent_scale * tang[0],
                tangent_scale * tang[1],
                head_width=0.008,
                head_length=0.012,
                fc='blue', ec='blue', alpha=0.9,
                length_includes_head=True
            )

        try:
            _, _, _, eigvals, eigvecs = metric_ellipse_data(x, y, scale=1.0)
            easy_dir = eigvecs[:, 0]
            easy_dir = easy_dir / (np.linalg.norm(easy_dir) + 1e-12)

            ax.arrow(
                x, y,
                easy_dir_scale * easy_dir[0],
                easy_dir_scale * easy_dir[1],
                head_width=0.008,
                head_length=0.012,
                fc='green', ec='green', alpha=0.9,
                length_includes_head=True
            )
        except Exception:
            pass


def plot_geodesic_with_ellipses_and_directions(
    start,
    target,
    ode_T=1.0,
    init_vel=None,
    method='hybr',
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    tangent_samples=12,
    save_path=None
):
    result, sol = solve_two_point_geodesic(start, target, T=ode_T, init_vel=init_vel, method=method)

    xs_curve = sol.y[0]
    ys_curve = sol.y[1]

    x0, y0 = start
    x1, y1 = target

    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(
            Z,
            origin='lower',
            extent=[0, 1, 0, 1],
            aspect='equal'
        )
        plt.colorbar(im, ax=ax, label='log det G')

    draw_metric_ellipses(
        ax=ax,
        n_grid=ellipse_grid,
        scale=ellipse_scale,
        color='white' if show_heatmap else 'black',
        alpha=0.9,
        linewidth=1.0
    )

    ax.plot(xs_curve, ys_curve, color='red', linewidth=2.5, label='geodesic')
    ax.scatter([x0], [y0], color='cyan', s=70, label='start')
    ax.scatter([x1], [y1], color='yellow', s=70, label='target')

    draw_tangent_and_easy_direction(
        ax=ax,
        sol=sol,
        n_samples=tangent_samples,
        tangent_scale=0.04,
        easy_dir_scale=0.05
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Geodesic + metric ellipses + tangent/easy directions")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()

    return result, sol


# =========================================================
# 14. 多条测地线：连到竖线 x = x_target 上
#     v1: 增加弧长计算 + JSON 日志保存
# =========================================================
def plot_geodesic_fan_to_vertical_line(
    start,
    x_target=0.2,
    y_min=0.2,
    y_max=0.6,
    T_num=12,                 # 测地线条数
    ode_T=1.0,                # ODE积分终点时间
    method='hybr',
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    max_step=0.01,
    rtol=1e-6,
    atol=1e-8,
    strict_open_interval=True,
    save_path=None,
    log_path=None,            # JSON 日志保存路径
    resume_log_path='auto'    # 'auto': 自动从 log_path 恢复; None: 不恢复; 或指定路径
):
    """
    从同一个 start 出发，连接到竖线 x = x_target 上的 T_num 个 target 点。
    若 strict_open_interval=True，则 target 的 y 取自开区间 (y_min, y_max)。

    v1 新增:
      - 每条测地线计算度规弧长和欧氏弧长
      - 所有结果保存到 JSON 日志文件
      - resume_log_path: 传入之前的 log JSON 路径，已成功的测地线直接用保存的
        初速度重放轨迹（不重新 shooting），只对未成功或缺失的条目重新求解
    """
    total_start_time = time.time()

    x0, y0 = start

    y_targets = np.linspace(y_min, y_max, T_num)

    # ---- 加载 resume 缓存 ----
    # 'auto' 模式: 自动从 log_path 恢复
    if resume_log_path == 'auto':
        resume_log_path = log_path

    resume_cache = {}
    if resume_log_path is not None and os.path.exists(resume_log_path):
        with open(resume_log_path, 'r', encoding='utf-8') as f:
            prev_log = json.load(f)
        for g in prev_log.get("geodesics", []):
            # 只缓存真正收敛的（误差 < 1e-2）
            if g.get("converged", g.get("endpoint_error_norm", 1e9) < 1e-2) and "initial_velocity" in g:
                key = (round(g["target"][0], 6), round(g["target"][1], 6))
                resume_cache[key] = g
        print(f"Resume: loaded {len(resume_cache)} cached geodesics from {resume_log_path}")

    fig, ax = plt.subplots(figsize=(8, 7))

    # 1) heatmap
    if show_heatmap:
        print("Computing heatmap ...")
        heatmap_start_time = time.time()

        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(
            Z,
            origin='lower',
            extent=[0, 1, 0, 1],
            aspect='equal'
        )
        plt.colorbar(im, ax=ax, label='log det G')

        heatmap_elapsed = time.time() - heatmap_start_time
        print(f"Heatmap done in {heatmap_elapsed:.3f} s")


    # 2) 画椭圆
    print("Drawing metric ellipses ...")
    ellipse_start_time = time.time()

    draw_metric_ellipses(
        ax=ax,
        n_grid=ellipse_grid,
        scale=ellipse_scale,
        color='white' if show_heatmap else 'black',
        alpha=0.9,
        linewidth=1.0
    )

    ellipse_elapsed = time.time() - ellipse_start_time
    print(f"Metric ellipses done in {ellipse_elapsed:.3f} s")

    # 3) 目标竖线
    ax.plot(
        [x_target, x_target],
        [y_min, y_max],
        linestyle='--',
        linewidth=1.5,
        color='cyan' if show_heatmap else 'blue',
        alpha=0.9,
        label=f'x = {x_target}'
    )
    all_results = []
    prev_vel = None  # 上一条成功的速度，作为额外候选

    # 4) tqdm 画多条测地线
    pbar = tqdm(
        enumerate(y_targets, start=1),
        total=len(y_targets),
        desc="Solving geodesics",
        ncols=120
    )

    for idx, y_t in pbar:
        geodesic_start_time = time.time()

        target = (x_target, float(y_t))
        cache_key = (round(target[0], 6), round(target[1], 6))

        # ---- resume: 如果缓存中有成功的结果，直接用保存的初速度重放 ----
        cached = resume_cache.get(cache_key)
        if cached is not None:
            cached_vel = np.array(cached["initial_velocity"], dtype=np.float64)
            shooting_elapsed = 0.0

            ivp_start_time = time.time()
            sol = solve_geodesic_ivp(
                x0, y0, cached_vel[0], cached_vel[1],
                T=ode_T,
                max_step=max_step,
                rtol=rtol,
                atol=atol
            )
            ivp_elapsed = time.time() - ivp_start_time

            xs_curve = sol.y[0]
            ys_curve = sol.y[1]
            end_err = np.array([xs_curve[-1] - target[0], ys_curve[-1] - target[1]])
            err_norm = np.linalg.norm(end_err)

            arc_start_time = time.time()
            metric_arc_len = compute_geodesic_arc_length(sol)
            euclid_arc_len = compute_euclidean_arc_length(sol)
            arc_elapsed = time.time() - arc_start_time

            geodesic_elapsed = time.time() - geodesic_start_time
            final_location = (float(xs_curve[-1]), float(ys_curve[-1]))

            # 构造一个伪 result 对象用于后续兼容
            class _CachedResult:
                def __init__(self, x, success):
                    self.x = x
                    self.success = success
                    self.message = "resumed from cache"
            result = _CachedResult(cached_vel, True)

            pbar.set_postfix({
                "idx": f"{idx}/{len(y_targets)}",
                "target_y": f"{y_t:.3f}",
                "CACHED": True,
                "err": f"{err_norm:.2e}",
            })
            print(f"[{idx}/{len(y_targets)}] target = ({target[0]:.4f}, {target[1]:.4f})  ** RESUMED FROM CACHE **")

        else:
            # ---- 正常 shooting ----
            shooting_start_time = time.time()
            result = robust_shooting_geodesic(
                start=start,
                target=target,
                T=ode_T,
                method=method,
                prev_vel=prev_vel
            )
            shooting_elapsed = time.time() - shooting_start_time

            u0, v0 = result.x

            ivp_start_time = time.time()
            sol = solve_geodesic_ivp(
                x0, y0, u0, v0,
                T=ode_T,
                max_step=max_step,
                rtol=rtol,
                atol=atol
            )
            ivp_elapsed = time.time() - ivp_start_time

            xs_curve = sol.y[0]
            ys_curve = sol.y[1]

            end_err = np.array([xs_curve[-1] - target[0], ys_curve[-1] - target[1]])
            err_norm = np.linalg.norm(end_err)

            arc_start_time = time.time()
            metric_arc_len = compute_geodesic_arc_length(sol)
            euclid_arc_len = compute_euclidean_arc_length(sol)
            arc_elapsed = time.time() - arc_start_time

            geodesic_elapsed = time.time() - geodesic_start_time
            final_location = (float(xs_curve[-1]), float(ys_curve[-1]))

            pbar.set_postfix({
                "idx": f"{idx}/{len(y_targets)}",
                "target_y": f"{y_t:.3f}",
                "success": bool(result.success),
                "err": f"{err_norm:.2e}",
                "arc_L": f"{metric_arc_len:.4f}",
                "time(s)": f"{geodesic_elapsed:.2f}"
            })

            print("=" * 80)
            print(f"[{idx}/{len(y_targets)}] target = ({target[0]:.4f}, {target[1]:.4f})")
            print("shooting success =", result.success)
            print("message =", result.message)
            print("initial velocity found =", result.x)
            print("endpoint error =", end_err)
            print("endpoint error norm =", err_norm)
            print(f"final location      = ({final_location[0]:.6f}, {final_location[1]:.6f})")
            print(f"metric arc length   = {metric_arc_len:.6f}")
            print(f"euclidean arc length= {euclid_arc_len:.6f}")
            print(f"shooting time = {shooting_elapsed:.3f} s")
            print(f"ivp time      = {ivp_elapsed:.3f} s")
            print(f"arc len time  = {arc_elapsed:.3f} s")
            print(f"total line    = {geodesic_elapsed:.3f} s")
            print("=" * 80)

        # 记录成功的速度供下一条使用
        if err_norm < 1e-2:
            prev_vel = result.x.copy()

        ax.plot(xs_curve, ys_curve, linewidth=2.0, alpha=0.95)
        ax.scatter([target[0]], [target[1]], s=35, color='yellow', edgecolors='black', zorder=5)
        ax.scatter([xs_curve[-1]], [ys_curve[-1]], s=20, color='lime', zorder=5)

        all_results.append({
            "index": idx,
            "target": target,
            "result": result,
            "sol": sol,
            "end_err": end_err,
            "err_norm": err_norm,
            "final_location": final_location,
            "metric_arc_length": metric_arc_len,
            "euclidean_arc_length": euclid_arc_len,
            "shooting_time": shooting_elapsed,
            "ivp_time": ivp_elapsed,
            "arc_len_time": arc_elapsed,
            "total_line_time": geodesic_elapsed
        })

    ax.scatter([x0], [y0], color='red', s=80, label='start', zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Fan of geodesics to x={x_target}, y in ({y_min}, {y_max})")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')

    total_elapsed = time.time() - total_start_time


    # ---- 汇总打印 ----
    print("\n" + "#" * 90)
    print("ALL GEODESICS DONE")
    print(f"number of geodesics = {len(y_targets)}")
    print(f"total elapsed time  = {total_elapsed:.3f} s")

    if len(all_results) > 0:
        avg_shooting = np.mean([d["shooting_time"] for d in all_results])
        avg_ivp = np.mean([d["ivp_time"] for d in all_results])
        avg_total = np.mean([d["total_line_time"] for d in all_results])
        avg_err = np.mean([d["err_norm"] for d in all_results])
        max_err = np.max([d["err_norm"] for d in all_results])
        avg_metric_arc = np.mean([d["metric_arc_length"] for d in all_results])
        avg_euclid_arc = np.mean([d["euclidean_arc_length"] for d in all_results])

        print(f"avg shooting time     = {avg_shooting:.3f} s")
        print(f"avg ivp time          = {avg_ivp:.3f} s")
        print(f"avg line time         = {avg_total:.3f} s")
        print(f"avg endpoint error    = {avg_err:.6e}")
        print(f"max endpoint error    = {max_err:.6e}")
        print(f"avg metric arc length = {avg_metric_arc:.6f}")
        print(f"avg euclid arc length = {avg_euclid_arc:.6f}")

        # 逐条汇总表
        print("\n" + "-" * 90)
        print(f"{'idx':>4s}  {'target':>16s}  {'final_loc':>20s}  "
              f"{'err_norm':>10s}  {'metric_L':>10s}  {'euclid_L':>10s}  {'time(s)':>8s}")
        print("-" * 90)
        for d in all_results:
            tx, ty = d["target"]
            fx, fy = d["final_location"]
            print(f"{d['index']:4d}  ({tx:.4f}, {ty:.4f})  "
                  f"({fx:.6f}, {fy:.6f})  "
                  f"{d['err_norm']:10.2e}  "
                  f"{d['metric_arc_length']:10.4f}  "
                  f"{d['euclidean_arc_length']:10.4f}  "
                  f"{d['total_line_time']:8.2f}")
        print("-" * 90)

    print("#" * 90 + "\n")


    # ---- 保存 JSON 日志 ----
    if log_path is None and save_path is not None:
        log_path = save_path.rsplit('.', 1)[0] + '_log.json'

    if log_path is not None:
        log_data = {
            "start": list(start),
            "x_target": x_target,
            "y_min": y_min,
            "y_max": y_max,
            "T_num": T_num,
            "ode_T": ode_T,
            "total_elapsed_s": total_elapsed,
            "geodesics": []
        }
        for d in all_results:
            log_data["geodesics"].append({
                "index": d["index"],
                "target": list(d["target"]),
                "final_location": list(d["final_location"]),
                "initial_velocity": d["result"].x.tolist(),
                "endpoint_error_norm": float(d["err_norm"]),
                "metric_arc_length": float(d["metric_arc_length"]),
                "euclidean_arc_length": float(d["euclidean_arc_length"]),
                "shooting_time_s": float(d["shooting_time"]),
                "ivp_time_s": float(d["ivp_time"]),
                "arc_len_time_s": float(d["arc_len_time"]),
                "total_line_time_s": float(d["total_line_time"]),
                "shooting_success": bool(d["result"].success),
                "converged": bool(d["err_norm"] < 1e-2)
            })

        if len(all_results) > 0:
            log_data["summary"] = {
                "avg_shooting_time_s": float(avg_shooting),
                "avg_ivp_time_s": float(avg_ivp),
                "avg_line_time_s": float(avg_total),
                "avg_endpoint_error": float(avg_err),
                "max_endpoint_error": float(max_err),
                "avg_metric_arc_length": float(avg_metric_arc),
                "avg_euclidean_arc_length": float(avg_euclid_arc)
            }

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Log saved to: {log_path}")

    return all_results


# =========================================================
# 15. 主程序
# =========================================================
if __name__ == "__main__":
    start = (0.9, 0.9)
    target = (0.2, 0.2)

    result, sol = plot_geodesic_with_metric_ellipses(
        start=start,
        target=target,
        ode_T=1.0,
        show_heatmap=True,
        heatmap_n=120,
        ellipse_grid=9,
        ellipse_scale=0.05,
        save_path='geodesic_09_to_02.png'
    )
