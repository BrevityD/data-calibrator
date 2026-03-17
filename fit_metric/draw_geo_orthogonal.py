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
    loc = xy.unsqueeze(0)
    v1 = math_model(loc)
    v2 = code_model(loc)
    J = torch.cat([v1, v2], dim=0).T
    A = J @ J.T
    A = A + eps * torch.eye(2, dtype=dtype, device=device)
    G = torch.linalg.inv(A)
    return G


def metric_mat(x, y, eps=1e-8):
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "输入坐标必须在 [0,1]"
    xy = torch.tensor([x, y], dtype=dtype, device=device)
    G = metric_tensor_xy(xy, eps=eps)
    return G.detach()


# =========================================================
# 4. Christoffel symbols
# =========================================================
jac_metric_tensor_xy = torch.func.jacrev(metric_tensor_xy)


def christoffel_symbols(x, y):
    xy = torch.tensor([x, y], dtype=dtype, device=device, requires_grad=True)
    G = metric_tensor_xy(xy)
    Ginv = torch.linalg.inv(G)
    J = jac_metric_tensor_xy(xy)
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
# 8. 度规椭圆
# =========================================================
def metric_ellipse_data(x, y, scale=0.06):
    G = metric_mat(x, y).cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    a = scale / np.sqrt(eigvals[0])
    b = scale / np.sqrt(eigvals[1])
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
                width, height, angle, _, _ = metric_ellipse_data(x, y, scale=scale)
                e = Ellipse(
                    xy=(x, y), width=width, height=height, angle=angle,
                    fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha
                )
                ax.add_patch(e)
            except Exception as ex:
                print(f"skip ellipse at ({x:.3f}, {y:.3f}): {ex}")


# =========================================================
# 9. log(det G) heatmap
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
# 10. 弧长
# =========================================================
def compute_geodesic_arc_length(sol):
    xs = sol.y[0]
    ys = sol.y[1]
    n_pts = len(xs)
    arc_length = 0.0
    for i in range(n_pts - 1):
        x_mid = np.clip(0.5 * (xs[i] + xs[i + 1]), 0.0, 1.0)
        y_mid = np.clip(0.5 * (ys[i] + ys[i + 1]), 0.0, 1.0)
        G = metric_mat(float(x_mid), float(y_mid)).cpu().numpy()
        dq = np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]], dtype=np.float64)
        ds2 = dq @ G @ dq
        if ds2 > 0:
            arc_length += np.sqrt(ds2)
    return arc_length


def compute_euclidean_arc_length(sol):
    xs = sol.y[0]
    ys = sol.y[1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    return np.sum(np.sqrt(dx**2 + dy**2))


# =========================================================
# 11. 度规正交方向计算
#     竖线 x=a 切向量 t=(0,1)
#     正交条件: n^T G t = 0
#     => n1*G12 + n2*G22 = 0  =>  n ∝ (G22, -G12)
# =========================================================
def compute_metric_orthogonal_to_vertical(x, y, direction='right'):
    """
    计算点 (x,y) 处相对于竖线 x=const 在度规意义下的单位正交方向。

    返回度规归一化后的方向向量 (n1, n2)。
    direction='right' 保证 n1>0，'left' 保证 n1<0。
    """
    G = metric_mat(x, y).cpu().numpy()
    n = np.array([G[1, 1], -G[0, 1]], dtype=np.float64)

    # 方向选择
    if direction == 'right' and n[0] < 0:
        n = -n
    elif direction == 'left' and n[0] > 0:
        n = -n

    # 度规归一化: ||n||_G = sqrt(n^T G n)
    norm_G = np.sqrt(n @ G @ n)
    if norm_G < 1e-15:
        return (0.0, 0.0)
    n = n / norm_G
    return (float(n[0]), float(n[1]))


# =========================================================
# 12. 沿竖线正交发射测地线扇
# =========================================================
def _shoot_one_direction(ax, x_line, y_points, speed, ode_T, direction,
                         max_step, rtol, atol, arrow_scale, all_results):
    """对一个方向 (right/left) 发射所有测地线并画图。"""
    pbar = tqdm(
        enumerate(y_points, start=1),
        total=len(y_points),
        desc=f"Geodesics ({direction})",
        ncols=120
    )

    for idx, y_i in pbar:
        t0 = time.time()

        n1, n2 = compute_metric_orthogonal_to_vertical(x_line, float(y_i), direction=direction)
        u0 = n1 * speed
        v0 = n2 * speed

        sol = solve_geodesic_ivp(x_line, float(y_i), u0, v0,
                                 T=ode_T, max_step=max_step, rtol=rtol, atol=atol)

        metric_arc = compute_geodesic_arc_length(sol)
        euclid_arc = compute_euclidean_arc_length(sol)
        elapsed = time.time() - t0

        xs_c = sol.y[0]
        ys_c = sol.y[1]

        # 画曲线
        ax.plot(xs_c, ys_c, linewidth=2.0, alpha=0.9)
        # 起点
        ax.scatter([x_line], [y_i], s=35, color='cyan', edgecolors='black', zorder=5)
        # 正交方向箭头
        ax.arrow(x_line, float(y_i), arrow_scale * n1, arrow_scale * n2,
                 head_width=0.008, head_length=0.012,
                 fc='magenta', ec='magenta', alpha=0.9, length_includes_head=True)

        pbar.set_postfix({
            "y": f"{y_i:.3f}", "dir": direction,
            "arc_L": f"{metric_arc:.4f}", "t(s)": f"{elapsed:.2f}"
        })

        # 正交性验证
        G = metric_mat(x_line, float(y_i)).cpu().numpy()
        t_vec = np.array([0.0, 1.0])
        n_vec = np.array([n1, n2])
        ortho_check = n_vec @ G @ t_vec

        all_results.append({
            "index": idx,
            "start": (x_line, float(y_i)),
            "direction": direction,
            "orthogonal_vector": (n1, n2),
            "initial_velocity": (u0, v0),
            "final_location": (float(xs_c[-1]), float(ys_c[-1])),
            "metric_arc_length": float(metric_arc),
            "euclidean_arc_length": float(euclid_arc),
            "orthogonality_check": float(ortho_check),
            "time_s": float(elapsed)
        })


def plot_geodesic_fan_orthogonal_to_vertical_line(
    x_line=0.5,
    y_min=0.1,
    y_max=0.9,
    n_points=10,
    speed=0.5,
    ode_T=1.0,
    direction='right',
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    max_step=0.01,
    rtol=1e-6,
    atol=1e-8,
    arrow_scale=0.03,
    save_path=None,
    log_path=None,
):
    """
    沿竖线 x=x_line 采样 n_points 个点，计算度规正交方向，
    以 speed 为初速度大小发射测地线。

    direction: 'right' / 'left' / 'both'
    """
    total_start = time.time()

    # 越靠近 y_min (接近0) 越密集：用幂次分布
    t = np.linspace(0, 1, n_points)
    y_points = y_min + (y_max - y_min) * t**2

    fig, ax = plt.subplots(figsize=(8, 7))

    # heatmap
    if show_heatmap:
        print("Computing heatmap ...")
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
        plt.colorbar(im, ax=ax, label='log det G')

    # 椭圆
    print("Drawing metric ellipses ...")
    draw_metric_ellipses(
        ax=ax, n_grid=ellipse_grid, scale=ellipse_scale,
        color='white' if show_heatmap else 'black', alpha=0.9, linewidth=1.0
    )

    # 竖线
    ax.plot([x_line, x_line], [y_min, y_max],
            linestyle='--', linewidth=1.5,
            color='cyan' if show_heatmap else 'blue',
            alpha=0.9, label=f'x = {x_line}')

    all_results = []

    if direction == 'both':
        dirs = ['right', 'left']
    else:
        dirs = [direction]

    for d in dirs:
        _shoot_one_direction(ax, x_line, y_points, speed, ode_T, d,
                             max_step, rtol, atol, arrow_scale, all_results)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Orthogonal geodesics from x={x_line}, y in [{y_min}, {y_max}]")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')

    total_elapsed = time.time() - total_start

    # ---- 汇总打印 ----
    print("\n" + "#" * 90)
    print("ALL ORTHOGONAL GEODESICS DONE")
    print(f"number of geodesics = {len(all_results)}")
    print(f"total elapsed time  = {total_elapsed:.3f} s")

    if len(all_results) > 0:
        avg_metric_arc = np.mean([d["metric_arc_length"] for d in all_results])
        avg_euclid_arc = np.mean([d["euclidean_arc_length"] for d in all_results])
        max_ortho_err = np.max([abs(d["orthogonality_check"]) for d in all_results])

        print(f"avg metric arc length  = {avg_metric_arc:.6f}")
        print(f"avg euclid arc length  = {avg_euclid_arc:.6f}")
        print(f"max |n^T G t| (ortho)  = {max_ortho_err:.2e}")

        print("\n" + "-" * 90)
        print(f"{'idx':>4s}  {'start':>16s}  {'dir':>6s}  "
              f"{'n^TGt':>10s}  {'metric_L':>10s}  {'euclid_L':>10s}  {'time(s)':>8s}")
        print("-" * 90)
        for d in all_results:
            sx, sy = d["start"]
            print(f"{d['index']:4d}  ({sx:.4f}, {sy:.4f})  {d['direction']:>6s}  "
                  f"{d['orthogonality_check']:10.2e}  "
                  f"{d['metric_arc_length']:10.4f}  "
                  f"{d['euclidean_arc_length']:10.4f}  "
                  f"{d['time_s']:8.2f}")
        print("-" * 90)

    print("#" * 90 + "\n")

    # ---- JSON 日志 ----
    if log_path is None and save_path is not None:
        log_path = save_path.rsplit('.', 1)[0] + '_log.json'

    if log_path is not None:
        log_data = {
            "x_line": x_line,
            "y_min": y_min,
            "y_max": y_max,
            "n_points": n_points,
            "speed": speed,
            "ode_T": ode_T,
            "direction": direction,
            "total_elapsed_s": total_elapsed,
            "geodesics": []
        }
        for d in all_results:
            log_data["geodesics"].append({
                "index": d["index"],
                "start": list(d["start"]),
                "direction": d["direction"],
                "orthogonal_vector": list(d["orthogonal_vector"]),
                "initial_velocity": list(d["initial_velocity"]),
                "final_location": list(d["final_location"]),
                "metric_arc_length": d["metric_arc_length"],
                "euclidean_arc_length": d["euclidean_arc_length"],
                "orthogonality_check": d["orthogonality_check"],
                "time_s": d["time_s"]
            })

        if len(all_results) > 0:
            log_data["summary"] = {
                "avg_metric_arc_length": float(avg_metric_arc),
                "avg_euclidean_arc_length": float(avg_euclid_arc),
                "max_orthogonality_error": float(max_ortho_err)
            }

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Log saved to: {log_path}")

    plt.show()
    return all_results


# =========================================================
# 13. 主程序
# =========================================================

# --- 示例：向右发射 ---
all_results = plot_geodesic_fan_orthogonal_to_vertical_line(
    x_line=0.2,
    y_min=0.05,
    y_max=0.6,
    n_points=12,
    speed=3.0,
    ode_T=3.0,
    direction='right',
    show_heatmap=True,
    heatmap_n=120,
    ellipse_grid=9,
    ellipse_scale=0.05,
    save_path='result/draw_geo_orthogonal/orthogonal_geodesics_right.png',
    log_path='result/draw_geo_orthogonal/orthogonal_geodesics_right_log.json',
)

# --- 示例：双向发射 ---
# all_results = plot_geodesic_fan_orthogonal_to_vertical_line(
#     x_line=0.5,
#     y_min=0.1,
#     y_max=0.9,
#     n_points=10,
#     speed=0.5,
#     ode_T=1.0,
#     direction='both',
#     show_heatmap=True,
#     save_path='orthogonal_geodesics_both.png',
# )
