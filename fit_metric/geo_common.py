"""
geo_common.py — 测地线计算共享基础设施。

所有模型加载和 device 设置通过 init() 延迟初始化，
import 本模块不会触发任何副作用。
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn

from scipy.integrate import solve_ivp
from matplotlib.patches import Ellipse


# =========================================================
# 0. 随机种子
# =========================================================
def seed_everything(seed: int = 42) -> int:
    """固定所有随机种子（Python、NumPy、PyTorch），确保实验可复现。"""
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
    """带 SiLU 激活的残差块，保证输出光滑性。"""

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
    """2D → 2D 向量场网络，用于拟合数据混合比例到指标的映射。

    结构: Linear(2→48) + SiLU → 2×SmoothResidualBlock → Linear(48→2)
    """

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
# 2. 模块级状态（延迟初始化）
# =========================================================
_device = None
_dtype = torch.float64
_math_model = None
_code_model = None
_jac_metric_tensor_xy = None


def init(
    device_str="cuda:4",
    seed=42,
    math_model_path='/public/home/jza/data_calibrate/data_mixture/metric_fit/v1.pth',
    code_model_path='/public/home/jza/data_calibrate/data_mixture/metric_fit/v2.pth',
):
    """延迟初始化：设置 device、加载两个向量场模型、构建度规张量的 jacrev。

    参数:
        device_str: CUDA 设备字符串，如 'cuda:4'
        seed: 随机种子
        math_model_path: 数学指标向量场模型权重路径
        code_model_path: 代码指标向量场模型权重路径
    """
    global _device, _dtype, _math_model, _code_model, _jac_metric_tensor_xy

    seed_everything(seed)

    _device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    _dtype = torch.float64

    _math_model = VectorField().to(_device).double()
    _code_model = VectorField().to(_device).double()

    _math_model.load_state_dict(torch.load(math_model_path, map_location=_device))
    _code_model.load_state_dict(torch.load(code_model_path, map_location=_device))

    _math_model.eval()
    _code_model.eval()

    _jac_metric_tensor_xy = torch.func.jacrev(metric_tensor_xy)


def get_device():
    """返回当前 torch.device，未调用 init() 时抛出 AssertionError。"""
    assert _device is not None, "geo_common.init() has not been called"
    return _device


def get_dtype():
    """返回当前计算精度（默认 torch.float64）。"""
    return _dtype


# =========================================================
# 3. 度规张量 G(x,y) = (J J^T)^(-1)
# =========================================================
def metric_tensor_xy(xy, eps=1e-8):
    """计算单点度规张量 G(x,y) = (J J^T)^{-1}。

    参数:
        xy: shape (2,) 的 tensor，坐标 (x, y) ∈ [0,1]²
        eps: 正则化常数，防止 J J^T 奇异

    返回:
        G: shape (2,2) 的度规张量
    """
    loc = xy.unsqueeze(0)
    v1 = _math_model(loc)
    v2 = _code_model(loc)
    J = torch.cat([v1, v2], dim=0).T
    A = J @ J.T
    A = A + eps * torch.eye(2, dtype=_dtype, device=_device)
    G = torch.linalg.inv(A)
    return G


def metric_tensor_xy_batch(xy_batch, eps=1e-8):
    """批量计算度规张量。

    参数:
        xy_batch: shape (B, 2) 的 tensor
        eps: 正则化常数

    返回:
        G: shape (B, 2, 2) 的度规张量
    """
    v1 = _math_model(xy_batch)
    v2 = _code_model(xy_batch)
    J = torch.stack([v1, v2], dim=-1)
    A = J @ J.transpose(-1, -2)
    A = A + eps * torch.eye(2, dtype=_dtype, device=_device)
    G = torch.linalg.inv(A)
    return G


def metric_mat(x, y, eps=1e-8):
    """便捷接口：给定标量坐标 (x, y) ∈ [0,1]，返回 detached 度规张量。"""
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "输入坐标必须在 [0,1]"
    xy = torch.tensor([x, y], dtype=_dtype, device=_device)
    G = metric_tensor_xy(xy, eps=eps)
    return G.detach()


# =========================================================
# 4. Christoffel symbols
# =========================================================
def christoffel_symbols(x, y):
    """计算 Christoffel 符号 Γ^k_{ij}(x, y)。

    通过 jacrev 自动微分度规张量得到 ∂g_{jl}/∂x^a，
    再由标准公式 Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij}) 计算。

    返回:
        Gamma: shape (2, 2, 2) 的 tensor，Gamma[k, i, j] = Γ^k_{ij}
    """
    xy = torch.tensor([x, y], dtype=_dtype, device=_device, requires_grad=True)
    G = metric_tensor_xy(xy)
    Ginv = torch.linalg.inv(G)
    J = _jac_metric_tensor_xy(xy)
    dg = J.permute(2, 0, 1).contiguous()

    term = torch.zeros((2, 2, 2), dtype=_dtype, device=_device)
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
    """测地线 ODE 右端项，供 scipy.integrate.solve_ivp 使用。

    状态 Y = [x, y, u, v]，其中 (x,y) 为位置，(u,v) 为速度。
    当 (x,y) 越出 [0,1]² 时返回零向量（配合边界事件停止积分）。
    """
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
    """边界停止事件：当轨迹触及 [0,1]² 边界时终止积分。"""
    x, y, u, v = Y
    return min(x, y, 1.0 - x, 1.0 - y)

hit_boundary.terminal = True
hit_boundary.direction = -1


# =========================================================
# 7. 初值测地线
# =========================================================
def solve_geodesic_ivp(x0, y0, u0, v0, T=1.0, max_step=0.01, rtol=1e-6, atol=1e-8):
    """求解测地线初值问题。

    从 (x0, y0) 出发，初速度 (u0, v0)，用 RK45 积分到 t=T。
    遇到 [0,1]² 边界时自动停止。

    返回:
        sol: scipy OdeResult，sol.y 形状 (4, N_pts)
    """
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
    """计算点 (x,y) 处度规椭圆的绘图参数。

    度规椭圆由 v^T G v = 1 定义，半轴长度为 1/sqrt(λ_i)。
    scale 控制椭圆在图上的缩放大小。

    返回:
        width, height: 椭圆宽高（已乘 2×scale）
        angle: 长轴旋转角度（度）
        eigvals: G 的特征值（升序）
        eigvecs: 对应特征向量
    """
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
    """在 matplotlib Axes 上绘制 n_grid × n_grid 的度规椭圆网格。"""
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
    """在 [0,1]² 上计算 n×n 网格的 log(det G) 值，用于热力图可视化。

    返回:
        xs, ys: 长度为 n 的坐标数组
        Z: shape (n, n) 的 log(det G) 值矩阵（非正定处为 NaN）
    """
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
    """计算 solve_ivp 解的度规弧长 L = Σ sqrt(dq^T G(mid) dq)。

    参数:
        sol: scipy OdeResult，sol.y shape (4, N_pts)

    返回:
        arc_length: float，度规意义下的弧长
    """
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
    """计算 solve_ivp 解的欧氏弧长，用于与度规弧长对比。"""
    xs = sol.y[0]
    ys = sol.y[1]
    dx = np.diff(xs)
    dy = np.diff(ys)
    return np.sum(np.sqrt(dx**2 + dy**2))
