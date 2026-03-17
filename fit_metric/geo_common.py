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
    """延迟初始化：设置 device、加载模型、构建 jacrev。"""
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
    assert _device is not None, "geo_common.init() has not been called"
    return _device


def get_dtype():
    return _dtype


# =========================================================
# 3. 度规张量 G(x,y) = (J J^T)^(-1)
# =========================================================
def metric_tensor_xy(xy, eps=1e-8):
    loc = xy.unsqueeze(0)
    v1 = _math_model(loc)
    v2 = _code_model(loc)
    J = torch.cat([v1, v2], dim=0).T
    A = J @ J.T
    A = A + eps * torch.eye(2, dtype=_dtype, device=_device)
    G = torch.linalg.inv(A)
    return G


def metric_tensor_xy_batch(xy_batch, eps=1e-8):
    v1 = _math_model(xy_batch)
    v2 = _code_model(xy_batch)
    J = torch.stack([v1, v2], dim=-1)
    A = J @ J.transpose(-1, -2)
    A = A + eps * torch.eye(2, dtype=_dtype, device=_device)
    G = torch.linalg.inv(A)
    return G


def metric_mat(x, y, eps=1e-8):
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "输入坐标必须在 [0,1]"
    xy = torch.tensor([x, y], dtype=_dtype, device=_device)
    G = metric_tensor_xy(xy, eps=eps)
    return G.detach()


# =========================================================
# 4. Christoffel symbols
# =========================================================
def christoffel_symbols(x, y):
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
