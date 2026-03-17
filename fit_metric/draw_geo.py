import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.optimize import root

from geo_common import (
    init,
    get_device,
    get_dtype,
    metric_tensor_xy,
    metric_tensor_xy_batch,
    metric_mat,
    christoffel_symbols,
    geodesic_rhs,
    hit_boundary,
    solve_geodesic_ivp,
    metric_ellipse_data,
    draw_metric_ellipses,
    compute_logdet_grid,
    compute_geodesic_arc_length,
    compute_euclidean_arc_length,
)


# =========================================================
# 1. shooting method
# =========================================================
def endpoint_error(vel, start, target, T=1.0, penalty=10.0):
    x0, y0 = start
    x1, y1 = target
    u0, v0 = vel

    sol = solve_geodesic_ivp(x0, y0, u0, v0, T=T)

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
# 2. 单条测地线 + 度规椭圆 + 可选 heatmap
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
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
        plt.colorbar(im, ax=ax, label='log det G')

    draw_metric_ellipses(
        ax=ax, n_grid=ellipse_grid, scale=ellipse_scale,
        color='white' if show_heatmap else 'black', alpha=0.9, linewidth=1.0
    )

    ax.plot(xs_curve, ys_curve, color='red', linewidth=2.5, label='geodesic')
    ax.scatter([x0], [y0], color='cyan', s=70, label='start')
    ax.scatter([x1], [y1], color='yellow', s=70, label='target')
    ax.scatter([xs_curve[-1]], [ys_curve[-1]], color='lime', s=45, label='reached')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Geodesic with local metric ellipses")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()

    return result, sol


# =========================================================
# 4. 多条测地线：连到竖线 x = x_target 上
# =========================================================
def plot_geodesic_fan_to_vertical_line(
    start, x_target=0.2, y_min=0.2, y_max=0.6,
    T_num=12, ode_T=1.0, method='hybr',
    show_heatmap=True, heatmap_n=120, ellipse_grid=9, ellipse_scale=0.05,
    max_step=0.01, rtol=1e-6, atol=1e-8,
    strict_open_interval=True,
    save_path=None, log_path=None, resume_log_path='auto'
):
    total_start_time = time.time()
    x0, y0 = start
    y_targets = np.linspace(y_min, y_max, T_num)

    # ---- 加载 resume 缓存 ----
    if resume_log_path == 'auto':
        resume_log_path = log_path

    resume_cache = {}
    if resume_log_path is not None and os.path.exists(resume_log_path):
        with open(resume_log_path, 'r', encoding='utf-8') as f:
            prev_log = json.load(f)
        for g in prev_log.get("geodesics", []):
            if g.get("converged", g.get("endpoint_error_norm", 1e9) < 1e-2) and "initial_velocity" in g:
                key = (round(g["target"][0], 6), round(g["target"][1], 6))
                resume_cache[key] = g
        print(f"Resume: loaded {len(resume_cache)} cached geodesics from {resume_log_path}")

    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        print("Computing heatmap ...")
        heatmap_start_time = time.time()
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
        plt.colorbar(im, ax=ax, label='log det G')
        print(f"Heatmap done in {time.time() - heatmap_start_time:.3f} s")

    print("Drawing metric ellipses ...")
    ellipse_start_time = time.time()
    draw_metric_ellipses(
        ax=ax, n_grid=ellipse_grid, scale=ellipse_scale,
        color='white' if show_heatmap else 'black', alpha=0.9, linewidth=1.0
    )
    print(f"Metric ellipses done in {time.time() - ellipse_start_time:.3f} s")

    ax.plot([x_target, x_target], [y_min, y_max], linestyle='--', linewidth=1.5,
            color='cyan' if show_heatmap else 'blue', alpha=0.9, label=f'x = {x_target}')

    all_results = []
    prev_vel = None

    pbar = tqdm(enumerate(y_targets, start=1), total=len(y_targets),
                desc="Solving geodesics", ncols=120)

    for idx, y_t in pbar:
        geodesic_start_time = time.time()
        target = (x_target, float(y_t))
        cache_key = (round(target[0], 6), round(target[1], 6))

        cached = resume_cache.get(cache_key)
        if cached is not None:
            cached_vel = np.array(cached["initial_velocity"], dtype=np.float64)
            shooting_elapsed = 0.0
            sol = solve_geodesic_ivp(x0, y0, cached_vel[0], cached_vel[1],
                                     T=ode_T, max_step=max_step, rtol=rtol, atol=atol)
            ivp_elapsed = time.time() - geodesic_start_time
            xs_curve = sol.y[0]; ys_curve = sol.y[1]
            end_err = np.array([xs_curve[-1] - target[0], ys_curve[-1] - target[1]])
            err_norm = np.linalg.norm(end_err)
            metric_arc_len = compute_geodesic_arc_length(sol)
            euclid_arc_len = compute_euclidean_arc_length(sol)
            arc_elapsed = 0.0
            geodesic_elapsed = time.time() - geodesic_start_time
            final_location = (float(xs_curve[-1]), float(ys_curve[-1]))

            class _CachedResult:
                def __init__(self, x, success):
                    self.x = x; self.success = success
                    self.message = "resumed from cache"
            result = _CachedResult(cached_vel, True)
            print(f"[{idx}/{len(y_targets)}] target = ({target[0]:.4f}, {target[1]:.4f})  ** RESUMED FROM CACHE **")
        else:
            shooting_start_time = time.time()
            result = robust_shooting_geodesic(start=start, target=target, T=ode_T,
                                              method=method, prev_vel=prev_vel)
            shooting_elapsed = time.time() - shooting_start_time
            u0, v0 = result.x
            ivp_start_time = time.time()
            sol = solve_geodesic_ivp(x0, y0, u0, v0, T=ode_T,
                                     max_step=max_step, rtol=rtol, atol=atol)
            ivp_elapsed = time.time() - ivp_start_time
            xs_curve = sol.y[0]; ys_curve = sol.y[1]
            end_err = np.array([xs_curve[-1] - target[0], ys_curve[-1] - target[1]])
            err_norm = np.linalg.norm(end_err)
            arc_start_time = time.time()
            metric_arc_len = compute_geodesic_arc_length(sol)
            euclid_arc_len = compute_euclidean_arc_length(sol)
            arc_elapsed = time.time() - arc_start_time
            geodesic_elapsed = time.time() - geodesic_start_time
            final_location = (float(xs_curve[-1]), float(ys_curve[-1]))

            pbar.set_postfix({"target_y": f"{y_t:.3f}", "err": f"{err_norm:.2e}",
                              "arc_L": f"{metric_arc_len:.4f}"})

        if err_norm < 1e-2:
            prev_vel = result.x.copy()

        ax.plot(xs_curve, ys_curve, linewidth=2.0, alpha=0.95)
        ax.scatter([target[0]], [target[1]], s=35, color='yellow', edgecolors='black', zorder=5)
        ax.scatter([xs_curve[-1]], [ys_curve[-1]], s=20, color='lime', zorder=5)

        all_results.append({
            "index": idx, "target": target, "result": result, "sol": sol,
            "end_err": end_err, "err_norm": err_norm, "final_location": final_location,
            "metric_arc_length": metric_arc_len, "euclidean_arc_length": euclid_arc_len,
            "shooting_time": shooting_elapsed, "ivp_time": ivp_elapsed,
            "arc_len_time": arc_elapsed, "total_line_time": geodesic_elapsed
        })

    ax.scatter([x0], [y0], color='red', s=80, label='start', zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Fan of geodesics to x={x_target}, y in ({y_min}, {y_max})")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')

    total_elapsed = time.time() - total_start_time
    print(f"\nALL GEODESICS DONE  total={total_elapsed:.3f}s")

    # ---- 保存 JSON 日志 ----
    if log_path is None and save_path is not None:
        log_path = save_path.rsplit('.', 1)[0] + '_log.json'

    if log_path is not None:
        log_data = {
            "start": list(start), "x_target": x_target,
            "y_min": y_min, "y_max": y_max, "T_num": T_num,
            "ode_T": ode_T, "total_elapsed_s": total_elapsed,
            "geodesics": []
        }
        for d in all_results:
            log_data["geodesics"].append({
                "index": d["index"], "target": list(d["target"]),
                "final_location": list(d["final_location"]),
                "initial_velocity": d["result"].x.tolist(),
                "endpoint_error_norm": float(d["err_norm"]),
                "metric_arc_length": float(d["metric_arc_length"]),
                "euclidean_arc_length": float(d["euclidean_arc_length"]),
                "shooting_time_s": float(d["shooting_time"]),
                "total_line_time_s": float(d["total_line_time"]),
                "shooting_success": bool(d["result"].success),
                "converged": bool(d["err_norm"] < 1e-2)
            })
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Log saved to: {log_path}")

    return all_results


# =========================================================
# 5. 主程序
# =========================================================
if __name__ == "__main__":
    init(device_str="cuda:4")

    start = (0.9, 0.9)
    target = (0.2, 0.2)

    result, sol = plot_geodesic_with_metric_ellipses(
        start=start, target=target, ode_T=1.0,
        show_heatmap=True, heatmap_n=120,
        ellipse_grid=9, ellipse_scale=0.05,
        save_path='result/draw_geo/geodesic_09_to_02.png'
    )
# =========================================================
def draw_tangent_and_easy_direction(ax, sol, n_samples=12, tangent_scale=0.04, easy_dir_scale=0.05):
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
                x, y, tangent_scale * tang[0], tangent_scale * tang[1],
                head_width=0.008, head_length=0.012,
                fc='blue', ec='blue', alpha=0.9, length_includes_head=True
            )

        try:
            _, _, _, eigvals, eigvecs = metric_ellipse_data(x, y, scale=1.0)
            easy_dir = eigvecs[:, 0]
            easy_dir = easy_dir / (np.linalg.norm(easy_dir) + 1e-12)
            ax.arrow(
                x, y, easy_dir_scale * easy_dir[0], easy_dir_scale * easy_dir[1],
                head_width=0.008, head_length=0.012,
                fc='green', ec='green', alpha=0.9, length_includes_head=True
            )
        except Exception:
            pass


def plot_geodesic_with_ellipses_and_directions(
    start, target, ode_T=1.0, init_vel=None, method='hybr',
    show_heatmap=True, heatmap_n=120, ellipse_grid=9, ellipse_scale=0.05,
    tangent_samples=12, save_path=None
):
    result, sol = solve_two_point_geodesic(start, target, T=ode_T, init_vel=init_vel, method=method)
    xs_curve = sol.y[0]
    ys_curve = sol.y[1]
    x0, y0 = start
    x1, y1 = target

    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        xs, ys, Z = compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
        plt.colorbar(im, ax=ax, label='log det G')

    draw_metric_ellipses(
        ax=ax, n_grid=ellipse_grid, scale=ellipse_scale,
        color='white' if show_heatmap else 'black', alpha=0.9, linewidth=1.0
    )

    ax.plot(xs_curve, ys_curve, color='red', linewidth=2.5, label='geodesic')
    ax.scatter([x0], [y0], color='cyan', s=70, label='start')
    ax.scatter([x1], [y1], color='yellow', s=70, label='target')

    draw_tangent_and_easy_direction(ax=ax, sol=sol, n_samples=tangent_samples,
                                    tangent_scale=0.04, easy_dir_scale=0.05)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Geodesic + metric ellipses + tangent/easy directions")
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.show()

    return result, sol
