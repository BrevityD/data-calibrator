"""
事后可视化 sft_via_geoguide 的训练轨迹和每个 epoch 的测地线。

读取 geoguide_log.json，在 log(det G) 热力图上绘制：
  - 模型归一化坐标的 epoch 间轨迹
  - 每个 epoch 起点处的测地线路径（从 JSON 中读取）
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import geo_common


def load_log(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw(
    log_path: str,
    geo_device: str = "cuda:0",
    show_heatmap: bool = True,
    heatmap_n: int = 120,
    ellipse_grid: int = 9,
    ellipse_scale: float = 0.05,
    save_path: str | None = None,
):
    log_data = load_log(log_path)
    cfg = log_data["config"]
    epochs = log_data["epochs"]

    if not epochs:
        print("No epoch data found in log.")
        return

    # --- 初始化 geo_common（heatmap / ellipse 需要模型） ---
    geo_common.init(
        device_str=geo_device,
        seed=cfg.get("seed", 42),
        math_model_path=cfg["math_model_path"],
        code_model_path=cfg["code_model_path"],
    )

    # --- 提取轨迹坐标 ---
    coords = np.array([e["normalized_xy"] for e in epochs])  # (n_epochs, 2)

    # --- 从 JSON 读取测地线路径 ---
    geo_paths = []
    for e in epochs:
        if "geodesic_path" in e:
            geo_paths.append(np.array(e["geodesic_path"]))
        else:
            geo_paths.append(None)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 7))

    if show_heatmap:
        print("Computing heatmap ...")
        xs, ys, Z = geo_common.compute_logdet_grid(n=heatmap_n)
        im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        plt.colorbar(im, ax=ax, label="log det G")

    geo_common.draw_metric_ellipses(
        ax=ax, n_grid=ellipse_grid, scale=ellipse_scale,
        color="white" if show_heatmap else "black", alpha=0.85, linewidth=0.8,
    )

    # 目标直线
    target_domain = cfg.get("geo_target_domain", "math")
    target_val = cfg.get("geo_target_value", 0.2)
    line_color = "cyan" if show_heatmap else "blue"
    if target_domain == "math":
        ax.axvline(x=target_val, linestyle="--", linewidth=1.5,
                   color=line_color, alpha=0.8, label=f"x = {target_val}")
    else:
        ax.axhline(y=target_val, linestyle="--", linewidth=1.5,
                   color=line_color, alpha=0.8, label=f"y = {target_val}")

    # 测地线路径（每个 epoch 一条，渐变色）
    valid_paths = [(i, p) for i, p in enumerate(geo_paths) if p is not None]
    n_geo = len(valid_paths)
    geo_cmap = plt.cm.Oranges
    for idx, (i, path) in enumerate(valid_paths):
        c = geo_cmap(0.35 + 0.6 * idx / max(n_geo - 1, 1))
        ax.plot(path[:, 0], path[:, 1], color=c, linewidth=1.2, alpha=0.7,
                label=f"geodesic ep{epochs[i]['epoch']}" if idx < 5 else None)

    if not valid_paths:
        print("[WARN] No geodesic_path found in log. "
              "Re-run sft_via_geoguide to generate paths, or use an older log.")

    # 模型轨迹（带箭头的折线）
    for i in range(len(coords) - 1):
        ax.annotate(
            "", xy=coords[i + 1], xytext=coords[i],
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0, mutation_scale=12),
        )

    # epoch 标记点
    for i, (x, y) in enumerate(coords):
        ep = epochs[i]["epoch"]
        color = "lime" if i == 0 else ("yellow" if i == len(coords) - 1 else "white")
        ax.scatter([x], [y], color=color, s=70, edgecolors="black", zorder=6)
        ax.annotate(
            f"ep{ep}", (x, y), textcoords="offset points", xytext=(6, 6),
            fontsize=7, color="white" if show_heatmap else "black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5) if show_heatmap else None,
        )

    # ratio 标注
    for i, e in enumerate(epochs):
        ratio_text = f"m:{e['math_ratio']:.2f}"
        ax.annotate(
            ratio_text, (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(6, -10),
            fontsize=6, color="white" if show_heatmap else "gray", alpha=0.9,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("normalized math loss")
    ax.set_ylabel("normalized code loss")
    ax.set_title(f"GeoGuide trajectory ({len(epochs)} epochs)")
    ax.legend(fontsize=7, loc="upper right")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    plt.show()


def main():
    default_log = str(_SCRIPT_DIR / "result" / "sft_via_geoguide" / "geoguide_log.json")
    default_save = str(_SCRIPT_DIR / "result" / "draw_geoguide_trajectory" / "trajectory.png")

    p = argparse.ArgumentParser(description="可视化 GeoGuide 训练轨迹和测地线")
    p.add_argument("--log_path", type=str, default=default_log,
                   help="geoguide_log.json 路径")
    p.add_argument("--geo_device", type=str, default="cuda:0")
    p.add_argument("--save_path", type=str, default=default_save)
    p.add_argument("--no_heatmap", action="store_true")
    p.add_argument("--heatmap_n", type=int, default=120)
    p.add_argument("--ellipse_grid", type=int, default=9)
    p.add_argument("--ellipse_scale", type=float, default=0.05)
    args = p.parse_args()

    draw(
        log_path=args.log_path,
        geo_device=args.geo_device,
        show_heatmap=not args.no_heatmap,
        heatmap_n=args.heatmap_n,
        ellipse_grid=args.ellipse_grid,
        ellipse_scale=args.ellipse_scale,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
