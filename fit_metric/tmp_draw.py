"""
临时脚本：对已有的 geoguide_log.json 重新计算测地线路径，
将 geodesic_path 补写回 JSON，然后调用 draw_geoguide_trajectory.py 画图。
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import geo_common
from variational_geodesic import multi_start_variational_geodesic_to_line


def recompute_and_patch(log_path: str, geo_device: str = "cuda:0"):
    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    cfg = log_data["config"]
    epochs = log_data["epochs"]

    geo_common.init(
        device_str=geo_device,
        seed=cfg.get("seed", 42),
        math_model_path=cfg["math_model_path"],
        code_model_path=cfg["code_model_path"],
    )

    target_domain = cfg.get("geo_target_domain", "math")
    target_axis = 0 if target_domain == "math" else 1

    for e in epochs:
        if "geodesic_path" in e:
            print(f"  epoch {e['epoch']}: already has geodesic_path, skipping")
            continue

        xy = tuple(e["normalized_xy"])
        print(f"  epoch {e['epoch']}: recomputing geodesic at ({xy[0]:.4f}, {xy[1]:.4f}) ...")

        path, energy, arc_length, info, candidates = multi_start_variational_geodesic_to_line(
            start=xy,
            x_target=cfg.get("geo_target_value", 0.2),
            K=cfg.get("geo_K", 12),
            N=cfg.get("geo_N", 50),
            refine_top_k=cfg.get("geo_refine_top_k", 3),
            refine_N=cfg.get("geo_refine_N", 100),
            verbose=False,
            target_axis=target_axis,
        )
        e["geodesic_path"] = path.tolist()

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Patched {log_path}")


def main():
    default_log = str(_SCRIPT_DIR / "result" / "sft_via_geoguide" / "geoguide_log.json")

    p = argparse.ArgumentParser(description="补算 geodesic_path 并写回 JSON，然后画图")
    p.add_argument("--log_path", type=str, default=default_log)
    p.add_argument("--geo_device", type=str, default="cuda:0")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--no_heatmap", action="store_true")
    args = p.parse_args()

    # 1. 补算路径写回 JSON
    recompute_and_patch(args.log_path, args.geo_device)

    # 2. 调用 draw_geoguide_trajectory 画图
    from draw_geoguide_trajectory import draw
    save_path = args.save_path or str(
        _SCRIPT_DIR / "result" / "draw_geoguide_trajectory" / "trajectory.png"
    )
    draw(
        log_path=args.log_path,
        geo_device=args.geo_device,
        show_heatmap=not args.no_heatmap,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
