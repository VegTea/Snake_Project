#!/usr/bin/env python3
"""计算加权 planar MAE：内圈9个均值×0.6 + 外圈16个均值×0.4"""

import argparse
import csv
import numpy as np

# 5×5 网格定义
VX_VALS = [-0.2, -0.1, 0.0, 0.1, 0.2]
VY_VALS = [-0.1, -0.05, 0.0, 0.05, 0.1]
# 内圈 (3×3): vx ∈ [-0.1, 0.0, 0.1], vy ∈ [-0.05, 0.0, 0.05]
INNER_VX = {-0.1, 0.0, 0.1}
INNER_VY = {-0.05, 0.0, 0.05}


def main():
    parser = argparse.ArgumentParser(description="计算加权 planar MAE")
    parser.add_argument("csv_path", nargs="?", default="eval_output/data/eval_mae.csv",
                        help="eval_mae.csv 路径")
    args = parser.parse_args()

    data = {}  # (vx, vy) -> planar_mae
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vx = round(float(row["vx"]), 2)
            vy = round(float(row["vy"]), 2)
            data[(vx, vy)] = float(row["planar_mae"])

    def is_inner(vx, vy):
        return round(vx, 2) in INNER_VX and round(vy, 2) in INNER_VY

    inner, outer = [], []
    for vx in VX_VALS:
        for vy in VY_VALS:
            vx, vy = round(vx, 2), round(vy, 2)
            mae = data.get((vx, vy), np.nan)
            if is_inner(vx, vy):
                inner.append(mae)
            else:
                outer.append(mae)

    avg_inner = np.mean(inner)
    avg_outer = np.mean(outer)
    weighted = avg_inner * 0.6 + avg_outer * 0.4

    print(f"内圈 9 个: {[f'{v:.4f}' for v in inner]}")
    print(f"内圈均值:   {avg_inner:.4f}")
    print(f"外圈 16 个: {[f'{v:.4f}' for v in outer]}")
    print(f"外圈均值:   {avg_outer:.4f}")
    print(f"───────────────")
    print(f"加权 MAE = {avg_inner:.4f} × 0.6 + {avg_outer:.4f} × 0.4 = {weighted:.4f}")


if __name__ == "__main__":
    main()
