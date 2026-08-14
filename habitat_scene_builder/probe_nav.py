#!/usr/bin/env python3
"""probe_nav.py — 逐段 find_path 定位别墅 navmesh 断点。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

ROOT = HERE.parent / "demo_run" / "villa_2f"

PTS = {
    "客厅":     (-3.0, 0.10, 1.70),
    "客厅门口": (1.2, 0.10, 4.35),
    "F1书房":   (4.30, 0.10, 4.30),
    "餐厅":     (-0.5, 0.10, -1.1),
    "厨房":     (-3.5, 0.10, -3.5),
    "F1走廊":   (2.0, 0.10, -1.5),
    "楼梯底":   (3.4, 0.10, -2.3),
    "第5步":    (4.35, 0.80, -1.30),
    "第10步":   (4.35, 1.50, 0.10),
    "转角平台": (3.8, 1.55, 0.65),
    "后段第3步": (2.30, 2.00, 0.65),
    "顶步":     (0.35, 2.95, 0.65),
    "F2走廊":   (-0.30, 2.95, 1.50),
    "F2书房":   (-0.30, 2.95, -3.60),
    "主卧":     (-1.50, 2.95, 2.00),
    "次卧":     (-3.50, 2.95, -1.50),
    "茶室":     (2.50, 2.95, 3.50),
}

# 主链：客厅 -> 楼梯 -> F2 走廊 -> F2 书房
CHAIN = ["客厅", "客厅门口", "F1书房", "F1走廊", "楼梯底", "第5步", "第10步",
         "转角平台", "后段第3步", "顶步", "F2走廊", "F2书房"]
# 支路
EXTRA = [("客厅", "餐厅"), ("餐厅", "厨房"),
         ("F2走廊", "主卧"), ("F2走廊", "次卧"), ("顶步", "茶室")]


def run(pf, hsim, pairs):
    for l1, l2 in pairs:
        sp = hsim.nav.ShortestPath()
        sp.requested_start = PTS[l1]
        sp.requested_end = PTS[l2]
        found = pf.find_path(sp)
        pts = np.asarray(sp.points)
        geo = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
        print(f"  {'OK ' if found else 'FAIL'} {l1} -> {l2}: {len(pts)} 点, {geo:.2f} m")


def main() -> None:
    hsim, _ = V2._import_habitat()
    for name in ("human", "spot"):
        path = ROOT / "navmeshes" / f"villa_2f_000__{name}.navmesh"
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(path))
        print(f"\n=== {name} ===")
        run(pf, hsim, list(zip(CHAIN, CHAIN[1:])))
        run(pf, hsim, EXTRA)


if __name__ == "__main__":
    main()
