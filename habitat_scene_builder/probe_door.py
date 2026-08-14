#!/usr/bin/env python3
"""probe_door.py — 沿门洞法线密集采样 snap_point，定位门洞断连位置。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

NAV = HERE / "island_maps" / "stage_only" / "navmeshes"

# (名称, 起点, 终点) — 沿穿过门洞的直线
LINES = {
    "厨房门 x=-2.0 z=-3.45 (F1)": [(-2.6, 0.10, -3.45), (-1.4, 0.10, -3.45)],
    "书房门 x=1.2 z=2.95 (F1)":   [(0.5, 0.10, 2.95), (1.9, 0.10, 2.95)],
    "走廊门 x=1.2 z=-1.45 (F1)":  [(0.5, 0.10, -1.45), (1.9, 0.10, -1.45)],
    "主卧门 z=0.2 x=-2.45 (F2)":  [(-2.45, 2.95, -0.5), (-2.45, 2.95, 0.9)],
    "茶室门 x=1.2 z=2.95 (F2)":   [(0.5, 2.95, 2.95), (1.9, 2.95, 2.95)],
}


def main() -> None:
    hsim, _ = V2._import_habitat()
    for name in ("human", "spot"):
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(NAV / f"villa_2f_000__{name}.navmesh"))
        print(f"\n=== {name} ===")
        for label, (a, b) in LINES.items():
            a, b = np.array(a), np.array(b)
            print(f"  {label}")
            for t in np.linspace(0, 1, 15):
                pt = a + (b - a) * t
                sn = pf.snap_point(pt.astype(np.float32))
                drift = float(np.linalg.norm(sn - pt))
                isl = int(pf.get_island(sn)) if not np.isnan(sn[0]) else -1
                print(f"    t={t:.2f} ({pt[0]:5.2f},{pt[2]:5.2f}) -> island={isl:<3} "
                      f"drift={drift:.2f} y={sn[1]:.2f}")


if __name__ == "__main__":
    main()
