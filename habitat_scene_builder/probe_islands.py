#!/usr/bin/env python3
"""probe_islands.py — 列出 navmesh 岛，报告每个探针点落在哪个岛。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

ROOT = HERE.parent / "demo_run" / "villa_2f"
from probe_nav import PTS  # noqa: E402


def main() -> None:
    hsim, _ = V2._import_habitat()
    for name in ("human", "spot"):
        path = ROOT / "navmeshes" / f"villa_2f_000__{name}.navmesh"
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(path))
        n = pf.num_islands
        areas = [(i, round(float(pf.island_area(i)), 2)) for i in range(n)]
        areas.sort(key=lambda t: -t[1])
        print(f"\n=== {name}: {n} 个岛, 面积前 12: {areas[:12]}")
        for label, pt in PTS.items():
            snapped = pf.snap_point(np.array(pt, dtype=np.float32))
            isl = int(pf.get_island(snapped))
            d = float(np.linalg.norm(snapped - np.array(pt)))
            print(f"  {label:<8} island={isl:<3} snap漂移={d:.2f}m  "
                  f"snapped=({snapped[0]:.2f},{snapped[1]:.2f},{snapped[2]:.2f})")


if __name__ == "__main__":
    main()
