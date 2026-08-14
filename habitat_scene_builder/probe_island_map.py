#!/usr/bin/env python3
"""probe_island_map.py — 网格采样 snap_point + get_island，画两层岛分布图。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

ROOT = HERE.parent / "demo_run" / "villa_2f"
if len(sys.argv) > 1:                      # 可指定其它 navmesh 目录
    ROOT = Path(sys.argv[1]).resolve()
from probe_nav import PTS  # noqa: E402


def main() -> None:
    hsim, _ = V2._import_habitat()
    out = HERE / "island_maps"
    out.mkdir(exist_ok=True)
    tag = ROOT.parent.name if ROOT.name == "navmeshes" else ROOT.name
    for name in ("human", "spot"):
        path = ROOT / "navmeshes" / f"villa_2f_000__{name}.navmesh"
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(path))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, (y, title) in zip(axes, [(0.10, "F1 y=0.10"), (2.95, "F2 y=2.95")]):
            xs = np.arange(-5.1, 5.15, 0.1)
            zs = np.arange(-5.1, 5.15, 0.1)
            grid = np.full((len(zs), len(xs)), -1, dtype=int)
            for iz, z in enumerate(zs):
                for ix, x in enumerate(xs):
                    sn = pf.snap_point(np.array([x, y, z], dtype=np.float32))
                    if abs(float(sn[1]) - y) < 0.45:   # 只收本层
                        grid[iz, ix] = int(pf.get_island(sn))
                    else:
                        grid[iz, ix] = -2               # snap 到别的层
            ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]],
                      cmap="tab20", vmin=-2, vmax=19, interpolation="nearest")
            ax.set_title(f"{name} {title}  (-1=无navmesh -2=snap到别层)")
            for label, (px, py, pz) in PTS.items():
                if abs(py - y) < 0.5:
                    ax.plot(px, pz, "k+", markersize=8)
                    ax.annotate(label, (px, pz), fontsize=7)
        fig.tight_layout()
        fp = out / f"islands_{tag}_{name}.png"
        fig.savefig(fp, dpi=110)
        print(fp)


if __name__ == "__main__":
    main()
