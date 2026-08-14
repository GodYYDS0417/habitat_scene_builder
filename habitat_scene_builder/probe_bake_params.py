#!/usr/bin/env python3
"""probe_bake_params.py — 扫 navmesh 参数，定位 F1 门洞断连的根因。"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"

# F1 厨房门两侧点 + F2 主卧门两侧点
PAIRS = [
    ("F1厨房门", (-3.0, 0.10, -3.45), (-1.0, 0.10, -3.45)),
    ("F1书房门", (0.5, 0.10, 2.95), (1.9, 0.10, 2.95)),
    ("F2主卧门", (-2.45, 2.95, -0.5), (-2.45, 2.95, 0.9)),
]

VARIANTS = [
    ("基准 r0.30 h1.70 cs0.05 ch0.20", dict()),
    ("cell_height=0.10", dict(cell_height=0.10)),
    ("agent_height=1.50", dict(height=1.50)),
    ("radius=0.25", dict(radius=0.25)),
    ("radius=0.30 h1.50 ch0.10", dict(height=1.50, cell_height=0.10)),
]


def main() -> None:
    hsim, _ = V2._import_habitat()
    empty = RUN / "scenes" / "_empty.scene_instance.json"
    if not empty.exists():
        empty.write_text(json.dumps({
            "stage_instance": {"template_name": "stages/villa_2f"},
            "default_lighting": "", "object_instances": []}))
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(empty)
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))

    base = dict(V2.EMBODIMENTS["human"])
    for label, over in VARIANTS:
        p = {**base, **over}
        ns = hsim.NavMeshSettings()
        ns.set_defaults()
        ns.cell_size = over.get("cell_size", V2.NAVMESH_CELL_SIZE)
        ns.cell_height = over.get("cell_height", V2.NAVMESH_CELL_HEIGHT)
        ns.agent_radius = p["radius"]
        ns.agent_height = p["height"]
        ns.agent_max_climb = p["max_climb"]
        ns.agent_max_slope = p["max_slope"]
        if hasattr(ns, "include_static_objects"):
            ns.include_static_objects = True
        ok = V2.recompute_navmesh_compat(sim, hsim, ns, True)
        pf = sim.pathfinder
        res = []
        for plabel, a, b in PAIRS:
            sp = hsim.nav.ShortestPath()
            sp.requested_start = a
            sp.requested_end = b
            found = pf.find_path(sp)
            res.append(f"{plabel}={'OK' if found else 'X'}")
        print(f"{label:<34} ok={ok} area={pf.navigable_area:6.1f}  " + "  ".join(res))
    sim.close()


if __name__ == "__main__":
    main()
