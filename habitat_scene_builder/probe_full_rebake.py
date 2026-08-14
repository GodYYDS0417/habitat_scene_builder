#!/usr/bin/env python3
"""probe_full_rebake.py — 全量 71 物体场景，对比 DYNAMIC 是否参与烘焙的差异。

A 组: DYNAMIC 保持 DYNAMIC（不参与烘焙）  B 组: 全部转 STATIC（builder 旧行为）
两组各烘焙 human，检查 5 条关键链路 + 客厅断带栅格，定位分岛元凶。"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from probe_nav import PTS, CHAIN, EXTRA  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"


def add_ramps(sim, hsim, mn):
    ramps = RUN / "stages" / "villa_2f_navramps.glb"
    tmp = Path(tempfile.mkdtemp(prefix="navramps_fr_"))
    (tmp / "_navramps.object_config.json").write_text(json.dumps({
        "render_asset": str(ramps), "collision_asset": str(ramps),
        "units_to_meters": 1.0, "mass": 100.0,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": True, "is_collidable": True}))
    sim.get_object_template_manager().load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(
        str(tmp / "_navramps.object_config.json"))
    import trimesh
    bb = trimesh.load(str(ramps)).bounds
    c = (bb[0] + bb[1]) / 2.0
    obj.translation = mn.Vector3(float(c[0]), float(c[1]), float(c[2]))
    obj.motion_type = hsim.physics.MotionType.STATIC
    return obj


def check_links(sim, hsim, label):
    ns = V2.make_navmesh_settings(hsim, V2.EMBODIMENTS["human"], include_static=True)
    ok = V2.recompute_navmesh_compat(sim, hsim, ns, True)
    pf = sim.pathfinder
    fails = []
    for l1, l2 in list(zip(CHAIN, CHAIN[1:])) + EXTRA:
        sp = hsim.nav.ShortestPath()
        sp.requested_start = PTS[l1]
        sp.requested_end = PTS[l2]
        if not pf.find_path(sp):
            fails.append(f"{l1}->{l2}")
    stat = "全通" if not fails else "断:" + ",".join(fails)
    print(f"  {label}: bake_ok={ok} area={float(pf.navigable_area):.1f} "
          f"isl={int(pf.num_islands)}  {stat}")
    # 所有点 island 归属 + drift
    print("    各点 island/drift:")
    for lbl, pt in PTS.items():
        sn = pf.snap_point(np.array(pt, dtype=np.float32))
        isl = int(pf.get_island(sn))
        d = float(np.linalg.norm(sn - np.array(pt)))
        flag = " <-- 漂移大" if d > 0.35 else ""
        print(f"      {lbl:<8} isl={isl:<3} drift={d:.2f} y_snap={sn[1]:.2f}{flag}")


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))
    rigid_mgr = sim.get_rigid_object_manager()
    ramp = add_ramps(sim, hsim, mn)

    # A 组：保持现状（STATIC=STATIC, DYNAMIC=DYNAMIC）
    check_links(sim, hsim, "A DYNAMIC不参与烘焙")

    # B 组：全部转 STATIC（builder 旧行为）
    saved = {}
    for h in rigid_mgr.get_object_handles(""):
        o = rigid_mgr.get_object_by_handle(h)
        if o is not None:
            saved[o.object_id] = o.motion_type
            o.motion_type = hsim.physics.MotionType.STATIC
    check_links(sim, hsim, "B 全部转STATIC(旧)")
    for h in rigid_mgr.get_object_handles(""):
        o = rigid_mgr.get_object_by_handle(h)
        if o is not None and o.object_id in saved:
            o.motion_type = saved[o.object_id]

    if ramp:
        rigid_mgr.remove_object_by_id(ramp.object_id)
    sim.close()


if __name__ == "__main__":
    main()
