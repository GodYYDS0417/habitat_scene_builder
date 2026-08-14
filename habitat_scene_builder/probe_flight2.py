#!/usr/bin/env python3
"""probe_flight2.py — 密集采样楼梯坡道 navmesh 覆盖，精确定位断点。

1) human: 沿段1/段2坡面每 0.1m 采一个点，打 snap 漂移 -> 断点位置
2) spot:  逐参数变体(default / climb0.2 / slope45 / radius0.3 / height1.7)
   找哪个参数导致坡道不可走
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
SLOPE = 0.145 / 0.28


def open_sim(hsim, mn):
    empty = RUN / "scenes" / "_empty.scene_instance.json"
    empty.write_text(json.dumps({
        "stage_instance": {"template_name": "stages/villa_2f"},
        "default_lighting": "",
        "object_instances": []}))
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(empty)
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))
    ramps = RUN / "stages" / "villa_2f_navramps.glb"
    tmp = Path(tempfile.mkdtemp(prefix="navramps_f2_"))
    (tmp / "_navramps.object_config.json").write_text(json.dumps({
        "render_asset": str(ramps), "collision_asset": str(ramps),
        "units_to_meters": 1.0, "mass": 100.0,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": True, "is_collidable": True}))
    sim.get_object_template_manager().load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(
        str(tmp / "_navramps.object_config.json"))
    assert obj is not None
    obj.motion_type = hsim.physics.MotionType.STATIC
    import trimesh
    bb = trimesh.load(str(ramps)).bounds
    c = (bb[0] + bb[1]) / 2.0
    obj.translation = mn.Vector3(float(c[0]), float(c[1]), float(c[2]))
    return sim


def bake(sim, hsim, preset):
    ns = V2.make_navmesh_settings(hsim, preset, include_static=True)
    ok = V2.recompute_navmesh_compat(sim, hsim, ns, True)
    assert ok
    return sim.pathfinder


def scan_line(pf, label, pts):
    print(f"  -- {label} --")
    bad = 0
    for name, pt in pts:
        sn = pf.snap_point(np.array(pt, dtype=np.float32))
        drift = float(np.linalg.norm(sn - np.array(pt)))
        flag = "" if drift < 0.35 else "  <-- 断"
        if drift >= 0.35:
            bad += 1
        print(f"    {name}: drift={drift:.2f} y_snap={sn[1]:.2f} "
              f"isl={int(pf.get_island(sn))}{flag}")
    return bad


def f2_pts():
    """段2坡面线 (z=0.65)，x 从 3.2 到 0.0 每 0.15m。"""
    pts = []
    for x in np.arange(3.2, -0.01, -0.15):
        if x >= 2.7:
            y = 1.595 - (x - 2.7) * SLOPE
        else:
            y = 1.595 + (2.7 - x) * SLOPE
        pts.append((f"x={x:.2f}", (float(x), float(y) + 0.12, 0.65)))
    pts.append(("F2楼板x=0.0", (0.0, 3.02, 0.65)))
    return pts


def f1_pts():
    """段1坡面线 (x=4.35)，z 从 -3.1 到 -0.1 每 0.15m。"""
    pts = []
    for z in np.arange(-3.1, -0.09, 0.15):
        y = 0.145 + (z + 2.6) * SLOPE
        pts.append((f"z={z:.2f}", (4.35, float(y) + 0.12, float(z))))
    return pts


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim = open_sim(hsim, mn)

    print("=== human 坡面密集采样（修墙后）===")
    pf = bake(sim, hsim, V2.EMBODIMENTS["human"])
    scan_line(pf, "段2 (z=0.65)", f2_pts())

    print("\n=== 具身 x 参数 全链路 find_path ===")
    from probe_nav import PTS, CHAIN, EXTRA

    def full_check(pf, label):
        fails = []
        pairs = list(zip(CHAIN, CHAIN[1:])) + EXTRA
        for l1, l2 in pairs:
            sp = hsim.nav.ShortestPath()
            sp.requested_start = PTS[l1]
            sp.requested_end = PTS[l2]
            if not pf.find_path(sp):
                fails.append(f"{l1}->{l2}")
        status = "全通" if not fails else "断: " + ",".join(fails)
        print(f"  {label:<44} area={pf.navigable_area:6.1f}  {status}")

    for ch, spot_climb in ((0.10, 0.20), (0.05, 0.15)):
        V2.NAVMESH_CELL_HEIGHT = ch
        full_check(bake(sim, hsim, V2.EMBODIMENTS["human"]),
                   f"ch={ch} human(climb0.20)")
        full_check(bake(sim, hsim, {**V2.EMBODIMENTS["spot"],
                                    "max_climb": spot_climb}),
                   f"ch={ch} spot(climb{spot_climb})")
        full_check(bake(sim, hsim, V2.EMBODIMENTS["fetch"]),
                   f"ch={ch} fetch(climb0.05)")
    sim.close()


if __name__ == "__main__":
    main()
