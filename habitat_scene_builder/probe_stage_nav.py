#!/usr/bin/env python3
"""probe_stage_nav.py — 空场景(仅 stage + navramps, 无家具)烘焙 navmesh，
验证门洞/楼梯本身是否连通。用于区分 stage 几何问题 vs 家具堵门。"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from probe_nav import PTS, CHAIN, EXTRA  # noqa: E402

ASSETS = HERE.parent / "demo_assets"
RUN = HERE.parent / "demo_run" / "villa_2f"


def main() -> None:
    hsim, mn = V2._import_habitat()
    # 复用正式 dataset（stage 朝向正确），写一份空物体的临时 scene_instance
    import json
    empty = RUN / "scenes" / "_empty.scene_instance.json"
    empty.write_text(json.dumps({
        "stage_instance": {"template_name": "stages/villa_2f"},
        "default_lighting": "",
        "object_instances": []}))
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(empty)
    sim_cfg.enable_physics = True
    agent_cfg = hsim.agent.AgentConfiguration()
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))

    # 加 navramps（与 _add_navramps 相同的模板写法）
    ramps = RUN / "stages" / "villa_2f_navramps.glb"
    tmp = Path(tempfile.mkdtemp(prefix="navramps_probe_"))
    (tmp / "_navramps.object_config.json").write_text(json.dumps({
        "render_asset": str(ramps), "collision_asset": str(ramps),
        "units_to_meters": 1.0, "mass": 100.0,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": True, "is_collidable": True}))
    sim.get_object_template_manager().load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(
        str(tmp / "_navramps.object_config.json"))
    assert obj is not None, "navramps 加载失败"
    obj.motion_type = hsim.physics.MotionType.STATIC
    # habitat 按包围盒中心重置物体局部原点，平移回原世界位置
    import trimesh
    bb = trimesh.load(str(ramps)).bounds
    c = (bb[0] + bb[1]) / 2.0
    obj.translation = mn.Vector3(float(c[0]), float(c[1]), float(c[2]))

    out_dir = HERE / "island_maps" / "stage_only" / "navmeshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("human", "spot"):
        preset = V2.EMBODIMENTS[name]
        ns = V2.make_navmesh_settings(hsim, preset, include_static=True)
        ok = V2.recompute_navmesh_compat(sim, hsim, ns, True)
        assert ok, f"navmesh 重算失败: {name}"
        pf = sim.pathfinder
        pf.save_nav_mesh(str(out_dir / f"villa_2f_000__{name}.navmesh"))
        print(f"\n=== {name} (stage-only) area={pf.navigable_area:.1f} m^2, "
              f"islands={pf.num_islands}")
        for label, pt in PTS.items():
            sn = pf.snap_point(np.array(pt, dtype=np.float32))
            print(f"  {label:<8} island={int(pf.get_island(sn)):<3} "
                  f"drift={float(np.linalg.norm(sn - np.array(pt))):.2f}m")
        print("  -- 主链 --")
        for l1, l2 in zip(CHAIN, CHAIN[1:]):
            sp = hsim.nav.ShortestPath()
            sp.requested_start = PTS[l1]
            sp.requested_end = PTS[l2]
            found = pf.find_path(sp)
            print(f"  {'OK ' if found else 'FAIL'} {l1} -> {l2}")
        print("  -- 支路 --")
        for l1, l2 in EXTRA:
            sp = hsim.nav.ShortestPath()
            sp.requested_start = PTS[l1]
            sp.requested_end = PTS[l2]
            found = pf.find_path(sp)
            print(f"  {'OK ' if found else 'FAIL'} {l1} -> {l2}")
    sim.close()


if __name__ == "__main__":
    main()
