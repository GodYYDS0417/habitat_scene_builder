#!/usr/bin/env python3
"""probe_ramp_only.py — 只放一块大地板 + navramps，隔离坡道烘焙问题。"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"


def add_obj(sim, hsim, glb: Path, name: str):
    tmp = Path(tempfile.mkdtemp(prefix=f"{name}_"))
    (tmp / f"{name}.object_config.json").write_text(json.dumps({
        "render_asset": str(glb), "collision_asset": str(glb),
        "up": [0.0, 1.0, 0.0], "front": [0.0, 0.0, -1.0],
        "units_to_meters": 1.0, "mass": 100.0,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": True, "is_collidable": True}))
    sim.get_object_template_manager().load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(
        str(tmp / f"{name}.object_config.json"))
    assert obj is not None
    obj.motion_type = hsim.physics.MotionType.STATIC
    return obj


def main() -> None:
    hsim, _ = V2._import_habitat()
    import trimesh
    tmp_stage = Path(tempfile.mkdtemp(prefix="rampstage_"))
    floor = trimesh.creation.box(extents=(12.0, 0.10, 12.0))
    floor.apply_translation((0, -0.05, 0))
    fp = tmp_stage / "floor.glb"
    floor.export(fp)
    (tmp_stage / "floor.stage_config.json").write_text(json.dumps({
        "render_asset": "floor.glb", "collision_asset": "floor.glb",
        "up": [0.0, 1.0, 0.0], "front": [0.0, 0.0, -1.0],
        "origin": [0.0, 0.0, 0.0], "units_to_meters": 1.0,
        "gravity": [0.0, -9.8, 0.0], "is_collidable": True}))
    (tmp_stage / "floor.scene_dataset_config.json").write_text(json.dumps({
        "stages": {"paths": {".json": ["."]}},
        "objects": {"paths": {".json": ["."]}},
        "scene_instances": {"paths": {".json": ["."]}}}))
    (tmp_stage / "empty.scene_instance.json").write_text(json.dumps({
        "stage_instance": {"template_name": "floor"},
        "object_instances": []}))

    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(tmp_stage / "floor.scene_dataset_config.json")
    sim_cfg.scene_id = str(tmp_stage / "empty.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))
    ramp_obj = add_obj(sim, hsim, RUN / "stages" / "villa_2f_navramps.glb", "navramps")
    print("ramp obj id:", ramp_obj.object_id,
          "aabb:", ramp_obj.collision_shape_aabb)
    import magnum as mn
    for (x, z) in [(4.35, -1.3), (1.4, 0.65), (4.35, 3.0)]:
        r = sim.cast_ray(hsim.geo.Ray(mn.Vector3(x, 5.0, z), mn.Vector3(0, -1, 0)))
        print(f"  ray({x},{z}) hits y:", [round(h.point.y, 3) for h in r.hits][:6])

    for name in ("human", "spot"):
        p = V2.EMBODIMENTS[name]
        ns = hsim.NavMeshSettings()
        ns.set_defaults()
        ns.cell_size = V2.NAVMESH_CELL_SIZE
        ns.cell_height = V2.NAVMESH_CELL_HEIGHT
        ns.agent_radius = p["radius"]
        ns.agent_height = p["height"]
        ns.agent_max_climb = p["max_climb"]
        ns.agent_max_slope = p["max_slope"]
        if hasattr(ns, "include_static_objects"):
            ns.include_static_objects = True
        V2.recompute_navmesh_compat(sim, hsim, ns, True)
        pf = sim.pathfinder
        v = np.asarray(pf.build_navmesh_vertices())
        m1 = (v[:,0]>3.7)&(v[:,0]<5.0)&(v[:,2]>-3.4)&(v[:,2]<0.0)&(v[:,1]>0.02)&(v[:,1]<1.6)
        m2 = (v[:,0]>0.0)&(v[:,0]<3.4)&(v[:,2]>0.15)&(v[:,2]<1.15)&(v[:,1]>1.2)&(v[:,1]<3.0)
        print(f"{name}: area={pf.navigable_area:.1f} 段1坡道顶点={m1.sum()} 段2={m2.sum()}")
        if m1.sum():
            print(f"   段1 y[{v[m1][:,1].min():.2f},{v[m1][:,1].max():.2f}] "
                  f"z[{v[m1][:,2].min():.2f},{v[m1][:,2].max():.2f}]")
        # 沿段1中线 find_path: (4.35,0.1,-3.2) -> (4.35,1.5,-0.2)
        sp = hsim.nav.ShortestPath()
        sp.requested_start = (4.35, 0.1, -3.2)
        sp.requested_end = (4.35, 1.5, -0.15)
        print(f"   段1爬坡: {'OK' if pf.find_path(sp) else 'FAIL'}")
    sim.close()


if __name__ == "__main__":
    main()
