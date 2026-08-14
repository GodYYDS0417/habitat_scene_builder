#!/usr/bin/env python3
"""probe_v2.py — 单步调试放置失败原因。"""
import sys, math
from pathlib import Path

sys.path.insert(0, "/root/C--Explore/habitat_scene_builder/habitat_scene_builder")
import habitat_scene_builder_v2 as V2

hsim, mn = V2._import_habitat()
import numpy as np

spec = V2.BuildSpec(
    stage_src=Path("/root/rl/simulation_habitat-main/data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb"),
    objects_src=Path("/root/C--Explore/habitat_scene_builder/demo_assets/objects"),
    out_root=Path("/root/C--Explore/habitat_scene_builder/demo_run/living_demo"),
    dataset_name="living_demo", scene_name="living_demo_000",
    embodiments=["spot"],
)
spec.object_classes = sorted(V2.OBJECT_PROFILES)

sim = V2.open_sim(hsim, spec, with_sensors=False)
print("pathfinder loaded:", sim.pathfinder.is_loaded)
if not sim.pathfinder.is_loaded:
    print("bootstrap:", V2.bootstrap_navmesh(sim, hsim))

# 1) 随机可行走点 + snap_point 行为
for i in range(5):
    pt = sim.pathfinder.get_random_navigable_point()
    print("nav pt:", np.asarray(pt), "island:", sim.pathfinder.island_radius(pt))

# 2) 表面检测细节
surfaces = V2.detect_surfaces(sim, hsim, mn, debug=True)
for s in surfaces:
    print("SURFACE", s.category, "center", tuple(round(c,2) for c in s.center),
          "ext", tuple(round(e,2) for e in s.extent_xz), "n=", s.n_points)

# 3) 试放一个 table 到地面块中心
floors = [s for s in surfaces if s.category == "floor"]
if floors:
    floor = max(floors, key=lambda s: s.n_points)
    mgr = sim.get_object_template_manager()
    handles = V2.resolve_template_handles(spec, mgr)
    print("handles:", len(handles), list(handles)[:5])
    rm = sim.get_rigid_object_manager()
    obj = rm.add_object_by_template_handle(handles["table"])
    print("obj:", obj)
    if obj is not None:
        node = obj.root_scene_node
        bb = hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        print("bb size:", bb.size_x(), bb.size_y(), bb.size_z())
        x, y, z = floor.center
        obj.translation = mn.Vector3(x, y + bb.size_y()/2 + 0.03, z)
        obj.motion_type = hsim.physics.MotionType.KINEMATIC
        print("contact at floor center:", V2.contact_test(sim, obj))
        # snap_point 检查
        try:
            snap = sim.pathfinder.snap_point(mn.Vector3(x, y, z))
            print("snap:", snap, "dist2d:", math.hypot(snap[0]-x, snap[2]-z))
        except Exception as e:
            print("snap_point EXC:", e)
        # 静置
        obj.motion_type = hsim.physics.MotionType.DYNAMIC
        V2.settle(sim, 1.5)
        t = obj.translation
        print("settled pos:", t.x, t.y, t.z, "contact:", V2.contact_test(sim, obj))
sim.close()
print("PROBE_DONE")
