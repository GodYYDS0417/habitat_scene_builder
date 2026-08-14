#!/usr/bin/env python3
"""probe_villa.py — 探针：向下射线看碰撞面高度 + 丢一个 keyboard 看落点。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402


def main():
    hsim, mn = V2._import_habitat()
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    import numpy as np

    spec = V2.BuildSpec(
        stage_src=Path("/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/stages/villa_2f.glb"),
        objects_src=Path("/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/objects"),
        out_root=Path("/root/C--Explore/habitat_scene_builder/demo_run/villa_2f"),
        dataset_name="villa_2f", scene_name="villa_2f_000",
        stage_up="y", object_up="y", units_to_meters=1.0,
        stage_collision_asset=None, embodiments=["spot"], interactive_ratio=0.7)
    sim = V2.open_sim(hsim, spec, with_sensors=False)

    # 1) 向下射线：看各点碰撞面
    spots = [("F1地板", (-4.55, 4.40)), ("餐桌", (-1.5, -1.1)), ("书桌", (2.4, 4.36)),
             ("茶几", (-3.32, 2.6)), ("厨房台面", (-3.5, -4.5)), ("床头柜F2", (-4.6, 4.25)),
             ("F2书桌", (-0.3, -4.36)), ("F2地板", (-3.0, 3.0))]
    for name, (x, z) in spots:
        for y0 in (1.5, 4.2):
            ray = hsim.geo.Ray(mn.Vector3(x, y0, z), mn.Vector3(0, -1, 0))
            hit = sim.cast_ray(ray)
            if hit.has_hits():
                print(f"{name:8s} ({x:6.2f},{z:6.2f}) from y={y0}: hit y = {hit.hits[0].point.y:.4f}")
            else:
                print(f"{name:8s} ({x:6.2f},{z:6.2f}) from y={y0}: 无命中")

    # 2) 丢 keyboard 到书桌上
    obj_tmpl_mgr = sim.get_object_template_manager()
    handles = V2.resolve_template_handles(spec, obj_tmpl_mgr)
    rigid_mgr = sim.get_rigid_object_manager()
    obj = rigid_mgr.add_object_by_template_handle(handles["keyboard"])
    obj.motion_type = hsim.physics.MotionType.DYNAMIC
    obj.translation = mn.Vector3(2.4, 0.80, 4.36)
    obj.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    for step in range(16):
        sim.step_physics(0.125)
        t = obj.translation
        print(f"t={(step+1)*0.125:5.3f}s  keyboard y={t.y:.4f} x={t.x:.3f} z={t.z:.3f}")
    sim.close()


if __name__ == "__main__":
    main()
