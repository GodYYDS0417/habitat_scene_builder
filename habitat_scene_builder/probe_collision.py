#!/usr/bin/env python3
"""probe_collision.py — 隔离测试：单个 desk 的 collision_shape_aabb 到底什么样。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402


def main():
    hsim, mn = V2._import_habitat()
    import habitat_sim

    # 极简 sim：无 stage（空世界），只注册 desk 模板
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    agent_cfg = hsim.agent.AgentConfiguration()
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))

    obj_tmpl_mgr = sim.get_object_template_manager()
    # 手动注册 desk 模板（直接用数据集里的 config）
    cfg_path = "/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/objects/desk.object_config.json"
    tid = obj_tmpl_mgr.load_configs(str(Path(cfg_path).parent))[0] if False else None
    import glob
    handles_before = set(obj_tmpl_mgr.get_file_template_handles(""))
    obj_tmpl_mgr.load_configs("/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/objects/")
    handles = [h for h in obj_tmpl_mgr.get_file_template_handles("")
               if "desk" in h]
    print("desk templates:", handles)
    tmpl_handle = [h for h in handles if h.endswith("desk.object_config.json")][0]

    rigid_mgr = sim.get_rigid_object_manager()
    obj = rigid_mgr.add_object_by_template_handle(tmpl_handle)
    obj.motion_type = hsim.physics.MotionType.STATIC
    obj.translation = mn.Vector3(0.0, 0.0, 0.0)

    aabb = obj.collision_shape_aabb
    print("collision_shape_aabb:")
    print("  min:", [round(v, 4) for v in (aabb.min.x, aabb.min.y, aabb.min.z)])
    print("  max:", [round(v, 4) for v in (aabb.max.x, aabb.max.y, aabb.max.z)])

    node = obj.root_scene_node
    cbb = node.cumulative_bb
    print("cumulative_bb (render):")
    print("  min:", [round(v, 4) for v in (cbb.min.x, cbb.min.y, cbb.min.z)])
    print("  max:", [round(v, 4) for v in (cbb.max.x, cbb.max.y, cbb.max.z)])

    # 模板属性里的 scale / up / front
    tmpl = obj_tmpl_mgr.get_template_by_handle(tmpl_handle)
    print("template scale:", tmpl.scale)
    print("template up:", tmpl.up, " front:", tmpl.front)
    print("render_asset:", tmpl.render_asset_handle)
    print("collision_asset:", tmpl.collision_asset_handle)
    print("use_bbox_for_collision:", tmpl.use_bounding_box_for_collision)
    print("join_collision_meshes:", tmpl.join_collision_meshes)
    sim.close()


if __name__ == "__main__":
    main()
