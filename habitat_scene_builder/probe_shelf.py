#!/usr/bin/env python3
"""probe_shelf.py — 隔离测试 shelf 碰撞：定位书被弹开/穿透的原因。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

OBJ_DIR = "/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/objects"


def main():
    hsim, mn = V2._import_habitat()

    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    agent_cfg = hsim.agent.AgentConfiguration()
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))

    obj_tmpl_mgr = sim.get_object_template_manager()
    obj_tmpl_mgr.load_configs(OBJ_DIR)
    rigid_mgr = sim.get_rigid_object_manager()

    def spawn(cls, x, y, z, motion):
        """与 instantiate_from_plan 同序：translation -> KINEMATIC -> (DYNAMIC 前再设位姿)。

        habitat 里 add 出来的物体默认 DYNAMIC，此时设 translation 只动渲染节点、
        不动 Bullet 刚体；必须经 KINEMATIC 过渡强制刚体按节点位姿重建。
        """
        h = [t for t in obj_tmpl_mgr.get_file_template_handles("")
             if t.endswith(f"{cls}.object_config.json")][0]
        obj = rigid_mgr.add_object_by_template_handle(h)
        obj.translation = mn.Vector3(x, y, z)
        obj.rotation = mn.Quaternion(mn.Vector3(0, 0, 0), 1.0)
        obj.motion_type = hsim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(x, y, z)
        obj.motion_type = motion
        return obj

    ST = hsim.physics.MotionType.STATIC
    DY = hsim.physics.MotionType.DYNAMIC

    shelf = spawn("shelf", 0.0, 0.815, 0.0, ST)
    st = shelf.translation
    print(f"shelf.translation after spawn = ({st.x:.4f},{st.y:.4f},{st.z:.4f})")

    # ---- 实验 1：书上表面刚好在 board1 顶上方 1.25cm，出生前后对比 ----
    book = spawn("book", 0.0, 0.8525, 0.0, DY)
    bt = book.translation
    print(f"book.translation  after spawn = ({bt.x:.4f},{bt.y:.4f},{bt.z:.4f})  (期望 0,0.8525,0)")
    dt = 1.0 / 240.0
    prev = np.array([bt.x, bt.y, bt.z])
    print("--- 0.2s 细步长追踪（仅打印有突变的点）---")
    for step in range(48):
        sim.step_physics(dt)
        t = book.translation
        cur = np.array([t.x, t.y, t.z])
        jump = np.linalg.norm(cur - prev)
        if step < 8 or jump > 0.01:
            print(f"t={(step+1)*dt:6.4f}s book=({t.x:7.4f},{t.y:7.4f},{t.z:7.4f}) jump={jump:.4f}")
        prev = cur

    rigid_mgr.remove_object_by_id(book.object_id)

    # ---- 实验 2：从 2.2m 高处自由落体，看最终停在哪个面 ----
    book2 = spawn("book", 0.0, 2.2, 0.0, DY)
    print("--- 高空落下（找支撑面真实高度）---")
    dt = 1.0 / 60.0
    last = None
    for step in range(180):
        sim.step_physics(dt)
        t = book2.translation
        if step % 30 == 29:
            print(f"t={(step+1)*dt:5.2f}s book2=({t.x:6.3f},{t.y:7.4f},{t.z:6.3f})")
        last = t
    print(f"book2 最终: ({last.x:.3f},{last.y:.4f},{last.z:.3f})")
    sim.close()


if __name__ == "__main__":
    main()
