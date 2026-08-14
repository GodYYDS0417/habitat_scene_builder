#!/usr/bin/env python3
"""probe_shelf2.py — 摸清 shelf 凹面碰撞的幻影接触来源。

实验矩阵:
  A. contact_test: 出生时是否已有接触
  B. 书从 board1 上方 10cm 落下, 看停在哪 / 是否被弹飞
  C. 多节点 GLB(每盒一节点) + join_collision_meshes=false 的对照
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

OBJ_DIR = "/root/C--Explore/habitat_scene_builder/demo_run/villa_2f/objects"


def make_split_shelf(tmpdir: Path) -> Path:
    """和 shelf 相同的 6 个盒子，但不 concatenate —— 每盒一个节点。"""
    import trimesh

    def _box(size, center):
        m = trimesh.creation.box(extents=size)
        m.apply_translation(center)
        return m

    parts = {
        f"board{i}": _box((0.90, 0.03, 0.30), (0, y, 0))
        for i, y in enumerate((0.40, 0.80, 1.20, 1.60))
    }
    parts["panelL"] = _box((0.03, 1.60, 0.30), (-0.435, 0.80, 0))
    parts["panelR"] = _box((0.03, 1.60, 0.30), (0.435, 0.80, 0))
    scene = trimesh.Scene(parts)
    glb = tmpdir / "split_shelf.glb"
    scene.export(glb)
    cfg = json.loads(Path(OBJ_DIR, "shelf.object_config.json").read_text())
    cfg["render_asset"] = "split_shelf.glb"
    cfg["collision_asset"] = "split_shelf.glb"
    cfg["join_collision_meshes"] = False
    (tmpdir / "split_shelf.object_config.json").write_text(json.dumps(cfg, indent=2))
    return tmpdir


def main():
    hsim, mn = V2._import_habitat()
    tmpdir = Path(tempfile.mkdtemp(prefix="split_shelf_"))
    make_split_shelf(tmpdir)

    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True
    agent_cfg = hsim.agent.AgentConfiguration()
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))

    obj_tmpl_mgr = sim.get_object_template_manager()
    obj_tmpl_mgr.load_configs(OBJ_DIR)
    obj_tmpl_mgr.load_configs(str(tmpdir))
    rigid_mgr = sim.get_rigid_object_manager()

    def spawn(cls_or_suffix, x, y, z, motion, directory=None):
        handles = [t for t in obj_tmpl_mgr.get_file_template_handles("")
                   if t.endswith(f"{cls_or_suffix}.object_config.json")]
        h = handles[0]
        obj = rigid_mgr.add_object_by_template_handle(h)
        obj.translation = mn.Vector3(x, y, z)
        obj.rotation = mn.Quaternion(mn.Vector3(0, 0, 0), 1.0)
        obj.motion_type = hsim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(x, y, z)
        obj.motion_type = motion
        return obj

    ST = hsim.physics.MotionType.STATIC
    DY = hsim.physics.MotionType.DYNAMIC

    for shelf_cls in ("shelf", "split_shelf"):
        print(f"\n================ {shelf_cls} ================")
        shelf = spawn(shelf_cls, 0.0, 0.815, 0.0, ST)
        n_bb = len(shelf.collision_shape_aabbs) if hasattr(shelf, "collision_shape_aabbs") else "n/a"
        print("collision sub-shape count:", n_bb)

        # A. 出生即接触?
        book = spawn("book", 0.0, 0.8525, 0.0, DY)
        n = sim.contact_test(book.object_id)
        print(f"A. contact_test@spawn: {n}")

        # B. 静置 1s
        ok = True
        for step in range(60):
            sim.step_physics(1.0 / 60.0)
        t = book.translation
        ok = abs(t.y - 0.84) < 0.03 and abs(t.x) < 0.2 and abs(t.z) < 0.2
        print(f"B. 静置 1s 后 book=({t.x:.3f},{t.y:.4f},{t.z:.3f}) "
              f"{'OK 停在 board1' if ok else 'FAIL 被弹飞/穿透'}")
        rigid_mgr.remove_object_by_id(book.object_id)

        # C. 从 10cm 上方落到 board1
        book = spawn("book", 0.05, 0.94, -0.03, DY)
        for step in range(90):
            sim.step_physics(1.0 / 60.0)
        t = book.translation
        ok = abs(t.y - 0.84) < 0.03 and abs(t.x) < 0.3 and abs(t.z) < 0.2
        print(f"C. 落 10cm 后 book=({t.x:.3f},{t.y:.4f},{t.z:.3f}) "
              f"{'OK 停在 board1' if ok else 'FAIL'}")
        rigid_mgr.remove_object_by_id(book.object_id)
        rigid_mgr.remove_object_by_id(shelf.object_id)
    sim.close()


if __name__ == "__main__":
    main()
