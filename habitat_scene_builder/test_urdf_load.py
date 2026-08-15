#!/usr/bin/env python3
"""test_urdf_load.py — 验证 Habitat 加载缺 mesh 的机器人 URDF 的行为。"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
G1 = HERE.parent / "demo_assets" / "robots" / "g1" / "g1.urdf"
GO2 = HERE.parent / "demo_assets" / "robots" / "go2" / "go2_description.urdf"
FETCH = HERE.parent / "demo_assets" / "robots" / "fetch" / "fetch.urdf"


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))

    aom = sim.get_articulated_object_manager()
    for name, path in (("g1", G1), ("go2", GO2), ("fetch", FETCH)):
        print(f"\n=== 加载 {name}: {path} ===")
        try:
            robot = aom.add_articulated_object_from_urdf(
                str(path), fixed_base=False)
            if robot is None:
                print("  返回 None")
                continue
            print(f"  加载成功: links={len(robot.get_link_ids())}")
            robot.translation = mn.Vector3(-3.0, 0.0, 1.7)
            print(f"  translation 设置 OK -> {robot.translation}")
            aom.remove_object_by_id(robot.object_id)
        except Exception as e:  # noqa: BLE001
            print(f"  加载失败: {type(e).__name__}: {e}")
    sim.close()


if __name__ == "__main__":
    main()
