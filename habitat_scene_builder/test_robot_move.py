#!/usr/bin/env python3
"""test_robot_move.py — 验证机器人 KINEMATIC 沿 navmesh 移动稳定不倒。"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from robot_actor import RobotActor  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
ROBOTS = HERE.parent / "demo_assets" / "robots"


def up_y(obj):
    m = obj.rotation.to_matrix()
    return float(m[1][1])   # base 本地 Y 轴在世界的 y 分量（1=竖直，~0=倒）


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))
    pf = sim.pathfinder
    pf.load_nav_mesh(str(RUN / "navmeshes" / "villa_2f_000__human.navmesh"))

    start = np.array([-3.0, 0.10, 1.70])   # 客厅
    goal = np.array([2.0, 0.10, -1.50])    # F1 走廊
    g1 = RobotActor(sim, hsim, mn, "g1", ROBOTS / "g1" / "g1.urdf", start, yaw=0.0)
    print(f"motion_type = {g1.obj.motion_type}")
    print(f"起点 pos={np.round(g1.pos,2)} up_y={up_y(g1.obj):.3f}")

    # 沿 navmesh 走到走廊，途中 step_physics 模拟物理
    def tick():
        sim.step_physics(1.0 / 60)
    ok = g1.nav_to(pf, hsim, goal, cb=tick)
    print(f"nav_to 成功={ok}  终点 pos={np.round(g1.pos,2)}  距目标={np.linalg.norm(g1.pos[[0,2]]-goal[[0,2]]):.2f}m")
    print(f"移动后 up_y={up_y(g1.obj):.3f}（应保持≈1.0 竖直不倒）")

    # 静置 2 秒验证 KINEMATIC 不被物理扰动
    for _ in range(120):
        sim.step_physics(1.0 / 60)
    print(f"静置2s后 pos={np.round(g1.pos,2)} up_y={up_y(g1.obj):.3f}")
    sim.close()


if __name__ == "__main__":
    main()
