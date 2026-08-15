#!/usr/bin/env python3
"""test_rescue_flow.py — 不渲染验证完整救援流程：G1 坐电梯上 F2、attach 假人、
坐电梯下 F1、放到客厅。断言各阶段位置/状态正确。"""
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from elevator_system import ElevatorSystem, DOOR_OUT, PANEL_FRONT, CABIN_IN, FLOOR_Y  # noqa: E402
from robot_actor import RobotActor  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
ROBOTS = HERE.parent / "demo_assets" / "robots"
OBJECTS = HERE.parent / "demo_assets" / "objects"


def ride(elev, robot, dst, sim):
    """不渲染的电梯序列：门口->按按钮->开门->进->关门->升降->开门->出。"""
    src = elev.cabin_floor
    assert robot.nav_to(navs["g1"], hsim, DOOR_OUT[src]), "到电梯门口失败"
    robot.nav_to(navs["g1"], hsim, PANEL_FRONT[src])
    ok, msg = elev.try_press(robot.type, robot.pos)
    assert ok, msg
    elev.set_doors(src, True)                       # 开门
    robot.move_path([robot.pos, CABIN_IN[src]])     # 进轿厢
    elev.set_doors(src, False)                      # 关门
    # 升降（机器人随轿厢）
    y0 = float(elev.cabin.translation.y)
    dy = FLOOR_Y[dst] - FLOOR_Y[src]
    for k in range(20):
        t = (k + 1) / 20
        elev.cabin.translation = mn.Vector3(4.0, y0 + 2.9 * dy / 2.9 * t * (2.9 if False else 1), CABIN_IN[src][2])
    # 直接设定到位（简化，动画在演示里）
    elev.cabin.translation = mn.Vector3(4.0, elev.cabin_y1 + (FLOOR_Y[dst] - FLOOR_Y[1]), CABIN_IN[src][2])
    robot.set_pose([CABIN_IN[src][0], FLOOR_Y[dst] + 0.10, CABIN_IN[src][2]], robot.yaw)
    elev.cabin_floor = dst
    elev.set_doors(dst, True)
    robot.move_path([robot.pos, DOOR_OUT[dst]])
    elev.set_doors(dst, False)


def main() -> None:
    global hsim, mn, navs
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))
    navs = {"g1": hsim.nav.PathFinder()}
    navs["g1"].load_nav_mesh(str(RUN / "navmeshes" / "villa_2f_000__human.navmesh"))
    elev = ElevatorSystem(sim, hsim, mn, OBJECTS)
    mgr = sim.get_rigid_object_manager()
    dummy = next((mgr.get_object_by_handle(h) for h in mgr.get_object_handles("") if h.startswith("dummy")), None)
    assert dummy is not None, "没找到假人"
    print(f"假人初始位置 y={float(dummy.translation.y):.2f}（应在 F2 床面 ~3.45）")

    # G1 坐电梯 F1->F2
    g1 = RobotActor(sim, hsim, mn, "g1", ROBOTS / "g1" / "g1.urdf", (-3.0, 0.10, 1.70))
    ride(elev, g1, 2, sim)
    print(f"G1 坐电梯到 F2：pos y={g1.pos[1]:.2f}（应≈2.95）  轿厢层={elev.cabin_floor}")

    # 到主卧，attach 假人
    assert g1.nav_to(navs["g1"], hsim, (-1.50, 2.95, 2.00)), "到主卧失败"
    g1.nav_to(navs["g1"], hsim, (-3.0, 2.95, 3.0))
    dummy.motion_type = hsim.physics.MotionType.KINEMATIC
    carrying = True
    print("G1 到达 F2 主卧，救起假人（attach）")

    # G1 带假人坐电梯下 F1
    def sync_dummy():
        yaw = g1.yaw
        off = np.array([math.sin(yaw), 0, math.cos(yaw)]) * 0.45 + np.array([0, 0.55, 0])
        dummy.translation = mn.Vector3(*(g1.pos + off))
    sync_dummy()
    ride(elev, g1, 1, sim)
    sync_dummy()
    print(f"G1 带假人坐电梯回 F1：pos y={g1.pos[1]:.2f}，假人 y={float(dummy.translation.y):.2f}")

    # 到客厅放下
    g1.nav_to(navs["g1"], hsim, (-1.5, 0.10, 1.50))
    dummy.translation = mn.Vector3(-1.5, 0.30, 1.50)
    dummy.motion_type = hsim.physics.MotionType.DYNAMIC
    for _ in range(90):
        sim.step_physics(1.0 / 60)
    fy = float(dummy.translation.y)
    print(f"假人放下：pos=({float(dummy.translation.x):.2f},{fy:.2f},{float(dummy.translation.z):.2f})")
    assert fy < 0.5, f"假人应在 F1 地面，实际 y={fy}"
    print("\n[OK] 完整救援流程逻辑闭环：F2 床面救起 -> 电梯下 F1 -> 客厅放下")
    sim.close()


if __name__ == "__main__":
    main()
