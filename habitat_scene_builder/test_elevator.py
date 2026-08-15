#!/usr/bin/env python3
"""test_elevator.py — 不渲染验证电梯系统：按钮判定 + 门开关 + 轿厢升降。"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from elevator_system import ElevatorSystem, BUTTON_REACH_Y, ROBOT_REACH  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
OBJ = HERE.parent / "demo_assets" / "objects"


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))

    elev = ElevatorSystem(sim, hsim, mn, OBJ)
    print(f"轿厢: {elev.cabin.handle if elev.cabin is not None else None}  y1={elev.cabin_y1}")
    print(f"门扇数={len(elev.doors)}  面板数={len(elev.panels)}  按钮触发高={BUTTON_REACH_Y:.2f}m")

    print("\n=== 按钮可达性判定（机器人站电梯门口 3.4,0,-3.0）===")
    robot_pos = (3.4, 0.0, -3.0)
    for rt in ("go2", "g1", "fetch"):
        ok, msg = elev.try_press(rt, robot_pos)
        print(f"  [{ '✓' if ok else '✗'}] {msg}")

    print("\n=== 门扇开关 ===")
    l0 = float(elev.doors[(1, 'l')].translation.x)
    r0 = float(elev.doors[(1, 'r')].translation.x)
    elev.set_doors(1, True, frames=10)
    l1 = float(elev.doors[(1, 'l')].translation.x)
    r1 = float(elev.doors[(1, 'r')].translation.x)
    print(f"  左门 x: {l0:.2f}->{l1:.2f}（左移 {(l0-l1):.2f}）  右门 x: {r0:.2f}->{r1:.2f}（右移 {(r1-r0):.2f}）")

    print("\n=== 轿厢升降 ===")
    if elev.cabin is not None:
        y0 = float(elev.cabin.translation.y)
        elev.move_cabin(2, frames=12)
        y1 = float(elev.cabin.translation.y)
        print(f"  轿厢 y: {y0:.2f} -> {y1:.2f}（Δ={y1-y0:.2f}，应为 ~2.90）")
        elev.move_cabin(1, frames=12)
        print(f"  返回 F1 y: {float(elev.cabin.translation.y):.2f}")
    sim.close()
    print("\n电梯系统逻辑验证完成")


if __name__ == "__main__":
    main()
