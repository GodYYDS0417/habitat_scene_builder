#!/usr/bin/env python3
"""verify_villa.py — 两层别墅验收：navmesh 跨层连通性 + 定点巡检渲染图。

1) navmesh: 从 F1 客厅到 F2 书房找路，路径 y 爬升 ~2.9m 即楼梯连通
2) 渲染: 每个功能区的定点视角 RGB 图 -> _preview/villa_check/

用法:
    python verify_villa.py   # 输出到 ../demo_run/villa_2f/_preview/villa_check/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

ROOT = HERE.parent / "demo_run" / "villa_2f"
OUT = ROOT / "_preview" / "villa_check"

# (名字, 相机位置, 注视点) —— 坐标对应 make_villa_plan.py 的户型锚点
VIEWS = [
    ("F1_living",   (0.20, 1.55, 0.60),  (-4.00, 0.40, 3.00)),   # 客厅: 沙发/茶几/电视一线
    ("F1_tv",       (-3.30, 1.50, 2.60), (0.90, 0.75, 2.60)),    # 从沙发看电视柜
    ("F1_dining",   (-0.10, 1.60, 0.50), (-2.60, 0.60, -2.80)),  # 餐厅+厨房
    ("F1_entry",    (-1.40, 1.60, -1.60), (4.20, 0.90, -4.10)),  # 玄关/楼梯/电梯
    ("F1_study",    (1.55, 1.55, 2.20),  (2.50, 0.70, 4.45)),    # F1 书房
    ("F2_corridor", (1.30, 4.35, 1.90),  (3.60, 3.30, -1.20)),   # F2 走廊看楼梯间栏杆
    ("F2_master",   (-0.60, 4.40, 0.90), (-4.20, 3.30, 3.20)),   # 主卧
    ("F2_bedroom2", (-0.90, 4.40, -1.10), (-3.90, 3.35, -4.00)), # 次卧
    ("F2_study",    (0.90, 4.35, -2.10), (-0.70, 3.60, -4.30)),  # F2 书房+书架
    ("F2_tea",      (0.90, 4.35, 1.90),  (2.70, 3.60, 3.80)),    # 茶室
    ("bird_F1",     (-0.60, 2.60, -0.40), (-0.75, 0.0, 0.10)),   # F1 俯瞰
    ("bird_F2",     (-0.60, 5.50, -0.40), (-0.75, 2.9, 0.10)),   # F2 俯瞰
]

# navmesh 跨层检查: (具身, F1 点, F2 点)
NAV_CHECKS = [
    ("human", (-3.0, 0.10, 2.6), (-0.3, 3.00, -4.0)),   # 客厅 -> F2 书房
    ("spot",  (-3.0, 0.10, 2.6), (2.45, 3.00, 3.3)),    # 客厅 -> F2 茶室
]


def check_navmesh(hsim) -> None:
    print("=" * 60)
    print("navmesh 跨层连通性")
    for name, p1, p2 in NAV_CHECKS:
        path = ROOT / "navmeshes" / f"villa_2f_000__{name}.navmesh"
        if not path.exists():
            print(f"[{name}] 缺 navmesh 文件: {path}")
            continue
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(path))
        sp = hsim.nav.ShortestPath()
        sp.requested_start = p1
        sp.requested_end = p2
        found = pf.find_path(sp)
        pts = np.asarray(sp.points)
        if not found or len(pts) < 2:
            print(f"[{name}] FAIL 未找到路径 ({p1} -> {p2})")
            continue
        ys = pts[:, 1]
        geo = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        print(f"[{name}] OK  {len(pts)} 点, 路径长 {geo:.1f} m, "
              f"y: {ys.min():.2f} -> {ys.max():.2f} (跨层需 ~2.9)")


def render_views(hsim, mn) -> None:
    from habitat_sim.utils.common import quat_from_magnum

    OUT.mkdir(parents=True, exist_ok=True)
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(ROOT / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = "villa_2f_000"
    sim_cfg.enable_physics = True

    sensor_spec = hsim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = hsim.SensorType.COLOR
    sensor_spec.resolution = [640, 640]
    sensor_spec.position = [0.0, 0.0, 0.0]

    agent_cfg = hsim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)

    print("=" * 60)
    print(f"定点巡检渲染 -> {OUT}")
    for name, eye, target in VIEWS:
        m = mn.Matrix4.look_at(mn.Vector3(*eye), mn.Vector3(*target),
                               mn.Vector3(0, 1, 0))
        q = quat_from_magnum(mn.Quaternion.from_matrix(m.rotation()))
        state = hsim.agent.AgentState()
        state.position = np.array(eye, dtype=np.float32)
        state.rotation = q
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["rgb"])[..., :3]
        Image.fromarray(rgb).save(OUT / f"{name}.png")
        print(f"  {name}.png")
    sim.close()


def main() -> None:
    hsim, mn = V2._import_habitat()
    check_navmesh(hsim)
    render_views(hsim, mn)
    print("验收完成")


if __name__ == "__main__":
    main()
