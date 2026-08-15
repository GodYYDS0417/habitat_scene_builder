#!/usr/bin/env python3
"""test_hetero_nav.py — 异构导航验证：Go2 爬楼梯跨层、G1/Fetch 楼层内移动。"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402
from robot_actor import RobotActor  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
ROBOTS = HERE.parent / "demo_assets" / "robots"
URDF = {"g1": ROBOTS/"g1"/"g1.urdf", "go2": ROBOTS/"go2"/"go2_description.urdf",
        "fetch": ROBOTS/"fetch"/"fetch.urdf"}
NAV = {"g1": "human", "go2": "spot", "fetch": "fetch"}


def main() -> None:
    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [hsim.agent.AgentConfiguration()]))

    navs = {}
    for key, nav in NAV.items():
        pf = hsim.nav.PathFinder()
        pf.load_nav_mesh(str(RUN / "navmeshes" / f"villa_2f_000__{nav}.navmesh"))
        navs[key] = pf

    # Go2 爬楼梯：F1 客厅 -> F2 走廊（跨层）
    go2 = RobotActor(sim, hsim, mn, "go2", URDF["go2"], (-3.0, 0.10, 1.70))
    ok = go2.nav_to(navs["go2"], hsim, (-0.30, 2.95, 1.50))
    print(f"[Go2 爬楼梯] F1客厅->F2走廊: {'OK' if ok else 'FAIL'}  终点 y={go2.pos[1]:.2f}")

    # G1 楼层内：F1 客厅 -> F1 书房
    g1 = RobotActor(sim, hsim, mn, "g1", URDF["g1"], (-3.0, 0.10, 1.70))
    ok = g1.nav_to(navs["g1"], hsim, (4.30, 0.10, 4.30))
    print(f"[G1 楼层内] F1客厅->F1书房: {'OK' if ok else 'FAIL'}  终点={np.round(g1.pos,2)}")

    # Fetch 楼层内：F1 客厅 -> 餐厅
    fetch = RobotActor(sim, hsim, mn, "fetch", URDF["fetch"], (-3.0, 0.10, 1.70))
    ok = fetch.nav_to(navs["fetch"], hsim, (-0.50, 0.10, -1.10))
    print(f"[Fetch 楼层内] F1客厅->餐厅: {'OK' if ok else 'FAIL'}  终点={np.round(fetch.pos,2)}")

    # Fetch 尝试爬楼梯（应失败：climb 0.05 < 楼梯 0.145）
    ok = fetch.nav_to(navs["fetch"], hsim, (-0.30, 2.95, 1.50))
    print(f"[Fetch 爬楼梯] F1->F2: {'OK(意外)' if ok else 'FAIL(预期，爬不上楼梯需坐电梯)'}")
    sim.close()


if __name__ == "__main__":
    main()
