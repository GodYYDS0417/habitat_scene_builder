#!/usr/bin/env python3
"""render_robots.py — 把 G1/Go2/Fetch 放进客厅，渲染验证外观。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
ROBOTS = HERE.parent / "demo_assets" / "robots"
OUT = RUN / "_preview" / "robots"

SPAWN = [
    ("g1", ROBOTS / "g1" / "g1.urdf", (-2.2, 0.0, 1.7), 1.2),
    ("go2", ROBOTS / "go2" / "go2_description.urdf", (-3.6, 0.0, 1.0), -0.6),
    ("fetch", ROBOTS / "fetch" / "fetch.urdf", (-1.0, 0.0, 0.6), 2.4),
]
VIEWS = [
    ("robots_wide", (-0.2, 1.6, -0.3), (-2.6, 0.6, 1.6)),
    ("g1_close", (-1.4, 1.3, 0.6), (-2.2, 0.9, 1.7)),
    ("go2_close", (-3.0, 0.8, 0.2), (-3.6, 0.35, 1.0)),
    ("fetch_close", (-0.4, 1.2, 1.6), (-1.0, 0.7, 0.6)),
]


def main() -> None:
    hsim, mn = V2._import_habitat()
    from habitat_sim.utils.common import quat_from_magnum
    OUT.mkdir(parents=True, exist_ok=True)
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    spec = hsim.CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = hsim.SensorType.COLOR
    spec.resolution = [640, 640]
    spec.position = mn.Vector3(0.0, 0.0, 0.0)
    agent_cfg = hsim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)

    aom = sim.get_articulated_object_manager()
    for name, path, pos, yaw in SPAWN:
        r = aom.add_articulated_object_from_urdf(str(path), fixed_base=False)
        if r is None:
            print(f"{name} 加载失败")
            continue
        r.translation = mn.Vector3(*pos)
        r.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
        print(f"{name} 放置于 {pos}")
    for _ in range(90):
        sim.step_physics(1.0 / 60)

    for name, eye, target in VIEWS:
        m = mn.Matrix4.look_at(mn.Vector3(*eye), mn.Vector3(*target), mn.Vector3(0, 1, 0))
        q = quat_from_magnum(mn.Quaternion.from_matrix(m.rotation()))
        st = hsim.agent.AgentState()
        st.position = np.array(eye, dtype=np.float32)
        st.rotation = q
        agent.set_state(st)
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["rgb"])[..., :3]
        Image.fromarray(rgb).save(OUT / f"{name}.png")
        print(f"  {name}.png")
    sim.close()
    print(f"渲染完成 -> {OUT}")


if __name__ == "__main__":
    main()
