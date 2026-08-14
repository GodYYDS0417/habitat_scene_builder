#!/usr/bin/env python3
"""record_villa_tour.py — 两层别墅巡检视频。

四段：
  1) F1 巡检：客厅 -> 餐厅 -> 厨房 -> 走廊 -> 楼梯底
  2) 爬楼梯到 F2：转角平台 -> 顶步（human navmesh 跨层）
  3) F2 巡检：走廊 -> 主卧 -> 茶室
  4) 电梯升降演示（轿厢 F1->F2）+ 推一个 DYNAMIC 小物件

相机朝路径切线（不是死盯物体中心），爬楼时视野沿楼梯方向。

产出: ../demo_run/villa_2f/_preview/villa_tour.mp4
用法: python record_villa_tour.py [--fps 30] [--resolution 512]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import habitat_scene_builder_v2 as V2  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"

# 巡检途经点（human navmesh 已验证连通）
TOUR = [
    (-3.0, 0.10, 1.70),    # F1 客厅
    (-0.5, 0.10, -1.10),   # 餐厅
    (-3.5, 0.10, -3.50),   # 厨房
    (-0.5, 0.10, -1.10),   # 折返餐厅
    (2.0, 0.10, -1.50),    # F1 走廊
    (3.4, 0.10, -2.30),    # 楼梯底
    (4.35, 0.80, -1.30),   # 第5步
    (4.35, 1.50, 0.10),    # 第10步
    (3.8, 1.55, 0.65),     # 转角平台
    (2.30, 2.00, 0.65),    # 后段第3步
    (0.35, 2.95, 0.65),    # 顶步
    (-0.30, 2.95, 1.50),   # F2 走廊
    (-1.50, 2.95, 2.00),   # 主卧
    (-0.30, 2.95, 1.50),   # 折返走廊
    (2.50, 2.95, 3.50),    # 茶室（终点）
]

EYE = 1.55   # 相机离地高度（人眼视角）


def find(pf, hsim, a, b):
    sp = hsim.nav.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    if pf.find_path(sp) and len(sp.points) >= 2:
        return [np.asarray(p, dtype=np.float32) for p in sp.points]
    return [np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)]


def build_path(pf, hsim):
    path: list[np.ndarray] = []
    for i in range(len(TOUR) - 1):
        seg = find(pf, hsim, TOUR[i], TOUR[i + 1])
        if path and seg:
            seg = seg[1:]
        path.extend(seg)
    # 重采样到近似匀速
    out: list[np.ndarray] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(seg / 0.07))
        for k in range(n):
            out.append(a + (b - a) * (k / n))
    out.append(path[-1])
    return out


def set_cam(agent, hsim, pos, yaw):
    st = agent.get_state()
    st.position = np.asarray(pos, dtype=np.float32)
    from habitat_sim.utils.common import quat_from_angle_axis
    st.rotation = quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)


def frame(sim, label=None):
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3].copy()
    if label:
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb)
        ImageDraw.Draw(im).text((12, 12), label, fill=(255, 255, 90))
        rgb = np.asarray(im)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()

    hsim, mn = V2._import_habitat()
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
    sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
    sim_cfg.enable_physics = True
    spec = hsim.CameraSensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = hsim.SensorType.COLOR
    spec.resolution = [args.resolution, args.resolution]
    spec.position = mn.Vector3(0.0, EYE, 0.0)
    spec.orientation = mn.Vector3(-0.12, 0.0, 0.0)   # 略俯视
    agent_cfg = hsim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)

    # 用 human navmesh（跨层连通）
    pf = sim.pathfinder
    pf.load_nav_mesh(str(RUN / "navmeshes" / "villa_2f_000__human.navmesh"))

    out = RUN / "_preview"
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    wr = imageio.get_writer(str(out / "villa_tour.mp4"), fps=args.fps,
                            codec="libx264", quality=8, macro_block_size=None)
    try:
        # ---- 段 1-3：F1 -> 爬楼 -> F2 巡检 ----
        path = build_path(pf, hsim)
        print(f"巡检路径 {len(path)} 帧 ({len(path)/args.fps:.0f}s)")
        prev_yaw = 0.0
        look = 10                      # lookahead：朝向前方 ~0.7m，爬楼不贴侧墙
        for i, p in enumerate(path):
            j = min(i + look, len(path) - 1)
            d = path[j] - p
            if abs(d[0]) + abs(d[2]) > 1e-4:
                yaw = math.atan2(-d[0], -d[2])
            else:
                yaw = prev_yaw
            dy = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
            yaw = prev_yaw + 0.25 * dy
            prev_yaw = yaw
            label = "F1" if p[1] < 1.5 else ("stairs" if p[1] < 2.7 else "F2")
            set_cam(agent, hsim, p, yaw)
            wr.append_data(frame(sim, label))

        # ---- 段 4a：电梯升降演示（F1 -> F2）----
        rigid_mgr = sim.get_rigid_object_manager()
        car = None
        for h in rigid_mgr.get_object_handles(""):
            if h.startswith("elevator_car"):
                car = rigid_mgr.get_object_by_handle(h)
                break
        if car is not None:
            print("电梯演示:", car.handle)
            car.motion_type = hsim.physics.MotionType.KINEMATIC
            ct = car.translation
            cx, cz = float(ct.x), float(ct.z)
            # 相机正对电梯井北门洞（x=4.0 门洞中轴），从楼梯间北侧地面看进井内。
            # 不能放正南——电梯紧邻楼梯段1，正南会被踏步挡住。
            cam_f1 = np.array([4.00, 0.10, -2.55], dtype=np.float32)
            yaw_f1 = math.atan2(-(4.0 - 4.00), -(-4.05 - (-2.55)))   # 正视 -z（北）
            set_cam(agent, hsim, cam_f1, yaw_f1)
            for _ in range(args.fps // 2):
                wr.append_data(frame(sim, "elevator @F1"))
            n = int(args.fps * 2.0)
            for k in range(n):
                y = 2.9 * k / (n - 1)
                car.translation = mn.Vector3(cx, y, cz)
                wr.append_data(frame(sim, "elevator F1 -> F2"))
            # 切到 F2 视角看轿厢到达
            cam_f2 = np.array([4.00, 2.95, -2.55], dtype=np.float32)
            set_cam(agent, hsim, cam_f2, yaw_f1)
            for _ in range(args.fps // 2):
                wr.append_data(frame(sim, "elevator @F2"))
        else:
            print("未找到 elevator_car，跳过电梯演示", file=sys.stderr)

        # ---- 段 4b：推一个 DYNAMIC 小物件 ----
        target = None
        for h in rigid_mgr.get_object_handles(""):
            o = rigid_mgr.get_object_by_handle(h)
            if o is not None and h.startswith("bowl"):
                target = o
                break
        if target is not None:
            tp = np.asarray(target.translation)
            cam = np.array([tp[0] + 1.2, max(0.10, tp[1] - 0.3), tp[2] + 0.4],
                           dtype=np.float32)
            snap = np.asarray(pf.snap_point(mn.Vector3(*cam)), dtype=np.float32)
            if np.all(np.isfinite(snap)):
                cam = snap
            yaw = math.atan2(-(tp[0] - cam[0]), -(tp[2] - cam[2]))
            set_cam(agent, hsim, cam, yaw)
            for _ in range(args.fps // 2):
                wr.append_data(frame(sim, "push (DYNAMIC)"))
            target.linear_velocity = mn.Vector3(0.7, 0.2, -0.4)
            for _ in range(int(args.fps * 1.5)):
                sim.step_physics(1.0 / args.fps)
                wr.append_data(frame(sim, "push (DYNAMIC)"))
            print("推物体:", target.handle)
    finally:
        wr.close()
        sim.close()
    print(f"视频 -> {out}/villa_tour.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
