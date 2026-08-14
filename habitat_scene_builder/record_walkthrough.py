#!/usr/bin/env python3
"""record_walkthrough.py — 给生成好的场景数据集录巡检视频。

内容两段：
  1) walkthrough：agent 沿 navmesh 在主房间走一圈（贪心最远点选路），
     航向沿路径切线平滑插值，展示整体布局。
  2) interaction：镜头对准一个 DYNAMIC 物体，给它一个水平速度，
     录下它滑动/倾倒的过程——证明场景是物理可交互的，不是动画。

产出：
  <dataset>/_preview/walkthrough.mp4        RGB 巡检 + 推物体
  <dataset>/_preview/walkthrough_rgbd.mp4   RGB | 深度 | 语义 三联屏

用法：
  python record_walkthrough.py \
      --dataset /path/living_demo.scene_dataset_config.json \
      --scene living_demo_000 [--fps 30] [--resolution 512]
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


def quat_yaw(hsim, yaw: float):
    from habitat_sim.utils.common import quat_from_angle_axis
    return quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0]))


def set_agent(sim, hsim, pos, yaw):
    agent = sim.agents[0]
    st = agent.get_state()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = quat_yaw(hsim, yaw)
    agent.set_state(st)


def tour_points(sim, n_wp: int = 5, min_island: float = 1.5) -> list[np.ndarray]:
    """在主导岛屿上贪心最远点采样 n_wp 个途经点，串成闭环路径。"""
    pf = sim.pathfinder
    pts: list[np.ndarray] = []
    for _ in range(400):
        p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
        if not np.all(np.isfinite(p)):
            continue
        if pf.island_radius(p) < min_island:
            continue
        pts.append(p)
        if len(pts) >= 200:
            break
    if len(pts) < 4:
        raise RuntimeError("navmesh 上采不到足够的点")
    # 最远点采样选路点
    pts_arr = np.stack(pts)
    wps = [pts_arr[0]]
    while len(wps) < n_wp:
        d = np.min(np.stack(
            [np.linalg.norm(pts_arr[:, [0, 2]] - w[[0, 2]], axis=1)
             for w in wps]), axis=0)
        wps.append(pts_arr[int(np.argmax(d))])
    # find_path 串联成环
    path: list[np.ndarray] = []
    for i in range(len(wps) + 1):
        a, b = wps[i % len(wps)], wps[(i + 1) % len(wps)]
        sp = hsim_paths(sim, a, b)
        if path and sp:
            sp = sp[1:]
        path.extend(sp)
    return path


def hsim_paths(sim, a, b) -> list[np.ndarray]:
    """find_path 兼容封装，失败时退化为直线。"""
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    if sim.pathfinder.find_path(sp) and len(sp.points) >= 2:
        return [np.asarray(p, dtype=np.float32) for p in sp.points]
    return [np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)]


def resample(path: list[np.ndarray], step: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = float(np.linalg.norm(b[[0, 2]] - a[[0, 2]]))
        n = max(1, int(seg / step))
        for k in range(n):
            t = k / n
            out.append(a * (1 - t) + b * t)
    return out


def grab_frame(sim, label: str | None = None) -> np.ndarray:
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3].copy()
    if label:
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb)
        ImageDraw.Draw(im).text((10, 10), label, fill=(255, 255, 80))
        rgb = np.asarray(im)
    return rgb


def grab_triptych(sim) -> np.ndarray:
    """RGB | 深度 | 语义 三联屏（各缩放到同高）。"""
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    d = np.clip(depth / 8.0, 0, 1) ** 0.5          # 近亮远暗
    depth_rgb = (np.stack([1 - d, np.abs(d - 0.5) * 2, d], -1) * 255).astype(np.uint8)
    sem = np.asarray(obs["semantic"], dtype=np.int32)
    palette = (np.stack([sem * 67 % 256, sem * 131 % 256, sem * 197 % 256], -1)
               ).astype(np.uint8)
    palette[sem == 0] = (40, 40, 40)
    h = rgb.shape[0]
    return np.concatenate([rgb, depth_rgb, palette], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hsim, mn = V2._import_habitat()
    import habitat_sim  # noqa

    spec = V2.BuildSpec(
        stage_src=Path("dummy.glb"),
        objects_src=None,
        out_root=Path(args.dataset).resolve().parent,
        dataset_name=Path(args.dataset).name.split(".")[0],
        scene_name=args.scene,
    )
    sim = V2.open_sim(hsim, spec, with_sensors=True, resolution=args.resolution)
    out_dir = Path(args.out) if args.out else spec.out_root / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    wr_rgb = imageio.get_writer(
        str(out_dir / "walkthrough.mp4"), fps=args.fps, codec="libx264",
        quality=8, macro_block_size=None)
    wr_tri = imageio.get_writer(
        str(out_dir / "walkthrough_rgbd.mp4"), fps=args.fps, codec="libx264",
        quality=8, macro_block_size=None)
    try:
        # ---- 段 1：navmesh 巡检（镜头始终对准摆放物中心，类似环绕展示）----
        rigid_mgr = sim.get_rigid_object_manager()
        obj_pos = []
        for h in rigid_mgr.get_object_handles(""):
            o = rigid_mgr.get_object_by_handle(h)
            if o is not None:
                obj_pos.append(np.asarray(o.translation))
        look_at = (np.mean(np.stack(obj_pos), axis=0)
                   if obj_pos else None)

        path = resample(tour_points(sim), step=0.045)
        print(f"巡检路径 {len(path)} 帧, 注视点 {None if look_at is None else np.round(look_at, 2)}")
        prev_yaw = None
        for i, p in enumerate(path):
            if look_at is not None:
                d = look_at - p
                yaw = math.atan2(-d[0], -d[2])  # habitat 前方是 -Z
            elif i < len(path) - 1:
                d = path[i + 1] - p
                yaw = math.atan2(-d[0], -d[2])
            else:
                yaw = prev_yaw or 0.0
            if prev_yaw is not None:            # 平滑镜头转动
                dy = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                yaw = prev_yaw + 0.35 * dy
            prev_yaw = yaw
            set_agent(sim, hsim, p, yaw)
            wr_rgb.append_data(grab_frame(sim, "walkthrough"))
            wr_tri.append_data(grab_triptych(sim))

        # ---- 段 2：推一个 DYNAMIC 物体 ----
        target = None
        for h in rigid_mgr.get_object_handles(""):
            o = rigid_mgr.get_object_by_handle(h)
            if o is not None and o.motion_type == hsim.physics.MotionType.DYNAMIC:
                target = o
                break
        if target is not None:
            tp = np.asarray(target.translation)
            # 相机架在物体附近的 navmesh 上（房间内侧），而不是盲区偏移
            snap = np.asarray(sim.pathfinder.snap_point(
                mn.Vector3(float(tp[0]), float(tp[1]), float(tp[2]))), dtype=np.float32)
            if not np.all(np.isfinite(snap)):
                snap = tp + np.array([0.9, 0.0, 0.9], dtype=np.float32)
            back = np.array([snap[0] - tp[0], 0.0, snap[2] - tp[2]], dtype=np.float32)
            n = float(np.linalg.norm(back))
            back = back / n if n > 1e-3 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            cam = snap + back * 0.5
            cam_snap = np.asarray(sim.pathfinder.snap_point(
                mn.Vector3(float(cam[0]), float(cam[1]), float(cam[2]))), dtype=np.float32)
            if np.all(np.isfinite(cam_snap)):
                cam = cam_snap
            yaw = math.atan2(-(tp[0] - cam[0]), -(tp[2] - cam[2]))
            set_agent(sim, hsim, cam, yaw)
            for _ in range(args.fps // 2):      # 先静止半秒
                wr_rgb.append_data(grab_frame(sim, "push demo (DYNAMIC object)"))
                wr_tri.append_data(grab_triptych(sim))
            # 沿「垂直于视线」的方向推：在台面上横向滑动，不会立刻飞出画面
            view = np.array([tp[0] - cam[0], 0.0, tp[2] - cam[2]], dtype=np.float32)
            view /= max(float(np.linalg.norm(view)), 1e-3)
            lateral = np.array([-view[2], 0.0, view[0]], dtype=np.float32)
            v = lateral * 0.9 + view * 0.15
            target.linear_velocity = mn.Vector3(float(v[0]), 0.3, float(v[2]))
            for _ in range(int(args.fps * 1.8)):
                sim.step_physics(1.0 / args.fps)
                wr_rgb.append_data(grab_frame(sim, "push demo (DYNAMIC object)"))
                wr_tri.append_data(grab_triptych(sim))
            print(f"推物体演示: {target.handle}")
        else:
            print("没有 DYNAMIC 物体，跳过推物体演示", file=sys.stderr)
    finally:
        wr_rgb.close()
        wr_tri.close()
        sim.close()
    print(f"视频输出 -> {out_dir}/walkthrough.mp4, walkthrough_rgbd.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
