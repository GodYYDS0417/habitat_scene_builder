#!/usr/bin/env python3
"""make_room_stage.py — 参数化生成空房间 stage GLB（默认 6m x 5m = 30 m^2）。

配合 habitat_scene_builder_v2 使用：真实扫描（MP3D 小公寓）可行走面积
只有 ~8 m^2，摆不下样板房布局。这个脚本生成一个带四面墙（南墙有窗）、
木地板、踢脚线的空房间，让语义放置有充足空间。

用法:
    python make_room_stage.py --out assets/stages/showroom.glb \
        --width 6.0 --depth 5.0 --height 2.7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _box(size, center, color):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    m.visual = trimesh.visual.ColorVisuals(
        m, face_colors=np.tile(np.array(color, dtype=np.uint8), (len(m.faces), 1)))
    return m


def build_room(width: float, depth: float, height: float,
               wall_t: float = 0.10, with_window: bool = True) -> trimesh.Trimesh:
    """原点 = 房间地面中心 (y=0 是地板顶面)。Y-up，单位米。"""
    WALL = (224, 218, 206, 255)      # 暖白墙
    FLOOR = (146, 116, 88, 255)      # 木地板
    BASE = (245, 245, 245, 255)      # 踢脚线
    SILL = (236, 232, 224, 255)      # 窗台/窗楣

    hw, hd = width / 2.0, depth / 2.0
    parts: list[trimesh.Trimesh] = []

    # 地板（顶面 y=0，向上法线）
    parts.append(_box((width + 2 * wall_t, 0.10, depth + 2 * wall_t),
                      (0, -0.05, 0), FLOOR))

    # 北墙（z = +hd）
    parts.append(_box((width + 2 * wall_t, height, wall_t),
                      (0, height / 2.0, hd + wall_t / 2.0), WALL))
    # 西墙 / 东墙
    parts.append(_box((wall_t, height, depth),
                      (-hw - wall_t / 2.0, height / 2.0, 0), WALL))
    parts.append(_box((wall_t, height, depth),
                      (hw + wall_t / 2.0, height / 2.0, 0), WALL))

    # 南墙（z = -hd）：留 1.4m 宽窗洞（窗台 0.9m，窗高 1.4m）
    if with_window:
        win_w, sill_h, win_h = 1.4, 0.9, 1.4
        seg_w = (width - win_w) / 2.0
        zc = -hd - wall_t / 2.0
        parts.append(_box((seg_w, height, wall_t),
                          (-(win_w + seg_w) / 2.0, height / 2.0, zc), WALL))
        parts.append(_box((seg_w, height, wall_t),
                          ((win_w + seg_w) / 2.0, height / 2.0, zc), WALL))
        parts.append(_box((win_w, sill_h, wall_t), (0, sill_h / 2.0, zc), WALL))
        lintel_h = height - sill_h - win_h
        parts.append(_box((win_w, lintel_h, wall_t),
                          (0, sill_h + win_h + lintel_h / 2.0, zc), SILL))
        # 窗台板
        parts.append(_box((win_w + 0.10, 0.04, wall_t + 0.08),
                          (0, sill_h - 0.02, -hd - wall_t / 2.0 + 0.02), SILL))
    else:
        parts.append(_box((width + 2 * wall_t, height, wall_t),
                          (0, height / 2.0, -hd - wall_t / 2.0), WALL))

    # 踢脚线（高 8cm，贴四面墙内侧；0.08 < 各具身 max_climb，不影响 navmesh）
    sk_h, sk_t = 0.08, 0.015
    parts.append(_box((width, sk_h, sk_t), (0, sk_h / 2.0, hd - sk_t / 2.0), BASE))
    parts.append(_box((width, sk_h, sk_t), (0, sk_h / 2.0, -hd + sk_t / 2.0), BASE))
    parts.append(_box((sk_t, sk_h, depth), (-hw + sk_t / 2.0, sk_h / 2.0, 0), BASE))
    parts.append(_box((sk_t, sk_h, depth), (hw - sk_t / 2.0, sk_h / 2.0, 0), BASE))

    return trimesh.util.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=float, default=6.0, help="x 方向内宽 (m)")
    ap.add_argument("--depth", type=float, default=5.0, help="z 方向内深 (m)")
    ap.add_argument("--height", type=float, default=2.7, help="层高 (m)")
    ap.add_argument("--no-window", action="store_true", help="南墙不开窗")
    args = ap.parse_args()

    mesh = build_room(args.width, args.depth, args.height,
                      with_window=not args.no_window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    area = args.width * args.depth
    print(f"{args.out}  内空 {args.width:.1f} x {args.depth:.1f} m = {area:.0f} m^2, "
          f"层高 {args.height:.1f} m, faces = {len(mesh.faces)}")


if __name__ == "__main__":
    main()
