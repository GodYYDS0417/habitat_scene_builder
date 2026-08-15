#!/usr/bin/env python3
"""make_demo_objects.py — 生成一套带正确类名和尺寸的演示用家具/小物体 GLB。

不下载任何外部资产，全部用 trimesh 参数化生成，尺寸按真实物体设定，
类名与 habitat_scene_builder_v2 的 SEMANTIC_RULES 对齐，方便语义放置。

用法:
    python make_demo_objects.py --out assets/objects
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _box(size, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return m


def _cyl(radius, height, center=(0, 0, 0), sections=24):
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.apply_translation(center)
    return m


def _sphere(radius, center=(0, 0, 0)):
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.apply_translation(center)
    return m


def _concat(parts):
    """多节点 Scene：每个零件一个节点，不要 concatenate 成单网格。

    实测（probe_shelf2.py）：单网格 + join_collision_meshes=true 的拼接凹面
    碰撞会在内腔产生幻影接触（书在书架搁板间被弹飞穿透）；多节点 GLB +
    join_collision_meshes=false 让每个零件得到独立凸包碰撞——我们的零件
    全是盒/柱/球，凸包即精确形状，box-box 是 Bullet 最稳的路径。
    """
    return trimesh.Scene({f"part{i}": p for i, p in enumerate(parts)})


def _zup_to_yup(obj):
    """trimesh 圆柱/圆锥默认沿 Z 轴建，转到 Y-up 并把底面放到 y=0。

    兼容 Trimesh 和 Scene（_concat 现在返回 Scene）。
    """
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    obj.apply_transform(rot)
    drop = trimesh.transformations.translation_matrix((0, -float(obj.bounds[0][1]), 0))
    obj.apply_transform(drop)
    return obj


# 所有模型原点放在物体底部中心 (y=0 是支撑面)，Y-up，单位米。
# 尺寸参考真实户型工程数据（安居客 94㎡ 2室2厅：沙发 3x0.8、电视柜 3x0.5、
# 茶几 2x0.6、餐桌 2x0.8、床 2x2m、床头柜 0.5x0.5、衣柜 3x0.6）。
def _make_dummy():
    """火灾救援假人（躺姿人形）：躯干+头+四肢，体长沿 ±z，背部贴床 y=0。"""
    parts = [
        _box((0.34, 0.16, 0.72), (0, 0.08, 0.10)),      # 躯干
        _sphere(0.11, (0, 0.10, 0.55)),                  # 头
        _cyl(0.07, 0.62, (0.0, 0.07, -0.45)),            # 双腿（合并）
        _cyl(0.045, 0.55, (0.22, 0.10, 0.18)),           # 右臂
        _cyl(0.045, 0.55, (-0.22, 0.10, 0.18)),          # 左臂
    ]
    return _concat(parts)


def _make_elevator_panel():
    """电梯按钮面板：面板 + 上下两个凸出按钮（装门旁墙上，中心约1.2m高）。"""
    return _concat([
        _box((0.14, 0.22, 0.04), (0, 0.11, 0)),       # 面板
        _cyl(0.026, 0.02, (0, 0.16, 0.028)),          # 上按钮（选层）
        _cyl(0.026, 0.02, (0, 0.07, 0.028)),          # 下按钮（呼叫）
    ])


BUILDERS = {
    # ---- 家具（STATIC）----
    "table": lambda: _concat([          # 餐桌 1.6x0.85（真实容量 2x0.8）
        _box((1.60, 0.04, 0.85), (0, 0.73, 0)),
        *[_box((0.06, 0.73, 0.06), (x, 0.365, z))
          for x in (-0.72, 0.72) for z in (-0.36, 0.36)],
    ]),
    "desk": lambda: _concat([
        _box((1.20, 0.04, 0.60), (0, 0.73, 0)),
        *[_box((0.05, 0.73, 0.05), (x, 0.365, z))
          for x in (-0.55, 0.55) for z in (-0.25, 0.25)],
    ]),
    "chair": lambda: _concat([
        _box((0.42, 0.04, 0.42), (0, 0.45, 0)),
        _box((0.42, 0.50, 0.04), (0, 0.72, -0.19)),
        *[_box((0.04, 0.45, 0.04), (x, 0.225, z))
          for x in (-0.18, 0.18) for z in (-0.18, 0.18)],
    ]),
    "sofa": lambda: _concat([           # 三人沙发 3.0x0.85（真实 3x0.8）
        _box((3.00, 0.42, 0.85), (0, 0.21, 0)),
        _box((3.00, 0.50, 0.20), (0, 0.67, -0.325)),
        _box((0.20, 0.22, 0.85), (-1.40, 0.53, 0)),
        _box((0.20, 0.22, 0.85), (1.40, 0.53, 0)),
    ]),
    "coffee_table": lambda: _concat([   # 茶几 1.4x0.7（真实容量 2x0.6）
        _box((1.40, 0.04, 0.70), (0, 0.40, 0)),
        *[_box((0.05, 0.40, 0.05), (x, 0.20, z))
          for x in (-0.62, 0.62) for z in (-0.28, 0.28)],
    ]),
    "tv_stand": lambda: _box((2.40, 0.45, 0.50), (0, 0.225, 0)),   # 电视柜
    "tv": lambda: _concat([             # 65 寸电视 1.45x0.82 + 底座
        _box((0.60, 0.03, 0.30), (0, 0.015, 0)),
        _box((0.08, 0.10, 0.06), (0, 0.08, 0)),
        _box((1.45, 0.82, 0.05), (0, 0.13 + 0.41, 0)),
    ]),
    "bed": lambda: _concat([            # 双人床 2.0x2.0（真实 2x2m）
        _box((2.00, 0.30, 2.00), (0, 0.15, 0)),
        _box((2.00, 0.25, 1.90), (0, 0.425, 0.03)),     # 床垫
        _box((2.00, 0.90, 0.10), (0, 0.75, -1.00)),     # 床头板
    ]),
    "cabinet": lambda: _box((1.80, 2.00, 0.60), (0, 1.00, 0)),    # 衣柜
    "nightstand": lambda: _box((0.50, 0.50, 0.45), (0, 0.25, 0)),  # 床头柜
    "counter": lambda: _concat([        # 厨房操作台 2.4x0.6
        _box((2.40, 0.82, 0.60), (0, 0.41, 0)),
        _box((2.44, 0.04, 0.64), (0, 0.84, 0)),         # 台面
    ]),
    "shelf": lambda: _concat([
        _box((0.90, 0.03, 0.30), (0, y, 0)) for y in (0.40, 0.80, 1.20, 1.60)
    ] + [_box((0.03, 1.60, 0.30), (x, 0.80, 0)) for x in (-0.435, 0.435)]),
    "lamp": lambda: _zup_to_yup(_concat([
        _cyl(0.10, 0.02, (0, 0, 0.01)),
        _cyl(0.015, 0.45, (0, 0, 0.245)),
        trimesh.creation.cone(radius=0.14, height=0.18, sections=24).apply_translation(
            (0, 0, 0.56)),
    ])),
    # ---- 小物体（DYNAMIC 可抓取）----
    "cup": lambda: _zup_to_yup(_cyl(0.04, 0.10, (0, 0, 0.05))),
    "mug": lambda: _zup_to_yup(_cyl(0.045, 0.11, (0, 0, 0.055))),
    "plate": lambda: _zup_to_yup(_cyl(0.11, 0.02, (0, 0, 0.01))),
    "bowl": lambda: _zup_to_yup(
        trimesh.creation.cone(radius=0.09, height=0.07, sections=24).apply_translation((0, 0, 0.035))),
    "bottle": lambda: _zup_to_yup(_concat([
        _cyl(0.040, 0.22, (0, 0, 0.11)),
        _cyl(0.014, 0.06, (0, 0, 0.25)),
    ])),
    "book": lambda: _box((0.20, 0.035, 0.145), (0, 0.0175, 0)),
    "apple": lambda: _sphere(0.042, (0, 0.042, 0)),
    "orange": lambda: _sphere(0.045, (0, 0.045, 0)),
    "box": lambda: _box((0.18, 0.12, 0.12), (0, 0.06, 0)),
    "keyboard": lambda: _box((0.36, 0.02, 0.13), (0, 0.01, 0)),
    "dummy": _make_dummy,
    # 电梯部件：门扇（演示时 KINEMATIC 滑动）；按钮面板（STATIC 装墙）
    "elevator_door": lambda: _box((0.40, 2.10, 0.05), (0, 1.05, 0)),
    "elevator_panel": _make_elevator_panel,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="只生成这些类名（默认全部）")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(BUILDERS)
    # 个别物体用深色以区分场景（电梯门扇 vs 浅色轿厢/墙）
    DARK_COLORS = {"elevator_door": (96, 99, 104, 255)}
    for name in names:
        if name not in BUILDERS:
            raise SystemExit(f"未知物体: {name}，可选: {sorted(BUILDERS)}")
        obj = BUILDERS[name]()
        geoms = list(obj.geometry.values()) if isinstance(obj, trimesh.Scene) else [obj]
        # 统一上色，避免默认材质太白
        color = DARK_COLORS.get(name, (180, 180, 180, 255))
        for g in geoms:
            g.visual = trimesh.visual.ColorVisuals(
                g, face_colors=np.tile(color, (len(g.faces), 1)))
        out = args.out / f"{name}.glb"
        obj.export(out)
        ext = obj.extents
        nfaces = sum(len(g.faces) for g in geoms)
        print(f"{out.name:<14} size = {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m, "
              f"parts = {len(geoms)}, faces = {nfaces}")
    print(f"\n完成 -> {args.out}（{len(names)} 个物体）")


if __name__ == "__main__":
    main()
