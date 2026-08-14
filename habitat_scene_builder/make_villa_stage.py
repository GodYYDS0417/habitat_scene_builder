#!/usr/bin/env python3
"""make_villa_stage.py — 两层别墅 stage 生成器（每层 10x10m = 100 m^2）。

参考真实户型工程数据（安居客/房天下 94㎡ 2室2厅：客厅开间 5.15m、
层高 2.9m、楼梯踏步高 150mm 宽 280mm）参数化生成：

  F1: 客厅(南) + 餐厅 + 厨房 + 玄关 + 客卫 + 书房 + 楼梯 + 电梯井
  F2: 主卧 + 次卧 + 书房 + 茶室 + 楼梯间 + 电梯井
  楼梯: 直跑 20 级(踏高 0.145m / 踏面 0.28m)，navmesh 可爬（human/spot）
  电梯: 1.4x1.4m 井道 + 独立轿厢 GLB(elevator_car.glb)，脚本升降

用法:
    python make_villa_stage.py --out ../demo_assets/stages/villa_2f.glb \
        --car-out ../demo_assets/objects/elevator_car.glb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# 布局常量（make_villa_plan.py 也引用同一组数值，改动时保持同步）
# ---------------------------------------------------------------------------
W, D = 10.0, 10.0          # 室内净宽/净深 (m)
EXT_T = 0.15               # 外墙厚
INT_T = 0.10               # 内墙厚
FLOOR_H = 2.9              # 层高（真实住宅标准层 2.9m）
SLAB_T = 0.12              # 楼板厚
STAIR = dict(x0=3.8, x1=4.9, z0=-2.6, rise=0.145, run=0.28, n=19)
SHAFT = dict(x0=3.2, x1=4.8, z0=-4.9, z1=-3.3, door_x0=3.6, door_x1=4.4)

WALL_C = (224, 218, 206, 255)     # 暖白内墙
EXT_C = (208, 202, 192, 255)      # 外墙
FLOOR_C = (146, 116, 88, 255)     # 木地板
SLAB_C = (210, 206, 198, 255)     # 楼板/吊顶
STAIR_C = (120, 96, 72, 255)      # 楼梯木
SHAFT_C = (188, 190, 194, 255)    # 电梯井
RAIL_C = (90, 88, 86, 255)        # 栏杆
CAR_C = (170, 176, 184, 255)      # 轿厢


def _box(size, center, color) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    m.visual = trimesh.visual.ColorVisuals(
        m, face_colors=np.tile(np.array(color, dtype=np.uint8), (len(m.faces), 1)))
    return m


def _wall(axis: str, at: float, lo: float, hi: float, y0: float, y1: float,
          t: float, color, openings=()) -> list[trimesh.Trimesh]:
    """沿轴的墙。openings: (pos0, pos1, oy0, oy1) — pos 沿墙方向坐标。"""
    parts: list[trimesh.Trimesh] = []
    segs: list[tuple[float, float]] = []
    cur = lo
    for (p0, p1, oy0, oy1) in sorted(openings):
        if p0 > cur:
            segs.append((cur, p0))
        cur = max(cur, p1)
        # 门楣 / 窗台以下
        if oy1 < y1:
            _add(parts, axis, at, p0, p1, oy1, y1, t, color)
        if oy0 > y0:
            _add(parts, axis, at, p0, p1, y0, oy0, t, color)
    if cur < hi:
        segs.append((cur, hi))
    for s0, s1 in segs:
        _add(parts, axis, at, s0, s1, y0, y1, t, color)
    return parts


def _add(parts, axis, at, p0, p1, y0, y1, t, color):
    if axis == "x":   # 墙沿 z 方向（固定在 x=at）
        size = (t, y1 - y0, p1 - p0)
        center = (at, (y0 + y1) / 2, (p0 + p1) / 2)
    else:             # 墙沿 x 方向（固定在 z=at）
        size = (p1 - p0, y1 - y0, t)
        center = ((p0 + p1) / 2, (y0 + y1) / 2, at)
    parts.append(_box(size, center, color))


def build_villa() -> trimesh.Trimesh:
    hw, hd = W / 2, D / 2          # 5.0, 5.0
    H = FLOOR_H                    # 2.9
    parts: list[trimesh.Trimesh] = []

    # ---- F1 地板（整片，含电梯井下方）----
    parts.append(_box((W + 2 * EXT_T, 0.10, D + 2 * EXT_T),
                      (0, -0.05, 0), FLOOR_C))

    # ---- 外墙 F1 (y 0..H) ----
    ez = [("z", -hd - EXT_T / 2, -hw - EXT_T, hw + EXT_T),   # 北墙 z=-5
          ("z", hd + EXT_T / 2, -hw - EXT_T, hw + EXT_T)]    # 南墙 z=+5
    # 北墙：入户门 x[-0.9,0] h2.1 + 厨房窗 x[-4.3,-3.1] sill0.9 高1.2
    parts += _wall("z", -hd - EXT_T / 2, -hw - EXT_T, hw + EXT_T, 0, H, EXT_T, EXT_C,
                   openings=[(-0.9, 0.0, 0.0, 2.1), (-4.3, -3.1, 0.9, 2.1)])
    # 南墙：客厅大窗 x[-4.0,-1.0] sill0.5 高1.8
    parts += _wall("z", hd + EXT_T / 2, -hw - EXT_T, hw + EXT_T, 0, H, EXT_T, EXT_C,
                   openings=[(-4.0, -1.0, 0.5, 2.3)])
    # 西墙：客厅窗 z[1.5,3.5] sill0.6 高1.6
    parts += _wall("x", -hw - EXT_T / 2, -hd - EXT_T, hd + EXT_T, 0, H, EXT_T, EXT_C,
                   openings=[(1.5, 3.5, 0.6, 2.2)])
    # 东墙：卫生间高窗 z[-4.5,-3.7] sill1.5 高0.6
    parts += _wall("x", hw + EXT_T / 2, -hd - EXT_T, hd + EXT_T, 0, H, EXT_T, EXT_C,
                   openings=[(-4.5, -3.7, 1.5, 2.1)])

    # ---- F1 内隔墙 ----
    # 厨房西墙 x=-2.0 (z -5..-2.4)，门 z[-3.9,-3.0]
    parts += _wall("x", -2.0, -hd, -2.4, 0, H, INT_T, WALL_C,
                   openings=[(-3.9, -3.0, 0.0, 2.1)])
    # 厨房北墙 z=-2.4 (x -5..-2.0)
    parts += _wall("z", -2.4, -hd, -2.0, 0, H, INT_T, WALL_C)
    # 玄关/卫生间隔墙 x=1.2 (z -5..-2.6)
    parts += _wall("x", 1.2, -hd, -2.6, 0, H, INT_T, WALL_C)
    # 玄关/走廊门洞墙 x=1.2 (z -2.6..-0.6)，门 z[-1.9,-1.0]
    parts += _wall("x", 1.2, -2.6, -0.6, 0, H, INT_T, WALL_C,
                   openings=[(-1.9, -1.0, 0.0, 2.1)])
    # 客厅/书房隔墙 x=1.2 (z -0.6..5)，门 z[3.9,4.8]（南端——真实户型门开墙端，
    # 电视墙中段留给 2.4m 电视柜；门居中会被电视柜堵死，实测 navmesh 断连）
    # z[0.2,1.1] 段必须留空：段 2 楼梯实体从 F1 地面升起穿过该面墙，
    # 若墙到顶(y=2.9)，墙顶面悬在坡道上方仅 0.55m —— navmesh 净高过滤
    # 会把坡道拦腰截断（实测断带 x∈[0.85,1.55]）。真实户型中这段是
    # 梯下开放空间，本就没有到顶的墙。
    parts += _wall("x", 1.2, -0.6, 0.2, 0, H, INT_T, WALL_C)
    parts += _wall("x", 1.2, 1.1, hd, 0, H, INT_T, WALL_C,
                   openings=[(3.9, 4.8, 0.0, 2.1)])
    # 卫生间北墙 z=-2.6 (x 1.2..3.0)，门 x[1.7,2.6]
    parts += _wall("z", -2.6, 1.2, 3.0, 0, H, INT_T, WALL_C,
                   openings=[(1.7, 2.6, 0.0, 2.1)])

    # ---- 楼梯（直跑 19 踏步 + 20 级上升，踏高 0.145 踏面 0.28）----
    # 中段 90° 转角平台（L 形楼梯）：后 9 级沿 -x 折返，
    # 顶步直接落在二层走廊（z=-0.1 洞口南缘），动线和真实住宅一致。
    # 平台宽 = 两梯段宽之和（2.2m，真实住宅做法）：必须与段 1 顶步共享整条边，
    # 只碰一个角物理和 navmesh 都连不上（实测踩空/断岛）。
    sx0, sx1 = STAIR["x0"], STAIR["x1"]
    plat_i = 9                                   # 前 10 级沿 +z
    plat_z0 = STAIR["z0"] + (plat_i + 1) * STAIR["run"]   # 0.2
    plat_top = (plat_i + 1) * STAIR["rise"]               # 1.45
    plat_d = 0.90                                # 平台进深
    plat_x1 = sx1                                # 4.9（盖住段 1 顶步出口）
    plat_x0 = 2.7                                # 段 2 从这里沿 -x 折返
    for i in range(STAIR["n"]):
        top = (i + 1) * STAIR["rise"]
        if i <= plat_i:                          # 段 1：沿 +z
            z0 = STAIR["z0"] + i * STAIR["run"]
            parts.append(_box((sx1 - sx0, top, STAIR["run"]),
                              ((sx0 + sx1) / 2, top / 2, z0 + STAIR["run"] / 2), STAIR_C))
        else:                                    # 段 2：沿 -x
            x1 = plat_x0 - (i - plat_i - 1) * STAIR["run"]
            parts.append(_box((STAIR["run"], top, plat_d),
                              (x1 - STAIR["run"] / 2, top / 2, plat_z0 + plat_d / 2),
                              STAIR_C))
    parts.append(_box((plat_x1 - plat_x0, plat_top, plat_d),
                      ((plat_x0 + plat_x1) / 2, plat_top / 2, plat_z0 + plat_d / 2),
                      STAIR_C))
    stair_top_x = plat_x0 - (STAIR["n"] - plat_i - 1) * STAIR["run"]  # 0.18

    # ---- F2 楼板 (y H-0.12..H)：楼梯间洞口 x[0.18,5.15] z[-2.7,1.1] + 电梯井口 ----
    # 按 z 条带分解（洞口 A=电梯井 x3.2..4.8/z-4.9..-3.3，洞口 B=楼梯间 x0.18..5.15/z-2.7..1.1）
    slab_boxes = [
        (-hw - EXT_T, hw + EXT_T, -hd - EXT_T, SHAFT["z0"]),   # z -5.15..-4.9 整片
        (-hw - EXT_T, SHAFT["x0"], SHAFT["z0"], SHAFT["z1"]),  # 井西
        (SHAFT["x1"], hw + EXT_T, SHAFT["z0"], SHAFT["z1"]),   # 井东
        (-hw - EXT_T, hw + EXT_T, SHAFT["z1"], -2.70),         # 井与楼梯间之间整片
        (-hw - EXT_T, 0.18, -2.70, 1.10),                      # 楼梯间西条（含顶步落点）
        (-hw - EXT_T, hw + EXT_T, 1.10, hd + EXT_T),           # 楼梯间南整片
    ]
    for (x0, x1, z0, z1) in slab_boxes:
        parts.append(_box((x1 - x0, SLAB_T, z1 - z0),
                          ((x0 + x1) / 2, H - SLAB_T / 2, (z0 + z1) / 2), SLAB_C))

    # ---- F2 外墙 (y H..2H) ----
    y2a, y2b = H, 2 * H
    parts += _wall("z", -hd - EXT_T / 2, -hw - EXT_T, hw + EXT_T, y2a, y2b, EXT_T, EXT_C,
                   openings=[(-4.3, -2.9, y2a + 0.6, y2a + 2.2),   # 次卧窗
                             (-0.9, 0.5, y2a + 0.6, y2a + 2.2)])   # 书房窗
    parts += _wall("z", hd + EXT_T / 2, -hw - EXT_T, hw + EXT_T, y2a, y2b, EXT_T, EXT_C,
                   openings=[(-3.9, -1.4, y2a + 0.5, y2a + 2.3)])  # 主卧窗
    parts += _wall("x", -hw - EXT_T / 2, -hd - EXT_T, hd + EXT_T, y2a, y2b, EXT_T, EXT_C,
                   openings=[(2.0, 4.0, y2a + 0.6, y2a + 2.2)])    # 主卧西窗
    parts += _wall("x", hw + EXT_T / 2, -hd - EXT_T, hd + EXT_T, y2a, y2b, EXT_T, EXT_C,
                   openings=[(3.2, 4.6, y2a + 0.6, y2a + 2.2)])    # 茶室东窗

    # ---- F2 内隔墙 ----
    # L 形走廊：横段 z[-0.6,0.2] + 竖段 x[-0.8,0.18] z[0.2,5] 直连楼梯顶步出口，
    # 各房间门都朝走廊开——上楼不必穿主卧（旧版主卧南墙 z=0.2 横贯 x-5..1.2，
    # 楼梯出来只能穿主卧北缘条带进走廊，衣柜一摆就全层断连，且户型不真实）。
    # 主卧北墙 z=0.2 (x -5..-0.8)，无门
    parts += _wall("z", 0.2, -hw, -0.8, y2a, y2b, INT_T, WALL_C)
    # 主卧东墙 x=-0.8 (z 0.2..5)，门 z[1.5,2.4] 朝走廊竖段
    parts += _wall("x", -0.8, 0.2, hd, y2a, y2b, INT_T, WALL_C,
                   openings=[(1.5, 2.4, y2a, y2a + 2.1)])
    # 次卧/书房北墙 z=-0.6 (x -5..1.2)：次卧门 x[-4.2,-3.3]，书房门 x[-0.9,0.0]
    parts += _wall("z", -0.6, -hw, 1.2, y2a, y2b, INT_T, WALL_C,
                   openings=[(-4.2, -3.3, y2a, y2a + 2.1),
                             (-0.9, 0.0, y2a, y2a + 2.1)])
    # 次卧/书房隔墙 x=-2.0 (z -5..-0.6)
    parts += _wall("x", -2.0, -hd, -0.6, y2a, y2b, INT_T, WALL_C)
    # 茶室/走廊隔墙 x=1.2 (z 1.1..5)，门 z[2.5,3.4]
    # 注意：南端必须从楼梯间洞口南缘 z=1.1 起——z<1.1 段下方没有楼板
    # （悬空墙），且墙底(y=2.9)压在段 2 坡道上方净高仅 0.53m，
    # navmesh 侵蚀后坡道被拦腰切断（实测断带 x∈[0.85,1.55]）。
    parts += _wall("x", 1.2, 1.1, hd, y2a, y2b, INT_T, WALL_C,
                   openings=[(2.5, 3.4, y2a, y2a + 2.1)])

    # ---- 楼梯间洞口 F2 栏杆（只围空洞边缘，避开顶步出口 x[-0.1,0.2] z[0.2,1.1]）----
    parts.append(_box((0.05, 0.95, 0.15 - (-2.70)),                       # 西缘南段
                      (0.075, H + 0.475, (-2.70 + 0.15) / 2), RAIL_C))
    parts.append(_box((hw + EXT_T - 0.075, 0.95, 0.05),                   # 北缘整段
                      ((0.075 + hw + EXT_T) / 2, H + 0.475, -2.725), RAIL_C))
    parts.append(_box((hw + EXT_T - 0.20, 0.95, 0.05),                    # 南缘（留顶步出口）
                      ((0.20 + hw + EXT_T) / 2, H + 0.475, 1.025), RAIL_C))

    # ---- 电梯井道（y 0..2H+? 贯通两层，南北各开门）----
    sx_lo, sx_hi = SHAFT["x0"], SHAFT["x1"]
    sz_lo, sz_hi = SHAFT["z0"], SHAFT["z1"]
    shaft_top = 2 * H - 0.3     # 井道顶 5.5
    # 西墙 / 东墙 / 南墙(底)
    parts += _wall("x", sx_lo, sz_lo, sz_hi, 0, shaft_top, INT_T, SHAFT_C)
    parts += _wall("x", sx_hi, sz_lo, sz_hi, 0, shaft_top, INT_T, SHAFT_C)
    parts += _wall("z", sz_lo, sx_lo, sx_hi, 0, shaft_top, INT_T, SHAFT_C)
    # 北墙（门脸）：F1 门洞 y0..2.1，F2 门洞 y2.9..5.0
    dx0, dx1 = SHAFT["door_x0"], SHAFT["door_x1"]
    parts += _wall("z", sz_hi, sx_lo, sx_hi, 0, shaft_top, INT_T, SHAFT_C,
                   openings=[(dx0, dx1, 0.0, 2.1),
                             (dx0, dx1, H, H + 2.1)])
    # 井道顶板
    parts.append(_box((sx_hi - sx_lo + INT_T, 0.10, sz_hi - sz_lo + INT_T),
                      ((sx_lo + sx_hi) / 2, shaft_top + 0.05, (sz_lo + sz_hi) / 2), SHAFT_C))

    # ---- 屋顶 ----
    parts.append(_box((W + 2 * EXT_T, 0.12, D + 2 * EXT_T),
                      (0, 2 * H + 0.06, 0), SLAB_C))

    return trimesh.util.concatenate(parts)


def build_navramps() -> trimesh.Trimesh:
    """楼梯坡道辅助网格：只参与 navmesh 烘焙，不渲染、不进物理、不进数据集。

    Recast 按 agent 半径(0.25~0.30m)侵蚀可行走面，0.28m 的细踏步被侵蚀后
    无处可走（实测段内/平台/层间多处断岛）。铺上 27~30° 坡道面后楼梯被
    当成斜坡（< 各具身 max_slope），层间 navmesh 连通。游戏行业标准做法。

    坡面恰好穿过每个踏步前缘顶线（踏步不戳出坡面），两端延长到与
    走廊地面/转角平台顶面/二层楼板相交——衔接处零高差。若端点搭在
    高一级（0.145m）的位置，walkableClimb 体素量化后（spot 仅 0.1m）
    爬不上该接缝，实测断岛。
    """
    sx0, sx1 = STAIR["x0"], STAIR["x1"]
    slope = STAIR["rise"] / STAIR["run"]               # 0.5179（踏高/踏面）
    ramps: list[trimesh.Trimesh] = []

    # 段 1：沿 +z。过踏步前缘顶线 (z=-2.6, y=0.145) -> (z=-0.08, y=1.45)
    # （第 10 级踏步顶 = 平台顶 1.45，齐平衔接）。底端延长到 z=-3.3，
    # 坡面在 z=-2.88 处穿出走廊地面（y=0），与地面相交而不是搭台阶。
    z_a, z_b = -3.3, -0.08
    y_a = STAIR["rise"] + (z_a - STAIR["z0"]) * slope          # -0.2175（埋入地下）
    y_b = 10 * STAIR["rise"]                                    # 1.45 = 平台顶
    r1 = trimesh.creation.box(extents=(sx1 - sx0, 0.05, float(np.hypot(z_b - z_a, y_b - y_a))))
    r1.apply_transform(trimesh.transformations.rotation_matrix(
        -float(np.arctan2(y_b - y_a, z_b - z_a)), [1, 0, 0]))
    r1.apply_translation(((sx0 + sx1) / 2, (y_a + y_b) / 2, (z_a + z_b) / 2))
    ramps.append(r1)

    # 段 2：沿 -x。过踏步前缘顶线 (x=2.7, y=1.595) -> (x=0.18, y=2.90=F2 楼板，
    # 齐平衔接)。底端延长到 x=3.3：坡面在 x=2.98 处与平台顶(1.45)相交，
    # x>2.98 段埋入平台实心体内（不可见），衔接零高差。
    x_a, x_b = 3.3, 0.18
    y2_a = 11 * STAIR["rise"] - (x_a - 2.7) * slope             # 1.2843（埋入平台）
    y2_b = 20 * STAIR["rise"]                                   # 2.90 = F2 楼板顶
    r2 = trimesh.creation.box(extents=(float(np.hypot(x_a - x_b, y2_b - y2_a)), 0.05, 0.90))
    r2.apply_transform(trimesh.transformations.rotation_matrix(
        -float(np.arctan2(y2_b - y2_a, x_a - x_b)), [0, 0, 1]))
    r2.apply_translation(((x_a + x_b) / 2, (y2_a + y2_b) / 2, 0.65))
    ramps.append(r2)

    # 段 1 底端水平衔接板：ramp 严格过踏步前缘顶线，其 y=0 穿出点(z=-2.88)
    # 必然在踏步北侧，而楼梯入口(z=-2.6)处 ramp 西侧面(x=3.8, 高0.145) erosion
    # 后与地面形成 0.1m 无 mesh 缝、切成两岛。加一块与 ramp 在入口处同高
    # (y=0.145, 即第一级踏步顶) 的水平板，把 ramp 西缘与地面实打实连通。
    # 0.145m 台缘 < human(0.20)/spot(0.15) 的 max_climb，可爬；fetch(0.05)
    # 本就不走楼梯（走电梯），不受影响。
    apron = trimesh.creation.box(extents=(4.6 - 3.4, 0.05, -2.55 - (-3.10)))
    apron.apply_translation(((3.4 + 4.6) / 2, 0.145 - 0.025, (-3.10 + -2.55) / 2))
    ramps.append(apron)
    return trimesh.util.concatenate(ramps)


def build_elevator_car() -> trimesh.Trimesh:
    """轿厢：1.3x1.3 平台 + 三面围板，门朝 +z。原点在底板中心底面。"""
    floor = _box((1.30, 0.08, 1.30), (0, 0.04, 0), CAR_C)
    back = _box((1.30, 2.20, 0.04), (0, 1.18, -0.63), CAR_C)
    left = _box((0.04, 2.20, 1.30), (-0.63, 1.18, 0), CAR_C)
    right = _box((0.04, 2.20, 1.30), (0.63, 1.18, 0), CAR_C)
    ceiling = _box((1.30, 0.04, 1.30), (0, 2.30, 0), CAR_C)
    return trimesh.util.concatenate([floor, back, left, right, ceiling])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="stage GLB 输出")
    ap.add_argument("--car-out", type=Path, default=None,
                    help="轿厢 GLB 输出（放进 --objects 目录即注册为可摆物体）")
    ap.add_argument("--nav-out", type=Path, default=None,
                    help="楼梯坡道 GLB 输出（命名 <stage>_navramps.glb 时"
                         " navmesh 烘焙自动使用，不渲染不进物理）")
    args = ap.parse_args()

    mesh = build_villa()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out)
    print(f"{args.out}  两层 {W:.0f}x{D:.0f}m = {W*D:.0f} m^2/层, "
          f"层高 {FLOOR_H} m, faces = {len(mesh.faces)}")

    if args.nav_out is not None:
        ramps = build_navramps()
        args.nav_out.parent.mkdir(parents=True, exist_ok=True)
        ramps.export(args.nav_out)
        print(f"{args.nav_out}  楼梯坡道(仅 navmesh 烘焙), faces = {len(ramps.faces)}")

    if args.car_out is not None:
        car = build_elevator_car()
        args.car_out.parent.mkdir(parents=True, exist_ok=True)
        car.export(args.car_out)
        print(f"{args.car_out}  轿厢 1.3x1.3 m, faces = {len(car.faces)}")


if __name__ == "__main__":
    main()
