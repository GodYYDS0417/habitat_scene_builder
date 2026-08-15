#!/usr/bin/env python3
"""elevator_system.py — 可交互电梯系统：按钮判定 + 门开关动画 + 轿厢选层升降。

异构约束核心：按钮面板装在离地 ~1.16m 的墙上。
  - Go2 机器狗（可达 ~0.5m）按不到 → 只能爬楼梯
  - G1 人形机器人（可达 ~1.6m）/ Fetch 机器车（臂端 ~1.2m）能按到 → 坐电梯

电梯交互是脚本化传送：机器人用 navmesh 走到门口前的走廊，按按钮后由
门开关 + 轿厢升降动画完成跨层（井道不参与 navmesh，避免机器人走进空井道）。
"""
from __future__ import annotations

import json as _json
import math
import tempfile
from pathlib import Path

import numpy as np

# ---- 电梯井几何（与 make_villa_stage.py 的 SHAFT 对齐）----
DOOR_X = (3.6, 4.4)        # 门洞 x 范围
# 门面（井道北墙中心平面 z=-3.3，墙厚 0.10）。
# 门扇贴走廊侧墙面安装（开门后滑到门洞两侧、贴在墙外表面可见）；
# 若放在墙中心平面，开门滑到实心墙段会被 0.10m 厚墙体完全包裹而不可见。
DOOR_Z = -3.235
F1_Y, F2_Y = 0.0, 2.9      # 楼层地面高
CABIN_XZ = (4.0, -4.10)    # 轿厢中心 xz
FLOOR_Y = {1: F1_Y, 2: F2_Y}

# 每层交互关键点：门口走廊点 / 按钮面板前 / 轿厢内
DOOR_OUT = {1: np.array([2.3, 0.10, -1.6]), 2: np.array([2.3, 2.95, -1.6])}
PANEL_FRONT = {1: np.array([3.42, 0.10, -2.85]), 2: np.array([3.42, 2.95, -2.85])}
CABIN_IN = {1: np.array([4.0, 0.10, -4.0]), 2: np.array([4.0, 2.95, -4.0])}
# 门洞正中（进出轿厢必须经此折点走直角折线，否则斜线插值会穿门框侧墙）
DOOR_MID = {1: np.array([4.0, 0.10, -3.00]), 2: np.array([4.0, 2.95, -3.00])}

# 门扇：每扇宽 0.4m。关门时左扇中心 x=3.8、右扇 x=4.2；开门各向外滑 0.42m
DOOR_CLOSE_X = {"l": 3.8, "r": 4.2}
DOOR_SLIDE = 0.42
DOOR_W = 0.40

# 按钮面板（门西侧墙上，凸出墙面安装便于看清）
PANEL_XZ = (3.42, -3.245)
PANEL_BASE_Y = 1.0                       # 面板底离地
BUTTON_REACH_Y = PANEL_BASE_Y + 0.16     # 上按钮中心 ≈1.16m

# 各机器人可达高度（按按钮能力）
ROBOT_REACH = {"g1": 1.60, "go2": 0.50, "fetch": 1.20}
ROBOT_NAME = {"g1": "人形机器人G1", "go2": "机器狗Go2", "fetch": "机器车Fetch"}

PRESS_DIST = 0.70    # 按按钮的最大水平距离


def _spawn_rigid(sim, hsim, mn, glb, bottom_center, motion, mass=5.0):
    """加载 GLB 为 rigid object，把 bbox 底面中心放到 bottom_center（处理原点重置）。"""
    tmp = Path(tempfile.mkdtemp(prefix="elev_"))
    cfg = tmp / "o.object_config.json"
    cfg.write_text(_json.dumps({
        "render_asset": str(glb), "collision_asset": str(glb),
        "up": [0.0, 1.0, 0.0], "front": [0.0, 0.0, -1.0],
        "units_to_meters": 1.0, "mass": mass,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": False, "is_collidable": True}))
    sim.get_object_template_manager().load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(str(cfg))
    node = obj.root_scene_node
    bb = hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
    dx = bottom_center[0] - (bb.min.x + bb.max.x) / 2
    dy = bottom_center[1] - bb.min.y
    dz = bottom_center[2] - (bb.min.z + bb.max.z) / 2
    t = obj.translation
    obj.translation = mn.Vector3(t.x + dx, t.y + dy, t.z + dz)
    obj.motion_type = motion
    return obj


class ElevatorSystem:
    """管理电梯轿厢、两层门扇、按钮面板，提供交互接口。"""

    def __init__(self, sim, hsim, mn, objects_dir):
        self.sim, self.hsim, self.mn = sim, hsim, mn
        STATIC = hsim.physics.MotionType.STATIC
        KIN = hsim.physics.MotionType.KINEMATIC
        od = Path(objects_dir)
        # 门扇：F1/F2 各左右两扇，KINEMATIC 可滑动
        self.doors = {}
        self.door_base = {}    # 关门时各门扇底面中心
        for fl, fy in ((1, F1_Y), (2, F2_Y)):
            for side in ("l", "r"):
                bc = (DOOR_CLOSE_X[side], fy, DOOR_Z)
                obj = _spawn_rigid(sim, hsim, mn, od / "elevator_door.glb", bc, KIN, mass=30.0)
                self.doors[(fl, side)] = obj
                self.door_base[(fl, side)] = np.array(
                    [obj.translation.x, obj.translation.y, obj.translation.z], dtype=np.float64)
        # 按钮面板：F1/F2 各一，STATIC
        self.panels = {}
        for fl, fy in ((1, F1_Y), (2, F2_Y)):
            self.panels[fl] = _spawn_rigid(
                sim, hsim, mn, od / "elevator_panel.glb",
                (PANEL_XZ[0], fy + PANEL_BASE_Y, PANEL_XZ[1]), STATIC, mass=5.0)
        # 轿厢：数据集中的 elevator_car，改 KINEMATIC
        self.cabin = None
        mgr = sim.get_rigid_object_manager()
        for h in mgr.get_object_handles(""):
            if h.startswith("elevator_car"):
                self.cabin = mgr.get_object_by_handle(h)
                break
        self.cabin_y1 = None
        if self.cabin is not None:
            self.cabin.motion_type = KIN
            self.cabin_y1 = float(self.cabin.translation.y)   # F1 时轿厢底 y
        self.cabin_floor = 1
        self.door_open = {1: False, 2: False}

    # ---- 按钮交互 ----
    def can_reach(self, robot_type) -> bool:
        return ROBOT_REACH.get(robot_type, 1.0) >= BUTTON_REACH_Y

    def try_press(self, robot_type, robot_pos) -> tuple[bool, str]:
        """机器人尝试按按钮。返回 (是否按到, 说明)。"""
        rname = ROBOT_NAME.get(robot_type, robot_type)
        dx = robot_pos[0] - PANEL_XZ[0]
        dz = robot_pos[2] - PANEL_XZ[1]
        dist = math.hypot(dx, dz)
        if dist > PRESS_DIST:
            return False, f"{rname} 离按钮太远 ({dist:.2f}m > {PRESS_DIST}m)"
        if not self.can_reach(robot_type):
            return False, (f"{rname} 够不到按钮（可达 {ROBOT_REACH[robot_type]:.2f}m "
                           f"< 按钮 {BUTTON_REACH_Y:.2f}m）")
        return True, f"{rname} 按到按钮（可达 {ROBOT_REACH[robot_type]:.2f}m ≥ {BUTTON_REACH_Y:.2f}m）"

    # ---- 门开关动画 ----
    def set_doors(self, floor, open_: bool, cb=None, frames=24):
        """门扇滑动开关（smoothstep 缓动）。cb() 每帧回调（渲染用）。"""
        for k in range(frames):
            t = (k + 1) / frames
            if not open_:
                t = 1.0 - t
            tt = t * t * (3 - 2 * t)
            for s_, sign in (("l", -1.0), ("r", 1.0)):
                base = self.door_base[(floor, s_)]
                newx = base[0] + sign * DOOR_SLIDE * tt
                self.doors[(floor, s_)].translation = self.mn.Vector3(
                    newx, base[1], base[2])
            if cb:
                cb()
        self.door_open[floor] = open_

    def _door_y(self, floor):
        return 0.0   # 门扇底面 y 由 door_base 决定（spawn 时已含 fy）

    # ---- 轿厢升降动画 ----
    def move_cabin(self, floor: int, cb=None, frames=90):
        """轿厢从当前楼层升到目标楼层。cb 每帧回调。"""
        if self.cabin is None or self.cabin_y1 is None:
            return
        y0 = float(self.cabin.translation.y)
        y1 = self.cabin_y1 + (FLOOR_Y[floor] - F1_Y)
        for k in range(frames):
            t = (k + 1) / frames
            tt = t * t * (3 - 2 * t)          # smoothstep 缓动
            y = y0 + (y1 - y0) * tt
            self.cabin.translation = self.mn.Vector3(
                CABIN_XZ[0], y, CABIN_XZ[1])
            if cb:
                cb()
        self.cabin_floor = floor

    # ---- 状态查询 ----
    def cabin_floor_y(self) -> float:
        return FLOOR_Y[self.cabin_floor]
