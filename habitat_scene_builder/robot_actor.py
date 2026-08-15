#!/usr/bin/env python3
"""robot_actor.py — 把 URDF 机器人包装成可沿 navmesh 路径移动的演员。

机器人用 KINEMATIC（不受物理、防倒），逐帧 set base translation/rotation。
关节保持加载默认姿态（站立/前伸臂），只整体移动 + 转向。
"""
from __future__ import annotations

import math

import numpy as np

from elevator_system import ROBOT_NAME, ROBOT_REACH  # noqa: E402


class RobotActor:
    """URDF 机器人演员：加载、KINEMATIC 定位、沿 navmesh 路径移动。

    URDF 均为 z-up 编写，而 Habitat 世界为 y-up 且加载器不做轴转换，
    因此 set_pose 需要施加固定校正旋转 R_x(-90°)，并把基座抬升
    BASE_LIFT（让 pos 语义统一为「脚底/落地触点」的楼层高度）。
    """

    # 模型姿态校正（z-up URDF -> y-up 世界），yaw 偏移 +90° 使
    # 连杆 +X（前进方向）在 yaw=0 时对应世界 -Z，与 move_path 的
    # 朝向约定 facing=(-sin yaw, -cos yaw) 对齐。
    _RX_FIX = None  # 延迟到首次使用时构建（需要 mn）
    BASE_LIFT = {"g1": 0.70, "go2": 0.42, "fetch": 0.07}

    def __init__(self, sim, hsim, mn, robot_type: str, urdf_path, pos, yaw: float = 0.0):
        self.sim, self.hsim, self.mn = sim, hsim, mn
        self.type = robot_type
        self.name = ROBOT_NAME.get(robot_type, robot_type)
        self.reach = ROBOT_REACH.get(robot_type, 1.0)
        self.lift = self.BASE_LIFT.get(robot_type, 0.0)
        aom = sim.get_articulated_object_manager()
        self.obj = aom.add_articulated_object_from_urdf(str(urdf_path), fixed_base=False)
        try:
            self.obj.motion_type = hsim.physics.MotionType.KINEMATIC
        except Exception:
            pass
        self._rx_fix = mn.Quaternion.rotation(mn.Rad(-math.pi / 2), mn.Vector3(1.0, 0.0, 0.0))
        self.yaw = float(yaw)
        self.set_pose(pos, yaw)
        # 步态引擎状态
        self._home_pose = np.array(self.obj.joint_positions)   # 加载默认站姿
        self.phase = 0.0          # 步态相位（左腿/FL 相位，右对侧 +π）
        self.gait_arms = True     # 搬运时置 False：手臂扶假人不摆动

    # ---- 步态关节驱动 ----
    # G1: 腿 hip_pitch(0/6) knee(3/9) ankle_pitch(4/10)，臂 shoulder_pitch(13/18)
    # Go2: thigh(1/4/7/10) calf(2/5/8/11)，trot 对角同相 FL+RR / FR+RL
    GAIT_STRIDE = {"g1": 1.0, "go2": 0.5}

    def _apply_gait(self, dist: float):
        """按行走距离推进相位，驱动腿/臂关节做周期性摆动（视觉步态）。"""
        stride = self.GAIT_STRIDE.get(self.type)
        if stride is None or dist <= 0.0:
            return
        self.phase = (self.phase + 2.0 * math.pi * dist / stride) % (2.0 * math.pi)
        s = math.sin(self.phase)
        home = self._home_pose
        p = np.array(self.obj.joint_positions)
        if self.type == "g1":
            hipA, kneeB, kneeC, armA = 0.40, 0.35, 0.15, 0.20
            p[0] = home[0] - hipA * s                     # 左髋 pitch（s>0 前抬）
            p[6] = home[6] + hipA * s                     # 右髋反相
            p[3] = home[3] + kneeC + kneeB * max(0.0, s)  # 左膝摆动相屈膝
            p[9] = home[9] + kneeC + kneeB * max(0.0, -s)
            p[4] = home[4] - ((p[0] - home[0]) + (p[3] - home[3])) * 0.7   # 踝补偿
            p[10] = home[10] - ((p[6] - home[6]) + (p[9] - home[9])) * 0.7
            if self.gait_arms:
                p[13] = home[13] + armA * s               # 左臂与左腿反相
                p[18] = home[18] - armA * s
        elif self.type == "go2":
            thighA, calfB = 0.30, 0.25
            p[1] = home[1] - thighA * s                   # FL thigh
            p[10] = home[10] - thighA * s                 # RR（与 FL 同相）
            p[4] = home[4] + thighA * s                   # FR
            p[7] = home[7] + thighA * s                   # RL（与 FR 同相）
            p[2] = home[2] + calfB * max(0.0, s)          # FL calf
            p[11] = home[11] + calfB * max(0.0, s)
            p[5] = home[5] + calfB * max(0.0, -s)
            p[8] = home[8] + calfB * max(0.0, -s)
        self.obj.joint_positions = p

    # ---- 位姿 ----
    def set_pose(self, pos, yaw):
        """pos 为脚底位置（楼层地面高度），基座自动抬升 self.lift。"""
        self.yaw = float(yaw)
        self.obj.translation = self.mn.Vector3(
            float(pos[0]), float(pos[1]) + self.lift, float(pos[2]))
        q_yaw = self.mn.Quaternion.rotation(
            self.mn.Rad(self.yaw + math.pi / 2), self.mn.Vector3(0.0, 1.0, 0.0))
        self.obj.rotation = q_yaw * self._rx_fix

    @property
    def pos(self) -> np.ndarray:
        """脚底位置（楼层地面高度）。"""
        t = self.obj.translation
        return np.array([t.x, t.y - self.lift, t.z], dtype=np.float64)

    # ---- 手臂关节（交互动作，如按电梯按钮）----
    # right_shoulder_pitch_joint 的 dof offset（G1: link25/dof18）
    ARM_DOF = {"g1": 18}

    def arm_pitch(self, target: float, cb=None, frames: int = 10):
        """右臂肩俯仰插值到 target（rad）。pitch<0 抬臂，约 -1.15 为前平举。"""
        dof = self.ARM_DOF.get(self.type)
        if dof is None:
            if cb:
                for _ in range(frames):
                    cb()
            return
        pos0 = np.array(self.obj.joint_positions)
        for k in range(frames):
            t = (k + 1) / frames
            p = pos0.copy()
            p[dof] = pos0[dof] + (target - pos0[dof]) * t
            self.obj.joint_positions = p
            if cb:
                cb()

    # ---- 沿路径移动 ----
    def move_path(self, path, cb=None, step: float = 0.06, turn: float = 0.25):
        """沿 navmesh 路径点移动（resample 近似匀速），朝向路径切线（平滑）。"""
        pts = [np.asarray(p, dtype=np.float64) for p in path]
        dense: list[np.ndarray] = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg = float(np.linalg.norm(b - a))
            n = max(1, int(seg / step))
            for k in range(n):
                dense.append(a + (b - a) * (k / n))
        if pts:
            dense.append(pts[-1])
        for i, p in enumerate(dense):
            if i < len(dense) - 1:
                d = dense[i + 1] - p
                yaw = math.atan2(-d[0], -d[2]) if abs(d[0]) + abs(d[2]) > 1e-4 else self.yaw
            else:
                yaw = self.yaw
            dy = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            yaw = self.yaw + turn * dy
            prev = self.pos
            self.set_pose(p, yaw)
            self._apply_gait(float(np.linalg.norm(self.pos - prev)))
            if cb:
                cb()

    def nav_to(self, pf, hsim, target, cb=None, step: float = 0.06):
        """find_path 到目标点并沿路径移动。"""
        sp = hsim.nav.ShortestPath()
        sp.requested_start = self.pos
        sp.requested_end = np.asarray(target, dtype=np.float32)
        if pf.find_path(sp) and len(sp.points) >= 2:
            self.move_path([np.asarray(p) for p in sp.points], cb=cb, step=step)
            return True
        return False
