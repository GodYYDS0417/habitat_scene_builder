#!/usr/bin/env python3
"""demo_villa_rescue.py — 两层别墅三机器人异构协同 + 电梯交互 + 火灾救援演示。

叙事（多视角跟随）：
  段1 居家巡检：G1 在 F1 走一圈，展示布局与可交互小物件
  段2 异构上楼对比：
      - Go2 机器狗按不到 1.16m 高的电梯按钮 → 只能爬楼梯上 F2
      - G1 人形机器人按按钮 → 门开→进→关→升→开→出 坐电梯上 F2
      - Fetch 机器车爬不上楼梯（climb 0.05<0.145）→ 也坐电梯
  段3 火灾救援：G1 到 F2 主卧，attach 假人，坐电梯搬回 F1 客厅放下

机器人运动学：URDF 加载为 KINEMATIC（防倒），沿 navmesh 路径移动。
电梯交互：按钮可达高度判定 + 门扇滑动 + 轿厢升降（井道不参与 navmesh）。
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
from elevator_system import (CABIN_IN, DOOR_MID, DOOR_OUT, PANEL_FRONT,  # noqa: E402
                             ElevatorSystem)
from robot_actor import RobotActor  # noqa: E402

RUN = HERE.parent / "demo_run" / "villa_2f"
ROBOTS = HERE.parent / "demo_assets" / "robots"
OBJECTS = HERE.parent / "demo_assets" / "objects"
URDF = {"g1": ROBOTS / "g1" / "g1.urdf",
        "go2": ROBOTS / "go2" / "go2_description.urdf",
        "fetch": ROBOTS / "fetch" / "fetch.urdf"}
NAVMESH = {"g1": "human", "go2": "spot", "fetch": "fetch"}


class Demo:
    def __init__(self, args):
        self.args = args
        self.hsim, self.mn = V2._import_habitat()
        hsim, mn = self.hsim, self.mn
        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(RUN / "villa_2f.scene_dataset_config.json")
        sim_cfg.scene_id = str(RUN / "scenes" / "villa_2f_000.scene_instance.json")
        sim_cfg.enable_physics = True
        spec = hsim.CameraSensorSpec()
        spec.uuid = "rgb"
        spec.sensor_type = hsim.SensorType.COLOR
        spec.resolution = [args.resolution, args.resolution]
        spec.position = mn.Vector3(0.0, 0.0, 0.0)
        agent_cfg = hsim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [spec]
        self.sim = hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))
        self.agent = self.sim.get_agent(0)
        self.elev = ElevatorSystem(self.sim, hsim, mn, OBJECTS)
        self.navs = {}
        for key, nav in NAVMESH.items():
            pf = hsim.nav.PathFinder()
            pf.load_nav_mesh(str(RUN / "navmeshes" / f"villa_2f_000__{nav}.navmesh"))
            self.navs[key] = pf
        # 找假人
        self.dummy = None
        mgr = self.sim.get_rigid_object_manager()
        for h in mgr.get_object_handles(""):
            if h.startswith("dummy"):
                self.dummy = mgr.get_object_by_handle(h)
                break
        self.robots = {}
        self.carrying = None
        import imageio.v2 as imageio
        out = RUN / "_preview"
        out.mkdir(parents=True, exist_ok=True)
        self.wr = imageio.get_writer(str(out / "villa_rescue.mp4"), fps=args.fps,
                                     codec="libx264", quality=8, macro_block_size=None)
        self.label = ""
        from PIL import ImageFont
        self.font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)

    # ---- 相机 ----
    def follow(self, robot, dist=1.6, height=1.4, look_h=0.7):
        yaw = robot.yaw
        back = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        cam = robot.pos + back * dist + np.array([0.0, height, 0.0])
        look = robot.pos + np.array([0.0, look_h, 0.0])
        self._set_cam(cam, look)

    def _set_cam(self, eye, target):
        from habitat_sim.utils.common import quat_from_magnum
        m = self.mn.Matrix4.look_at(self.mn.Vector3(*eye), self.mn.Vector3(*target),
                                    self.mn.Vector3(0, 1, 0))
        q = quat_from_magnum(self.mn.Quaternion.from_matrix(m.rotation()))
        st = self.hsim.agent.AgentState()
        st.position = np.array(eye, dtype=np.float32)
        st.rotation = q
        self.agent.set_state(st)

    def fixed(self, eye, target):
        self._set_cam(np.array(eye, dtype=np.float32), np.array(target, dtype=np.float32))

    # ---- 帧 ----
    def _carry_pose(self, robot):
        """背负位姿：假人竖直贴在机器人背后（胸朝机器人背、头朝上）。

        水平横抱的假人体长 1.7m，转弯时横扫墙体必穿模；竖直背负的
        footprint 与机器人相当（背后 0.28m），是消防救援的标准背法。
        """
        yaw = robot.yaw
        back = np.array([math.sin(yaw), 0.0, math.cos(yaw)])   # 背后方向
        pos = robot.pos + back * 0.28 + np.array([0.0, 0.55, 0.0])
        q_yaw = self.mn.Quaternion.rotation(self.mn.Rad(yaw), self.mn.Vector3(0, 1, 0))
        q_up = self.mn.Quaternion.rotation(self.mn.Rad(-math.pi / 2), self.mn.Vector3(1, 0, 0))
        return pos, q_yaw * q_up

    def emit(self, n=1, follow_robot=None, label=None, cam=None):
        for _ in range(n):
            if follow_robot is not None:
                if self.carrying is not None:
                    _p, _q = self._carry_pose(follow_robot)
                    self.carrying.translation = self.mn.Vector3(*_p)
                    self.carrying.rotation = _q
                if cam is None:
                    self.follow(follow_robot)
            if cam is not None:
                self.fixed(*cam)
            self.sim.step_physics(1.0 / self.args.fps)
            obs = self.sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"])[..., :3].copy()
            txt = label if label is not None else self.label
            if txt:
                from PIL import Image, ImageDraw
                im = Image.fromarray(rgb)
                ImageDraw.Draw(im).text((14, 14), txt, font=self.font,
                                        fill=(255, 255, 90), stroke_width=2,
                                        stroke_fill=(0, 0, 0))
                rgb = np.asarray(im)
            self.wr.append_data(rgb)

    # ---- 机器人 ----
    def spawn(self, rtype, pos, yaw=0.0):
        r = RobotActor(self.sim, self.hsim, self.mn, rtype, URDF[rtype], pos, yaw)
        self.robots[rtype] = r
        return r

    def nav(self, robot, target, label=None, follow=True):
        robot.nav_to(self.navs[robot.type], self.hsim, target,
                     cb=lambda: self.emit(1, follow_robot=robot if follow else None, label=label))

    def move_seg(self, robot, *waypoints, label=None, follow=True, cam=None):
        """直线插值移动（用于进/出轿厢等无 navmesh 段），支持多折点。"""
        pts = [robot.pos] + [np.asarray(w, dtype=np.float64) for w in waypoints]
        robot.move_path(pts,
                        cb=lambda: self.emit(1, follow_robot=robot if follow else None,
                                             label=label, cam=cam))

    # ---- 电梯完整序列 ----
    def ride_elevator(self, robot, dst_floor, tag):
        elev = self.elev
        src = elev.cabin_floor
        # 1) 走到电梯门口走廊
        self.nav(robot, DOOR_OUT[src], label=f"{tag} 走向电梯")
        # 2) 走到按钮前，面向面板站定，特写拍抬臂按按钮
        self.nav(robot, PANEL_FRONT[src], label=f"{tag} 走向按钮面板")
        robot.set_pose(PANEL_FRONT[src], 0.0)      # 面向 -z 按钮墙
        fy = 2.9 * (src - 1)
        press_cam = ((4.15, fy + 1.35, -2.35), (3.42, fy + 1.16, -3.25))
        robot.arm_pitch(-1.15, cb=lambda: self.emit(1, cam=press_cam,
                                                    label=f"{tag} 抬手伸向按钮"))
        ok, msg = elev.try_press(robot.type, robot.pos)
        self.emit(self.args.fps // 2, cam=press_cam, label=msg)
        robot.arm_pitch(0.0, cb=lambda: self.emit(1, cam=press_cam,
                                                  label=f"{tag} 收回手臂"))
        if not ok:
            self.emit(self.args.fps, follow_robot=robot, label=f"{tag} 按不到按钮，改走楼梯")
            return False
        # 3) 开门（相机按 src 楼层）
        fy_src = 2.9 * (src - 1)
        self.fixed((2.6, fy_src + 1.6, -1.2), (4.0, fy_src + 1.2, -3.5))
        elev.set_doors(src, True, cb=lambda: self.emit(1, label=f"{tag} 电梯门开"))
        # 4) 走进轿厢（经门洞正中折点走直角折线，避免斜穿门框侧墙）
        self.move_seg(robot, DOOR_MID[src], CABIN_IN[src], label=f"{tag} 走进电梯")
        # 5) 关门（相机按 src 楼层）
        self.fixed((2.6, fy_src + 1.6, -1.2), (4.0, fy_src + 1.2, -3.5))
        elev.set_doors(src, False, cb=lambda: self.emit(1, label=f"{tag} 电梯门关"))
        # 6) 升降（机器人随轿厢）
        cab_y0 = float(elev.cabin.translation.y)
        robot_y0 = robot.pos[1]
        dy = 2.9 * (dst_floor - src)
        for k in range(self.args.fps * 2):
            t = (k + 1) / (self.args.fps * 2)
            tt = t * t * (3 - 2 * t)
            elev.cabin.translation = self.mn.Vector3(4.0, cab_y0 + dy * tt, -4.10)
            robot.set_pose([robot.pos[0], robot_y0 + dy * tt, robot.pos[2]], robot.yaw)
            ry = float(robot.pos[1])
            cam = ((4.55, ry + 2.05, -3.60), (3.95, ry + 0.70, -4.15))
            self.emit(1, follow_robot=robot, cam=cam,
                      label=f"{tag} 电梯{'升' if dst_floor>src else '降'}至 F{dst_floor}")
        elev.cabin_floor = dst_floor
        # 7) 开门（相机按 dst 楼层）
        fy_dst = 2.9 * (dst_floor - 1)
        self.fixed((2.6, fy_dst + 1.6, -1.2), (4.0, fy_dst + 1.2, -3.5))
        elev.set_doors(dst_floor, True, cb=lambda: self.emit(1, label=f"{tag} 电梯门开（F{dst_floor}）"))
        # 8) 走出轿厢：门洞段直线插值出到门口，再沿 navmesh 回走廊
        #    （F2 门口 z=-3.0 是 0.6m 宽楼板带，带外是楼梯间挑空洞口；
        #      直线插值到走廊会悬空穿护栏，必须走 navmesh）
        exit_cam = ((2.6, fy_dst + 1.6, -1.2), (4.0, fy_dst + 1.2, -3.5))
        self.move_seg(robot, DOOR_MID[dst_floor], label=f"{tag} 走出电梯", cam=exit_cam)
        robot.nav_to(self.navs[robot.type], self.hsim, DOOR_OUT[dst_floor],
                     cb=lambda: self.emit(1, follow_robot=robot, cam=exit_cam,
                                          label=f"{tag} 走出电梯"))
        elev.set_doors(dst_floor, False, cb=lambda: self.emit(1, label=f"{tag} 电梯门关"))
        return True

    def close(self):
        self.wr.close()
        self.sim.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    d = Demo(args)

    if args.smoke:
        d.spawn("g1", (-2.2, 0.10, 1.70), yaw=0.6)
        d.spawn("go2", (-3.6, 0.10, 1.00), yaw=-0.6)
        d.spawn("fetch", (-1.0, 0.10, 0.60), yaw=2.4)
        for _ in range(40):
            d.emit(1, follow_robot=d.robots["g1"], label="smoke: G1/Go2/Fetch 加载")
        d.close()
        print("smoke OK ->", RUN / "_preview" / "villa_rescue.mp4")
        return 0

    # ============ 段 1：居家巡检（G1 在 F1 走一圈） ============
    g1 = d.spawn("g1", (-3.0, 0.10, 1.70), yaw=0.6)
    d.label = "段1 居家巡检"
    d.nav(g1, (-0.5, 0.10, -1.10), label="G1 巡检：客厅→餐厅")
    d.nav(g1, (2.0, 0.10, -1.50), label="G1 巡检：餐厅→走廊")

    # ============ 段 2：异构上楼对比 ============
    # G1 巡检完先回客厅（瞬移无帧；避免站在 Go2 上楼梯的路径上被穿模）
    g1.set_pose((-3.0, 0.10, 1.70), 0.6)
    # Go2 机器狗爬楼梯（按不到按钮）
    go2 = d.spawn("go2", (-1.5, 0.10, 2.50), yaw=0.0)
    ok, msg = d.elev.try_press("go2", (3.4, 0.10, -3.0))
    d.fixed((1.0, 1.4, -0.6), (3.42, 1.16, -3.2))
    d.emit(d.args.fps, label=f"Go2 机器狗够不到按钮（可达0.5m<1.16m）→ 改爬楼梯")
    d.nav(go2, (3.4, 0.10, -2.30), label="Go2 爬楼梯：走向楼梯")
    d.nav(go2, (0.35, 2.95, 0.65), label="Go2 爬楼梯：登上 F2")
    d.nav(go2, (-0.30, 2.95, 1.50), label="Go2 到达 F2 走廊")
    # Go2 进茶室待命（避开 G1 沿走廊西侧去主卧的救援路径，防止穿模）
    if not go2.nav_to(d.navs[go2.type], d.hsim, (2.5, 2.95, 3.3),
                      cb=lambda: d.emit(1, follow_robot=go2, label="Go2 进入茶室搜索待命")):
        d.nav(go2, (0.9, 2.95, 0.4), label="Go2 在楼梯间待命")

    # G1 坐电梯上 F2（已在段 2 开头归位客厅）
    d.ride_elevator(g1, 2, "G1")

    # ============ 段 3：火灾救援 ============
    d.nav(g1, (-1.50, 2.95, 2.00), label="G1 救援：F2 走廊→主卧")
    # 到假人旁 attach
    if d.dummy is not None:
        d.nav(g1, (-3.0, 2.95, 3.0), label="G1 救援：接近被困者")
        d.dummy.motion_type = d.hsim.physics.MotionType.KINEMATIC
        # 扶起动画：从床上躺姿平滑插值到背上竖直背负位（避免瞬移穿床）
        p0 = np.array(d.dummy.translation, dtype=np.float64)
        q0 = d.dummy.rotation
        n_lift = d.args.fps
        for k in range(n_lift):
            t = (k + 1) / n_lift
            tt = t * t * (3 - 2 * t)
            p1, q1 = d._carry_pose(g1)
            d.dummy.translation = d.mn.Vector3(*(p0 + (p1 - p0) * tt))
            d.dummy.rotation = d.mn.math.slerp(q0, q1, tt)
            d.emit(1, follow_robot=g1, label="G1 背起被困者")
        d.carrying = d.dummy
        g1.gait_arms = False          # 搬运时手臂扶假人，不随步态摆动
        d.ride_elevator(g1, 1, "G1 搬运")
        d.nav(g1, (-1.5, 0.10, 1.50), label="G1 搬运：送到 F1 安全区")
        # 放下动画：从背负位平滑放到地面躺姿（避免瞬移穿地）
        d.carrying = None
        g1.gait_arms = True
        p0 = np.array(d.dummy.translation, dtype=np.float64)
        q0 = d.dummy.rotation
        p1 = np.array([-1.5, 0.12, 1.50], dtype=np.float64)
        q1 = d.mn.Quaternion.rotation(d.mn.Rad(0.0), d.mn.Vector3(1.0, 0.0, 0.0))
        n_drop = d.args.fps
        for k in range(n_drop):
            t = (k + 1) / n_drop
            tt = t * t * (3 - 2 * t)
            d.dummy.translation = d.mn.Vector3(*(p0 + (p1 - p0) * tt))
            d.dummy.rotation = d.mn.math.slerp(q0, q1, tt)
            d.emit(1, follow_robot=g1, label="G1 放下被困者")
        d.dummy.motion_type = d.hsim.physics.MotionType.DYNAMIC
        d.emit(d.args.fps, follow_robot=g1, label="G1 放下被困者，救援完成")

    d.emit(d.args.fps, follow_robot=g1, label="演示结束")
    d.close()
    print(f"视频 -> {RUN}/_preview/villa_rescue.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
