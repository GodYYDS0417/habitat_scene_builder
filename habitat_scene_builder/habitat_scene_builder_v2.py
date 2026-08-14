#!/usr/bin/env python3
"""
habitat_scene_builder_v2.py
===========================
habitat_scene_builder 的改进版：从「随机撒点 + 全转 STATIC」升级为
「语义规则放置 + 可交互物体保持 DYNAMIC + 人工可干预」。

相对原版的核心改动（对应技术方案 Phase 0-6）：

  Phase 0  物理参数按物体类型分别配置（质量/摩擦/阻尼/精确碰撞），
           小物体 0.3kg 量级、加线性/角阻尼，防飞防滚。
  Phase 1  家具等大物体静置后 STATIC；可抓取小物体按 --interactive-ratio
           保持 DYNAMIC，scene_instance.json 写入正确 motion_type。
  Phase 2  detect_surfaces(): 从场景上方做网格化竖直射线投射，自动发现
           地面/桌面/台面/架子等可放置表面（高度带 + 法线过滤 + 聚类）。
  Phase 3  SEMANTIC_RULES 语义规则库：杯子/盘子 -> 桌面，椅子 -> 挨着
           桌子并面向桌子，沙发/柜子 -> 地面靠墙（启发式），可用
           --rules my_rules.json 整体覆盖（人工干预入口 1）。
  Phase 4  放置管线：先放家具(STATIC) -> 重新探测表面 -> 在家具表面上
           放小物体(DYNAMIC, 下落吸附) -> 处理相邻/朝向关系 ->
           加 ±30° 随机 yaw 和厘米级位置扰动。
  Phase 5  NavMesh 流程：全部物体临时 STATIC -> 逐具身 bake -> 恢复
           可抓取物体为 DYNAMIC 后再写 scene_instance。
  Phase 6  验证：冷启动重载 + RGB-D/semantic 抽检 + 可交互性验证
           （对 DYNAMIC 物体施加速度，确认位移）+ 稳定性检查
           （静置后是否倾倒），全部写入 _preview/build_report.json。

人工干预接口（「不做纯 AI 不可控生成」）：
  * --plan-out plan.json  自动放置后导出完整放置方案（类名/位姿/类型）。
  * --plan-in  plan.json  跳过自动放置，严格按人工编辑过的方案落位。
    即「AI 起草 -> 人改 JSON -> 一键重建」的工作流。
  * --rules rules.json    用自定义语义规则库覆盖内置 SEMANTIC_RULES。
  * --seed                全流程确定性，同参数结果可复现。

用法示例：
    # 语义模式一键生成（自动探测表面 + 规则放置 + 交互验证）
    python habitat_scene_builder_v2.py \
        --stage /path/real_scan.glb --objects assets/objects \
        --out data/my_scene --name my_scene \
        --num-objects 14 --interactive-ratio 0.7

    # 人工干预：先导出方案，编辑后按方案重建
    python habitat_scene_builder_v2.py --stage s.glb --objects assets/objects \
        --out data/x --name x --plan-out my_plan.json
    #   ... 手工编辑 my_plan.json ...
    python habitat_scene_builder_v2.py --stage s.glb --objects assets/objects \
        --out data/x2 --name x2 --plan-in my_plan.json

    # 原版随机模式仍可用（物理参数修复依然生效）
    python habitat_scene_builder_v2.py --stage s.glb --objects assets/objects \
        --out data/y --name y --no-semantic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# 具身预设（与原版一致）
# ---------------------------------------------------------------------------
EMBODIMENTS: dict[str, dict[str, float]] = {
    "default": dict(radius=0.10, height=1.50, max_climb=0.20, max_slope=45.0),
    "human":   dict(radius=0.30, height=1.70, max_climb=0.20, max_slope=45.0),
    "fetch":   dict(radius=0.30, height=1.50, max_climb=0.05, max_slope=30.0),
    "stretch": dict(radius=0.30, height=1.40, max_climb=0.05, max_slope=30.0),
    "spot":    dict(radius=0.25, height=0.65, max_climb=0.15, max_slope=35.0),
    "drone":   dict(radius=0.20, height=0.30, max_climb=1.00, max_slope=85.0),
}

NAVMESH_CELL_SIZE = 0.05
# 体素高度量化：0.20 把 walkableHeight 量化成 9 voxels(1.8m) 使 2.1m 门洞
# 断连；0.10 修门洞但 spot max_climb=0.15 只剩 1 voxel(0.10m)，楼梯坡道
# 接缝爬不上（实测整段坡道断岛）；0.05 时 0.15m->3 voxels，坡道/门洞全通。
NAVMESH_CELL_HEIGHT = 0.05

UP_VECTORS = {
    "y": ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
    "z": ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
}

MESH_EXTS = (".glb", ".gltf", ".ply", ".obj")

# ---------------------------------------------------------------------------
# 物体物理档案（Phase 0）与语义规则库（Phase 3）
# ---------------------------------------------------------------------------
# furniture=True  -> 静置后 STATIC（不可抓取）；False -> 小物体，可 DYNAMIC
OBJECT_PROFILES: dict[str, dict[str, Any]] = {
    # 家具
    "table":      dict(furniture=True,  mass=15.0),
    "desk":       dict(furniture=True,  mass=12.0),
    "chair":      dict(furniture=True,  mass=5.0),
    "sofa":       dict(furniture=True,  mass=30.0),
    "coffee_table": dict(furniture=True, mass=12.0),
    "tv_stand":   dict(furniture=True,  mass=25.0),
    "tv":         dict(furniture=True,  mass=8.0),
    "bed":        dict(furniture=True,  mass=50.0),
    "counter":    dict(furniture=True,  mass=60.0),
    "cabinet":    dict(furniture=True,  mass=40.0),
    "elevator_car": dict(furniture=True, mass=500.0),
    "nightstand": dict(furniture=True,  mass=12.0),
    "shelf":      dict(furniture=True,  mass=25.0),
    "lamp":       dict(furniture=False, mass=1.2),
    # 小物体（可抓取）
    "cup":        dict(furniture=False, mass=0.25),
    "mug":        dict(furniture=False, mass=0.30),
    "plate":      dict(furniture=False, mass=0.40),
    "bowl":       dict(furniture=False, mass=0.30),
    "bottle":     dict(furniture=False, mass=0.50),
    "book":       dict(furniture=False, mass=0.45),
    "apple":      dict(furniture=False, mass=0.15, round=True),
    "orange":     dict(furniture=False, mass=0.18, round=True),
    "box":        dict(furniture=False, mass=0.60),
    "keyboard":   dict(furniture=False, mass=0.80),
}
DEFAULT_PROFILE = dict(furniture=False, mass=0.5)

# 语义放置规则。surface 类别由 detect_surfaces 的高度带分类给出：
#   "floor"(地面) / "table"(桌台高度带) / "shelf"(高架子带)
# on:           直接放到该类表面上
# next_to:      放到某类已放置家具旁边（距离区间，面向它）
# against_wall: 贴墙（启发式：取房间包络边缘的地面点，背靠边界）
SEMANTIC_RULES: dict[str, dict[str, Any]] = {
    # 小物体 -> 表面
    "cup":      {"on": ["table"], "upright": True},
    "mug":      {"on": ["table"], "upright": True},
    "plate":    {"on": ["table"]},
    "bowl":     {"on": ["table"]},
    "bottle":   {"on": ["table", "shelf", "floor"], "upright": True},
    "book":     {"on": ["table", "shelf", "floor"], "flat": True},
    "apple":    {"on": ["table", "floor"]},
    "orange":   {"on": ["table", "floor"]},
    "box":      {"on": ["floor", "table"]},
    "keyboard": {"on": ["table"]},
    "lamp":     {"on": ["table", "floor"], "upright": True},
    # 家具 -> 地面 + 相邻关系（样板房布局：沙发/书桌居中围合，柜子贴墙）
    "table":    {"on": ["floor"]},
    "desk":     {"on": ["floor"]},
    "chair":    {"on": ["floor"], "next_to": ["table", "desk"],
                 "distance": [0.35, 0.65], "facing": "support"},
    "sofa":     {"on": ["floor"]},
    "coffee_table": {"on": ["floor"]},
    "tv_stand": {"on": ["floor"], "against_wall": True},
    "tv":       {"on": ["table"]},
    "bed":      {"on": ["floor"]},
    "counter":  {"on": ["floor"], "against_wall": True},
    "cabinet":  {"on": ["floor"], "against_wall": True},
    "nightstand": {"on": ["floor"], "against_wall": True},
    "shelf":    {"on": ["floor"], "against_wall": True},
}

# 高度带分类阈值（相对参考地面高度，米）
SURFACE_BANDS = {
    "floor": (-0.25, 0.25),
    "table": (0.45, 1.15),
    "shelf": (1.15, 1.95),
}

# ---------------------------------------------------------------------------
# 基础数据结构 / JSON 工具
# ---------------------------------------------------------------------------

@dataclass
class BuildSpec:
    stage_src: Path
    objects_src: Path | None
    out_root: Path
    dataset_name: str
    scene_name: str
    stage_up: str = "y"
    object_up: str = "y"
    units_to_meters: float = 1.0
    stage_collision_asset: Path | None = None
    embodiments: list[str] = field(default_factory=lambda: ["default"])
    interactive_ratio: float = 0.7

    object_classes: list[str] = field(default_factory=list)
    semantic_id_map: dict[str, int] = field(default_factory=dict)

    @property
    def dataset_config_path(self) -> Path:
        return self.out_root / f"{self.dataset_name}.scene_dataset_config.json"

    @property
    def scene_instance_path(self) -> Path:
        return self.out_root / "scenes" / f"{self.scene_name}.scene_instance.json"

    @property
    def stage_handle(self) -> str:
        return f"stages/{self.stage_src.stem}"

    def object_handle(self, cls: str) -> str:
        return f"objects/{cls}"

    def object_config_path(self, cls: str) -> Path:
        return self.out_root / "objects" / f"{cls}.object_config.json"

    def navmesh_handle(self, embodiment: str) -> str:
        return f"{self.scene_name}__{embodiment}"

    def navmesh_path(self, embodiment: str) -> Path:
        return self.out_root / "navmeshes" / f"{self.navmesh_handle(embodiment)}.navmesh"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {level:<5} {msg}", flush=True)


def die(msg: str) -> None:
    log(msg, "FATAL")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. SCAFFOLD（Phase 0：物理参数修复）
# ---------------------------------------------------------------------------

def object_config_payload(cls: str, mesh_name: str, semantic_id: int,
                          up: list[float], front: list[float],
                          units_to_meters: float) -> dict[str, Any]:
    profile = OBJECT_PROFILES.get(cls, DEFAULT_PROFILE)
    is_furniture = bool(profile["furniture"])
    mass = float(profile["mass"])
    return {
        "render_asset": mesh_name,
        "collision_asset": mesh_name,
        "up": up,
        "front": front,
        "units_to_meters": units_to_meters,
        "mass": mass,
        "friction_coefficient": 0.6 if not is_furniture else 0.5,
        "rolling_friction_coefficient": 0.005,
        "spinning_friction_coefficient": 0.003,
        "restitution_coefficient": 0.1,
        "margin": 0.005,
        "linear_damping": 0.2,
        "angular_damping": 0.3,
        # 不能用 bbox：我们的模型原点在底面，bbox 碰撞会以几何中心为准，
        # 导致碰撞体整体下沉半高——桌上物体悬空。
        # 也不能 join 拼接凹面：实测拼接凹面会在书架内腔产生幻影接触，
        # 把书弹飞（probe_shelf2.py）。多节点 GLB + join=false => 每零件
        # 独立凸包（盒/柱/球的凸包即精确形状），box-box 是 Bullet 最稳路径。
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": False,
        "is_collidable": True,
        "compute_COM_from_shape": True,
        "semantic_id": semantic_id,
        "shader_type": "pbr",
        "user_defined": {
            "class_name": cls,
            "graspable": not is_furniture,
            "manipulation_type": "rigid",
        },
    }


def scaffold(spec: BuildSpec) -> BuildSpec:
    log(f"scaffold -> {spec.out_root}")
    for sub in ("stages", "objects", "scenes", "navmeshes", "_preview"):
        (spec.out_root / sub).mkdir(parents=True, exist_ok=True)

    if not spec.stage_src.exists():
        die(f"stage 资产不存在: {spec.stage_src}")
    stage_dst = spec.out_root / "stages" / spec.stage_src.name
    if stage_dst.resolve() != spec.stage_src.resolve():
        shutil.copy2(spec.stage_src, stage_dst)
    # 楼梯坡道辅助网格（仅 navmesh 烘焙用）随 stage 一起拷贝
    nav_src = spec.stage_src.with_name(f"{spec.stage_src.stem}_navramps.glb")
    if nav_src.exists():
        shutil.copy2(nav_src, spec.out_root / "stages" / nav_src.name)

    up, front = UP_VECTORS[spec.stage_up]
    stage_cfg: dict[str, Any] = {
        "render_asset": stage_dst.name,
        "collision_asset": stage_dst.name,
        "up": up,
        "front": front,
        "origin": [0.0, 0.0, 0.0],
        "units_to_meters": spec.units_to_meters,
        "gravity": [0.0, -9.8, 0.0],
        "margin": 0.03,
        "friction_coefficient": 0.4,
        "restitution_coefficient": 0.1,
        "is_collidable": True,
        "shader_type": "material",
    }
    if spec.stage_collision_asset is not None:
        coll_dst = spec.out_root / "stages" / spec.stage_collision_asset.name
        if coll_dst.resolve() != spec.stage_collision_asset.resolve():
            shutil.copy2(spec.stage_collision_asset, coll_dst)
        stage_cfg["collision_asset"] = coll_dst.name
    write_json(spec.out_root / "stages" / f"{spec.stage_src.stem}.stage_config.json",
               stage_cfg)

    classes: list[str] = []
    semantic_map: dict[str, int] = {}
    if spec.objects_src is not None:
        if not spec.objects_src.is_dir():
            die(f"--objects 需要是目录: {spec.objects_src}")
        meshes = sorted(
            p for p in spec.objects_src.iterdir()
            if p.suffix.lower() in MESH_EXTS and p.is_file()
        )
        if not meshes:
            log(f"{spec.objects_src} 里没找到 {MESH_EXTS} 资产，场景将只有 stage", "WARN")

        up_o, front_o = UP_VECTORS[spec.object_up]
        for idx, mesh in enumerate(meshes, start=1):
            cls = mesh.stem
            dst = spec.out_root / "objects" / mesh.name
            if dst.resolve() != mesh.resolve():
                shutil.copy2(mesh, dst)
            write_json(
                spec.object_config_path(cls),
                object_config_payload(cls, dst.name, idx, up_o, front_o,
                                      spec.units_to_meters),
            )
            classes.append(cls)
            semantic_map[cls] = idx

    spec.object_classes = classes
    spec.semantic_id_map = semantic_map
    write_json(spec.out_root / "semantic_id_map.json",
               {"background": 0, **semantic_map})

    write_json(spec.dataset_config_path, build_dataset_config(spec, baked=[]))
    # 场景实例始终先写空：pass 1 仿真器从这里加载场景，若残留上次
    # 运行的物体，新放置会叠在旧物体上（悬空/堵路/navmesh 错乱）
    write_scene_instance(spec, object_instances=[])

    log(f"scaffold 完成: stage=1, object 类别={len(classes)}, "
        f"具身={len(spec.embodiments)}")
    return spec


def build_dataset_config(spec: BuildSpec, baked: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "stages": {"paths": {".json": ["stages"]}},
        "objects": {"paths": {".json": ["objects"]}},
        "articulated_objects": {"paths": {".json": ["urdf"]}},
        "light_setups": {"paths": {".json": ["lights"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}},
        "semantic_scene_descriptor_instances": {"paths": {".json": ["semantics"]}},
    }
    if baked:
        cfg["navmesh_instances"] = {
            spec.navmesh_handle(e): f"navmeshes/{spec.navmesh_handle(e)}.navmesh"
            for e in baked
        }
    return cfg


def register_navmeshes(spec: BuildSpec, baked: list[str]) -> list[str]:
    on_disk = [e for e in baked if spec.navmesh_path(e).exists()]
    missing = [e for e in baked if e not in on_disk]
    if missing:
        log(f"navmesh 文件缺失，不登记: {', '.join(missing)}", "WARN")
    write_json(spec.dataset_config_path, build_dataset_config(spec, on_disk))
    return on_disk


def write_scene_instance(
    spec: BuildSpec,
    object_instances: list[dict[str, Any]],
    primary_embodiment: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage_instance": {"template_name": spec.stage_handle},
        "default_lighting": "",
        "object_instances": object_instances,
        "articulated_object_instances": [],
    }
    if primary_embodiment is not None:
        payload["navmesh_instance"] = spec.navmesh_handle(primary_embodiment)
    write_json(spec.scene_instance_path, payload)


# ---------------------------------------------------------------------------
# 2. 仿真侧基础设施
# ---------------------------------------------------------------------------

def _import_habitat():
    try:
        import habitat_sim  # noqa: F401
        import magnum as mn  # noqa: F401
    except ImportError as exc:
        die(
            f"需要 habitat-sim 才能跑 populate/navmesh/validate 阶段 ({exc})。\n"
            "       安装:  conda install habitat-sim withbullet -c conda-forge -c aihabitat\n"
            "       或者只跑配置生成:  --scaffold-only"
        )
    return sys.modules["habitat_sim"], sys.modules["magnum"]


def contact_test(sim, obj) -> bool:
    if hasattr(obj, "contact_test"):
        return bool(obj.contact_test())
    return bool(sim.contact_test(obj.object_id))


def make_navmesh_settings(hsim, preset: dict[str, float], include_static: bool):
    ns = hsim.NavMeshSettings()
    ns.set_defaults()
    ns.cell_size = NAVMESH_CELL_SIZE
    ns.cell_height = NAVMESH_CELL_HEIGHT
    ns.agent_radius = preset["radius"]
    ns.agent_height = preset["height"]
    ns.agent_max_climb = preset["max_climb"]
    ns.agent_max_slope = preset["max_slope"]
    if hasattr(ns, "include_static_objects"):
        ns.include_static_objects = include_static
    return ns


def recompute_navmesh_compat(sim, hsim, settings, include_static: bool) -> bool:
    if hasattr(settings, "include_static_objects"):
        return sim.recompute_navmesh(sim.pathfinder, settings)
    try:
        return sim.recompute_navmesh(sim.pathfinder, settings, include_static)
    except TypeError:
        return sim.recompute_navmesh(sim.pathfinder, settings)


def open_sim(hsim, spec: BuildSpec, with_sensors: bool, resolution: int = 512):
    sim_cfg = hsim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(spec.dataset_config_path)
    sim_cfg.scene_id = str(spec.scene_instance_path)
    sim_cfg.enable_physics = True

    agent_cfg = hsim.agent.AgentConfiguration()
    specs = []
    if with_sensors:
        for uuid, stype in (
            ("rgb", hsim.SensorType.COLOR),
            ("depth", hsim.SensorType.DEPTH),
            ("semantic", hsim.SensorType.SEMANTIC),
        ):
            s = hsim.CameraSensorSpec()
            s.uuid = uuid
            s.sensor_type = stype
            s.resolution = [resolution, resolution]
            # 0.3.1 的 position/orientation 需要 magnum Vector3，纯列表会报错
            import magnum as _mn
            s.position = _mn.Vector3(0.0, 1.2, 0.0)
            s.orientation = _mn.Vector3(-0.30, 0.0, 0.0)  # 略微俯视
            specs.append(s)
    agent_cfg.sensor_specifications = specs
    return hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))


def bootstrap_navmesh(sim, hsim) -> bool:
    loose = dict(radius=0.05, height=0.80, max_climb=0.30, max_slope=45.0)
    ns = make_navmesh_settings(hsim, loose, include_static=False)
    ok = recompute_navmesh_compat(sim, hsim, ns, False)
    if not ok or not sim.pathfinder.is_loaded:
        return False
    log(f"bootstrap navmesh 可行走面积 = {sim.pathfinder.navigable_area:.2f} m^2")
    return True


def resolve_template_handles(spec: BuildSpec, obj_tmpl_mgr) -> dict[str, str]:
    handles: dict[str, str] = {}
    for cls in spec.object_classes:
        found = obj_tmpl_mgr.get_template_handles(cls)
        if not found:
            log(f"模板未注册，跳过: {cls}", "WARN")
            continue
        want = spec.object_config_path(cls).resolve()
        exact = [h for h in found if Path(h).resolve() == want]
        if exact:
            handles[cls] = exact[0]
        else:
            handles[cls] = found[0]
            if len(found) > 1:
                log(f"'{cls}' 子串命中 {len(found)} 个模板，取 {found[0]}", "WARN")
    return handles


def settle(sim, seconds: float, hz: int = 60) -> None:
    for _ in range(int(seconds * hz)):
        sim.step_physics(1.0 / hz)


def obj_pose(obj) -> tuple[list[float], list[float]]:
    t = obj.translation
    q = obj.rotation
    return ([float(t.x), float(t.y), float(t.z)],
            [float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)])


def make_instance(spec: BuildSpec, cls: str, translation, rotation, motion_type: str,
                  rule: str = "", extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    inst = {
        "template_name": spec.object_handle(cls),
        "translation": [float(v) for v in translation],
        "rotation": [float(v) for v in rotation],
        "motion_type": motion_type,
        "translation_origin": "COM",
        "uniform_scale": 1.0,
    }
    if rule:
        inst["user_defined"] = {"placement_rule": rule, **(extra or {})}
    return inst


# ---------------------------------------------------------------------------
# 3. Phase 2 — 表面检测（射线投射）
# ---------------------------------------------------------------------------

@dataclass
class Surface:
    category: str            # floor / table / shelf
    center: tuple[float, float, float]
    extent_xz: tuple[float, float]   # 半宽（xz）
    height: float
    n_points: int


def _cast_down(sim, hsim, mn, x: float, z: float, y_from: float):
    """竖直向下打一条射线，返回 (hit_point, normal)。失败返回 None。

    habitat-sim 0.3.x: sim.cast_ray(ray) -> RaycastResults(.has_hits(), .hits)
    hits[i] 是 RayHitInfo(.point, .normal, .object_id, .ray_distance)。
    """
    ray = hsim.geo.Ray(mn.Vector3(x, y_from, z), mn.Vector3(0.0, -1.0, 0.0))
    try:
        res = sim.cast_ray(ray)
    except Exception:
        return None
    hits = None
    if hasattr(res, "has_hits"):
        if not res.has_hits():
            return None
        hits = res.hits
    elif hasattr(res, "has_hit"):  # 更老的单发 RayHitInfo 接口
        if not res.has_hit:
            return None
        hits = [res]
    if not hits:
        return None
    hit = hits[0]
    p, n = hit.point, hit.normal
    return (float(p.x), float(p.y), float(p.z)), (float(n.x), float(n.y), float(n.z))


def _estimate_floor_y(sim, n_samples: int = 150) -> Optional[float]:
    """从 navmesh 随机可行走点估计主地面高度。

    取高度直方图的「众数」而不是中位数：loose navmesh（max_climb 大）
    会把楼梯/地下室/二楼都纳入可行走区，多楼层样本下中位数不稳定，
    而面积最大的一层对应的主峰是稳定的。
    """
    try:
        if not sim.pathfinder.is_loaded:
            return None
    except Exception:
        return None
    ys: list[float] = []
    for _ in range(n_samples):
        try:
            pt = sim.pathfinder.get_random_navigable_point()
        except Exception:
            break
        y = float(pt[1])
        if y == y:  # 过滤 NaN
            ys.append(y)
    if len(ys) < 10:
        return None
    bins: dict[int, int] = {}
    for y in ys:
        k = int(y // 0.25)
        bins[k] = bins.get(k, 0) + 1
    best = max(bins.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return best * 0.25 + 0.125


def _surface_reachable(sim, mn, s: "Surface", max_snap: float = 1.5) -> bool:
    """表面质心投影到 navmesh 的距离不超过 max_snap 才认为可达（可放置）。"""
    try:
        if not sim.pathfinder.is_loaded:
            return True
        snap = sim.pathfinder.snap_point(mn.Vector3(s.center[0], s.height, s.center[2]))
    except Exception:
        return True
    if any(v != v for v in snap):
        return False
    dx = float(snap[0]) - s.center[0]
    dz = float(snap[2]) - s.center[2]
    return dx * dx + dz * dz <= max_snap * max_snap


def detect_surfaces(
    sim, hsim, mn,
    grid_step: float = 0.20,
    max_surface_extent: float = 3.0,
    min_points: int = 4,
    scan_above_floor: float = 1.6,
    debug: bool = False,
) -> list[Surface]:
    """网格化向下射线扫描，按高度带聚类出可放置表面。

    关键：射线起点必须低于天花板（默认地面以上 1.6m），否则封闭扫描
    会全部命中屋顶/二楼地板，得到完全错误的「地面」。
    地面高度优先用 navmesh 采样中位数；无 navmesh 时退回命中点低分位数。
    """
    bounds = sim.get_active_scene_graph().get_root_node().cumulative_bb
    xmin, xmax = float(bounds.min.x), float(bounds.max.x)
    ymin, ymax = float(bounds.min.y), float(bounds.max.y)
    zmin, zmax = float(bounds.min.z), float(bounds.max.z)
    nav_floor_y = _estimate_floor_y(sim)
    y_from = (nav_floor_y + scan_above_floor) if nav_floor_y is not None else ymax + 0.5

    hits: list[tuple[float, float, float]] = []
    x = xmin + grid_step / 2
    while x < xmax:
        z = zmin + grid_step / 2
        while z < zmax:
            r = _cast_down(sim, hsim, mn, x, z, y_from)
            if r is not None:
                pt, n = r
                if n[1] > 0.85:      # 近似水平的面才可放置
                    hits.append(pt)
            z += grid_step
        x += grid_step
    if not hits:
        log("表面检测：没有任何命中", "WARN")
        return []

    import numpy as np
    ys = np.array([h[1] for h in hits])
    # 参考地面：优先 navmesh 中位数；否则退回命中点低分位数（开放场景）
    floor_y = nav_floor_y if nav_floor_y is not None else float(np.percentile(ys, 5.0))
    if debug:
        log(f"表面检测: {len(hits)} 个水平命中点, 参考地面 y={floor_y:.3f} "
            f"(navmesh={nav_floor_y is not None}, 射线起点 y={y_from:.3f})")

    # 按高度带分桶
    banded: dict[str, list[tuple[float, float, float]]] = {k: [] for k in SURFACE_BANDS}
    for pt in hits:
        rel = pt[1] - floor_y
        for band, (lo, hi) in SURFACE_BANDS.items():
            if lo <= rel < hi:
                banded[band].append(pt)
                break

    surfaces: list[Surface] = []
    for band, pts in banded.items():
        if len(pts) < min_points:
            continue
        # 简单 2D 网格聚类：把相邻格子连成一片表面
        cells: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for px, py, pz in pts:
            key = (int(px // grid_step), int(pz // grid_step))
            cells.setdefault(key, []).append((px, py, pz))
        visited: set[tuple[int, int]] = set()
        for key in cells:
            if key in visited:
                continue
            stack, cluster = [key], []
            visited.add(key)
            while stack:
                kx, kz = stack.pop()
                cluster.extend(cells[(kx, kz)])
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (kx + dx, kz + dz)
                        if nk in cells and nk not in visited:
                            visited.add(nk)
                            stack.append(nk)
            if len(cluster) < min_points:
                continue
            xs = [c[0] for c in cluster]
            zs = [c[2] for c in cluster]
            ysv = [c[1] for c in cluster]
            ex = (max(xs) - min(xs)) / 2.0
            ez = (max(zs) - min(zs)) / 2.0
            if ex > max_surface_extent or ez > max_surface_extent:
                # 太大的片（比如整层地板）按固定块切开，便于后续采样均匀
                nx = max(1, int(math.ceil((max(xs) - min(xs)) / max_surface_extent)))
                nz = max(1, int(math.ceil((max(zs) - min(zs)) / max_surface_extent)))
                for ix in range(nx):
                    for iz in range(nz):
                        sub = [c for c in cluster
                               if min(xs) + ix * (max(xs) - min(xs) + 1e-6) / nx <= c[0]
                               < min(xs) + (ix + 1) * (max(xs) - min(xs) + 1e-6) / nx
                               and min(zs) + iz * (max(zs) - min(zs) + 1e-6) / nz <= c[2]
                               < min(zs) + (iz + 1) * (max(zs) - min(zs) + 1e-6) / nz]
                        if len(sub) >= min_points:
                            surfaces.append(_make_surface(band, sub))
                continue
            surfaces.append(_make_surface(band, cluster))

    # 剔除 navmesh 不可达的表面（阳台外、二楼夹层顶等假表面）
    surfaces = [s for s in surfaces if _surface_reachable(sim, mn, s)]
    surfaces.sort(key=lambda s: (SURFACE_BANDS.get(s.category, (0, 0))[0], -s.n_points))
    counts: dict[str, int] = {}
    for s in surfaces:
        counts[s.category] = counts.get(s.category, 0) + 1
    log("表面检测: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if counts else "表面检测: 无可用表面")
    return surfaces


def _make_surface(category: str, pts: list[tuple[float, float, float]]) -> Surface:
    import numpy as np
    arr = np.asarray(pts)
    center = arr.mean(axis=0)
    ex = float(max(arr[:, 0].max() - arr[:, 0].min(), 0.05)) / 2.0
    ez = float(max(arr[:, 2].max() - arr[:, 2].min(), 0.05)) / 2.0
    return Surface(
        category=category,
        center=(float(center[0]), float(center[1]), float(center[2])),
        extent_xz=(ex, ez),
        height=float(center[1]),
        n_points=len(pts),
    )


def sample_point_on_surface(rng: random.Random, surface: Surface,
                            margin: float) -> tuple[float, float]:
    ex = max(surface.extent_xz[0] - margin, 0.01)
    ez = max(surface.extent_xz[1] - margin, 0.01)
    return (surface.center[0] + rng.uniform(-ex, ex),
            surface.center[2] + rng.uniform(-ez, ez))


# ---------------------------------------------------------------------------
# 4. Phase 3/4 — 语义放置引擎
# ---------------------------------------------------------------------------

class SemanticPlacer:
    def __init__(self, sim, hsim, mn, spec: BuildSpec,
                 handles: dict[str, str], rules: dict[str, dict[str, Any]],
                 seed: int):
        self.sim = sim
        self.hsim = hsim
        self.mn = mn
        self.spec = spec
        self.handles = handles
        self.rules = rules
        self.rng = random.Random(seed)
        self.rigid_mgr = sim.get_rigid_object_manager()
        self.placed_furniture: list[tuple[Any, str]] = []
        self.placed_small: list[tuple[Any, str]] = []
        self._meta: list[dict[str, Any]] = []   # 与 placed_furniture 对齐
        self._dims: dict[str, tuple[float, float, float, float]] = {}

    # ---- 低层 ----
    def _class_dims(self, cls: str) -> tuple[float, float, float, float]:
        """测量模板局部包围盒: (size_x, size_y, size_z, min_y)，懒加载缓存。"""
        if cls not in self._dims:
            obj = self.rigid_mgr.add_object_by_template_handle(self.handles[cls])
            node = obj.root_scene_node
            bb = self.hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
            self._dims[cls] = (float(bb.size_x()), float(bb.size_y()),
                               float(bb.size_z()), float(bb.min.y))
            self.rigid_mgr.remove_object_by_id(obj.object_id)
        return self._dims[cls]

    def _spawn(self, cls: str, xyz, yaw: float, dynamic: bool):
        obj = self.rigid_mgr.add_object_by_template_handle(self.handles[cls])
        if obj is None:
            return None
        _sx, _sy, _sz, bb_min_y = self._class_dims(cls)
        # 抬升量 = 包围盒底到局部原点的距离 + 2cm 间隙。
        # 不能按「原点在几何中心」加 size_y/2：参数化网格的原点在底面，
        # 加半高会让物体先生成在半空，静置时掉到椅面/半墙/别的家具上。
        lift = 0.02 - bb_min_y
        obj.translation = self.mn.Vector3(float(xyz[0]), float(xyz[1]) + lift, float(xyz[2]))
        obj.rotation = self.mn.Quaternion.rotation(
            self.mn.Rad(float(yaw)), self.mn.Vector3.y_axis())
        # 接触检测期间必须保持 KINEMATIC：
        # DYNAMIC 刚体的 Bullet 投机接触边距会让 contact_test 对
        # 「悬空 3cm」也判定接触地面，导致 100% 误杀。
        obj.motion_type = self.hsim.physics.MotionType.KINEMATIC
        if contact_test(self.sim, obj):
            self.rigid_mgr.remove_object_by_id(obj.object_id)
            return None
        return obj

    def _settle_supported(self, obj, seconds: float = 1.0,
                          max_fall: float = 0.10) -> bool:
        """转 DYNAMIC 静置，并验证「下面确实有支撑」。

        生成时包围盒底距表面只有 2cm，有支撑则落地位移 ≈ 2cm；
        假表面/边缘位置会直接掉下去（位移几十厘米），判失败由调用方移除。
        """
        y0 = float(obj.translation.y)
        obj.motion_type = self.hsim.physics.MotionType.DYNAMIC
        settle(self.sim, seconds)
        return abs(float(obj.translation.y) - y0) <= max_fall

    def _footprint_radius(self, obj) -> float:
        node = obj.root_scene_node
        bb = self.hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        return 0.5 * max(float(bb.size_x()), float(bb.size_z()))

    # ---- 家具放置（地面）----
    def place_furniture(self, max_per_class: int = 1, chair_count: int = 4,
                        min_nav_area: float = 4.0) -> None:
        floors = [s for s in
                  detect_surfaces(self.sim, self.hsim, self.mn)
                  if s.category == "floor"]
        if not floors:
            log("没有检测到地面表面，家具放置跳过", "WARN")
            return
        floors.sort(key=lambda s: -s.n_points)
        floor = floors[0]

        furniture_classes = [c for c in self.spec.object_classes
                             if c != "elevator_car"   # 轿厢只由 plan 精确落位
                             and (self.rules.get(c, {}).get("on") == ["floor"]
                             or OBJECT_PROFILES.get(c, DEFAULT_PROFILE)["furniture"])]
        for cls in furniture_classes:
            if cls not in self.handles:
                continue
            rule = self.rules.get(cls, {})
            want = chair_count if cls == "chair" else max_per_class
            if rule.get("next_to"):
                continue  # 椅子等主家具放完再处理
            use_wall = bool(rule.get("against_wall"))
            placed_n = 0
            for _try in range(want * 25):          # 接触/支撑失败要重试
                if placed_n >= want:
                    break
                if use_wall:
                    xyz, yaw = self._sample_wall_pose(cls, floors)
                else:
                    xyz, yaw = self._sample_floor_pose(floor, against_wall=False)
                if xyz is None:
                    continue
                obj = self._spawn(cls, xyz, yaw, dynamic=True)
                if obj is None:
                    continue
                if not self._settle_supported(obj, 1.0):
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
                if not self._clear_of_furniture(obj):
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
                obj.motion_type = self.hsim.physics.MotionType.STATIC
                self.placed_furniture.append((obj, cls))
                self._meta.append(dict(cls=cls, wall=use_wall))
                placed_n += 1

        # 有 next_to 规则的（典型：chair next_to table）
        for cls in furniture_classes:
            rule = self.rules.get(cls, {})
            if not rule.get("next_to") or cls not in self.handles:
                continue
            supports = [(o, c) for (o, c) in self.placed_furniture
                        if c in rule["next_to"]]
            if not supports:
                continue
            want = chair_count if cls == "chair" else max_per_class
            d_lo, d_hi = rule.get("distance", [0.35, 0.65])
            placed_n = 0
            for _try in range(want * 25):
                if placed_n >= want:
                    break
                sup_obj, _sup_cls = self.rng.choice(supports)
                st = sup_obj.translation
                sup_r = self._footprint_radius(sup_obj)
                ang = self.rng.uniform(0.0, 2.0 * math.pi)
                dist = sup_r + self.rng.uniform(d_lo, d_hi) + 0.1
                px, pz = st.x + math.cos(ang) * dist, st.z + math.sin(ang) * dist
                # 面向支撑物
                yaw = math.atan2(st.x - px, st.z - pz)
                # 不能用 st.y 当地面：网格原点可能在几何中心（如 table
                # 原点高 = 地面 + 半高），要从椅子位置实测地面高度
                gy = self._ground_height(px, floor.height + 0.5, pz)
                if gy is None or abs(gy - floor.height) > 0.35:
                    continue   # 悬空（楼梯口/虚空）或落到别的台面上
                obj = self._spawn(cls, (px, gy, pz), yaw, dynamic=True)
                if obj is None:
                    continue
                if not self._settle_supported(obj, 1.0):
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
                if not self._clear_of_furniture(obj, ignore_id=sup_obj.object_id):
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
                if self._chair_back_blocked(obj):
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
                obj.motion_type = self.hsim.physics.MotionType.STATIC
                self.placed_furniture.append((obj, cls))
                self._meta.append(dict(cls=cls, wall=False))
                placed_n += 1
        log(f"家具放置完成: {len(self.placed_furniture)} 件")
        self._ensure_navigable(min_nav_area)

    # ---- 家具间距/贴墙合理性检查 ----
    def _clear_of_furniture(self, obj, ignore_id: Optional[int] = None,
                            min_gap: float = 0.15) -> bool:
        """与已摆家具保持最小外接圆间隙，防扎堆。"""
        if not self.placed_furniture:
            return True
        r = self._footprint_radius(obj)
        for other, _cls in self.placed_furniture:
            if ignore_id is not None and other.object_id == ignore_id:
                continue
            ro = self._footprint_radius(other)
            d = math.hypot(float(obj.translation.x - other.translation.x),
                           float(obj.translation.z - other.translation.z))
            if d < r + ro + min_gap:
                return False
        return True

    def _chair_back_blocked(self, obj, back_gap: float = 0.30) -> bool:
        """椅子背面方向有墙/家具则拒绝（防止嵌进墙里）。"""
        q = obj.rotation.to_matrix()
        back_x, back_z = -float(q[0][2]), -float(q[2][2])   # 背面 = -forward
        t = obj.translation
        back_dist = self._cast_horizontal(float(t.x), float(t.y) + 0.35,
                                          float(t.z),
                                          math.atan2(back_x, back_z))
        return back_dist is not None and back_dist < back_gap

    # ---- 可通行性验收：navmesh 面积不够就撤家具 ----
    def _ensure_navigable(self, min_area: float) -> None:
        if min_area <= 0 or not self.placed_furniture:
            return
        # 用最难通过的具身参数（fetch：半径大、爬升低）做验收
        settings = make_navmesh_settings(
            self.hsim, dict(radius=0.30, height=1.50, max_climb=0.05, max_slope=30.0),
            include_static=True)

        def measure() -> float:
            recompute_navmesh_compat(self.sim, self.hsim, settings, True)
            if not self.sim.pathfinder.is_loaded:
                return 0.0
            return float(self.sim.pathfinder.navigable_area)

        area = measure()
        removed: list[str] = []
        while area < min_area and len(self.placed_furniture) > 2:
            idx = self._pick_removal()
            obj, cls = self.placed_furniture.pop(idx)
            self._meta.pop(idx)
            self.rigid_mgr.remove_object_by_id(obj.object_id)
            removed.append(cls)
            area = measure()
        if removed:
            log(f"可通行性验收: 撤除 {len(removed)} 件挡路家具 ({', '.join(removed)})，"
                f"navmesh {area:.2f} m^2 >= {min_area:.1f}", "WARN")
        else:
            log(f"可通行性验收: navmesh {area:.2f} m^2 (阈值 {min_area:.1f})")

    def _pick_removal(self) -> int:
        """撤除优先级：多余椅子（留 2 把）> 非贴墙家具里侵蚀面积最大的。
        贴墙家具（柜/架/床头柜）不挡主通道，尽量保留，保样板房观感。"""
        chairs = [i for i, (_o, c) in enumerate(self.placed_furniture) if c == "chair"]
        if len(chairs) > 2:
            return chairs[-1]
        non_wall = [i for i, m in enumerate(self._meta) if not m.get("wall")]
        pool = non_wall if non_wall else list(range(len(self.placed_furniture)))

        def eroded_area(i: int) -> float:
            # navmesh 按 agent 半径 0.3 侵蚀，占地损失 ~ (w+0.6)(d+0.6)
            _o, cls = self.placed_furniture[i]
            sx, _sy, sz, _my = self._class_dims(cls)
            return (sx + 0.6) * (sz + 0.6)

        return max(pool, key=eroded_area)

    # ---- 水平射线找真墙 ----
    def _ground_height(self, x: float, y_from: float, z: float):
        """从 (x, y_from, z) 竖直向下打射线，返回命中点高度（无命中 None）。"""
        out = _cast_down(self.sim, self.hsim, self.mn, x, z, y_from)
        if out is None:
            return None
        (px, py, pz), _n = out
        return py

    def _cast_horizontal(self, x: float, y: float, z: float, ang: float):
        """从 (x,y,z) 沿水平方向 ang 打射线，返回命中水平距离（无命中 None）。"""
        d = self.mn.Vector3(math.sin(ang), 0.0, math.cos(ang))
        ray = self.hsim.geo.Ray(self.mn.Vector3(x, y, z), d)
        try:
            res = self.sim.cast_ray(ray, 3.0)
        except TypeError:
            res = self.sim.cast_ray(ray)
        hits = []
        if hasattr(res, "has_hits"):
            if not res.has_hits():
                return None
            hits = res.hits
        elif hasattr(res, "has_hit"):
            if not res.has_hit:
                return None
            hits = [res]
        if not hits:
            return None
        pt = hits[0].point
        return math.hypot(float(pt[0]) - x, float(pt[2]) - z)

    def _sample_wall_pose(self, cls: str, floors: list[Surface]):
        """真·贴墙放置：地面采样点 + 水平射线找最近的墙/大家具，背靠它放。

        旧启发式「地面块包络边缘 = 墙」是错的——块边缘往往是房间中间的
        聚类切分线或扫描既有家具的遮挡边界，放那里不是悬空就是堵路。
        """
        sx, _sy, sz, _my = self._class_dims(cls)
        half_depth = 0.5 * min(sx, sz)
        weights = [max(s.n_points, 1) for s in floors]
        for _ in range(25):
            floor = self.rng.choices(floors, weights=weights, k=1)[0]
            px, pz = sample_point_on_surface(self.rng, floor, margin=0.35)
            hy = floor.height + 0.5
            best = None
            for k in range(8):
                ang = k * math.pi / 4.0
                dist = self._cast_horizontal(px, hy, pz, ang)
                if dist is not None and (best is None or dist < best[0]):
                    best = (dist, ang)
            if best is None:
                continue
            dist, wall_ang = best
            if dist < half_depth + 0.08 or dist > 1.8:
                continue   # 贴得太死放不下 / 附近没有可靠遮挡面
            push = dist - half_depth - 0.04
            cx = px + math.sin(wall_ang) * push
            cz = pz + math.cos(wall_ang) * push
            yaw = wall_ang + math.pi       # 背对墙、面朝房间
            try:
                snap = self.sim.pathfinder.snap_point(
                    self.mn.Vector3(cx, floor.height, cz))
            except Exception:
                continue
            if any(v != v for v in snap):
                continue
            if (snap[0] - cx) ** 2 + (snap[2] - cz) ** 2 > 1.0:
                continue
            return (cx, floor.height, cz), yaw
        return None, None

    def _sample_floor_pose(self, floor: Surface, against_wall: bool = False):
        # against_wall 参数已废弃：贴墙请用 _sample_wall_pose（水平射线找真墙）
        for _ in range(30):
            px, pz = sample_point_on_surface(self.rng, floor, margin=0.6)
            yaw = self.rng.uniform(0.0, 2.0 * math.pi)
            # 必须在 navmesh 附近（避免放到扫描不可达区域）
            try:
                snap = self.sim.pathfinder.snap_point(
                    self.mn.Vector3(px, floor.height, pz))
            except Exception:
                continue
            if any(v != v for v in snap):
                continue
            dx = snap[0] - px
            dz = snap[2] - pz
            if dx * dx + dz * dz > 0.8 * 0.8:
                continue
            return (px, floor.height, pz), yaw
        return None, None

    # ---- 小物体放置（表面）----
    def place_small_objects(self, num_objects: int,
                            min_spacing: float = 0.12) -> None:
        # 家具 STATIC 后重新探测表面：此时桌面/柜顶进入 "table"/"shelf" 高度带
        surfaces = detect_surfaces(self.sim, self.hsim, self.mn)
        by_band: dict[str, list[Surface]] = {}
        for s in surfaces:
            by_band.setdefault(s.category, []).append(s)
        if not any(by_band.values()):
            log("没有可用表面，小物体放置跳过", "WARN")
            return

        small_classes = [c for c in self.spec.object_classes
                         if not OBJECT_PROFILES.get(c, DEFAULT_PROFILE)["furniture"]]
        placed_positions: list[tuple[float, float, float]] = []
        attempts, max_attempts = 0, num_objects * 50
        while len(self.placed_small) < num_objects and attempts < max_attempts:
            attempts += 1
            cls = self.rng.choice(small_classes)
            if cls not in self.handles:
                continue
            rule = self.rules.get(cls, {})
            wanted_bands = rule.get("on", ["floor"])
            candidates = [s for band in wanted_bands for s in by_band.get(band, [])]
            if not candidates:
                continue
            surface = self.rng.choice(candidates)
            px, pz = sample_point_on_surface(self.rng, surface, margin=0.08)
            # 间距约束，防扎堆
            if any((px - qx) ** 2 + (pz - qz) ** 2 < min_spacing ** 2
                   and abs(surface.height - qy) < 0.3
                   for qx, qy, qz in placed_positions):
                continue
            yaw = self.rng.uniform(-math.pi / 6, math.pi / 6)  # ±30° 自然扰动
            obj = self._spawn(cls, (px, surface.height, pz), yaw, dynamic=True)
            if obj is None:
                continue
            # 落定验证：掉太远（假表面/滚下桌子）判失败移除
            if not self._settle_supported(obj, 1.2, max_fall=0.35):
                self.rigid_mgr.remove_object_by_id(obj.object_id)
                continue
            # 姿态验证：非球形物体落定后明显倾倒（杯口朝侧）也移除重试
            if not OBJECT_PROFILES.get(cls, DEFAULT_PROFILE).get("round"):
                rot = obj.rotation.to_matrix()
                if float(rot[1][1]) < 0.7:
                    self.rigid_mgr.remove_object_by_id(obj.object_id)
                    continue
            t = obj.translation
            placed_positions.append((float(t.x), float(t.y), float(t.z)))
            self.placed_small.append((obj, cls))
        log(f"小物体放置完成: {len(self.placed_small)} 个（{attempts} 次采样）")

    # ---- 汇总导出（Phase 1：interactive_ratio 决定 DYNAMIC 名单）----
    def finalize_instances(self, interactive_ratio: float) -> list[dict[str, Any]]:
        instances: list[dict[str, Any]] = []
        for obj, cls in self.placed_furniture:
            t, q = obj_pose(obj)
            instances.append(make_instance(self.spec, cls, t, q, "STATIC",
                                           rule="furniture"))
        small = list(self.placed_small)
        self.rng.shuffle(small)
        n_dynamic = int(round(len(small) * interactive_ratio))
        dynamic_ids: set[int] = set()
        for i, (obj, cls) in enumerate(small):
            keep_dynamic = i < n_dynamic
            t, q = obj_pose(obj)
            inst = make_instance(
                self.spec, cls, t, q,
                "DYNAMIC" if keep_dynamic else "STATIC",
                rule="on_surface",
                extra={"graspable": keep_dynamic},
            )
            instances.append(inst)
            if keep_dynamic:
                dynamic_ids.add(obj.object_id)
        # 可抓取物体恢复 DYNAMIC；其余小物体 STATIC 进 navmesh
        for obj, cls in small:
            obj.motion_type = (self.hsim.physics.MotionType.DYNAMIC
                               if obj.object_id in dynamic_ids
                               else self.hsim.physics.MotionType.STATIC)
        log(f"motion_type: STATIC={len(instances) - n_dynamic}, DYNAMIC={n_dynamic}")
        return instances


# ---------------------------------------------------------------------------
# 5. 原版随机放置（--no-semantic 回退），物理参数修复仍然生效
# ---------------------------------------------------------------------------

def populate_random(sim, hsim, mn, spec: BuildSpec, num_objects: int, seed: int,
                    settle_seconds: float, min_island_radius: float,
                    interactive_ratio: float) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    obj_tmpl_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()
    handles = resolve_template_handles(spec, obj_tmpl_mgr)
    if not handles:
        log("没有可用 object 类别，跳过 populate", "WARN")
        return []

    placed: list[tuple[Any, str]] = []
    attempts, max_attempts = 0, num_objects * 40
    while len(placed) < num_objects and attempts < max_attempts:
        attempts += 1
        cls = rng.choice(list(handles))
        profile = OBJECT_PROFILES.get(cls, DEFAULT_PROFILE)
        obj = rigid_mgr.add_object_by_template_handle(handles[cls])
        if obj is None:
            continue
        pt = sim.pathfinder.get_random_navigable_point()
        if sim.pathfinder.island_radius(pt) < min_island_radius:
            rigid_mgr.remove_object_by_id(obj.object_id)
            continue
        node = obj.root_scene_node
        bb = hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        obj.translation = mn.Vector3(pt[0], pt[1] + bb.size_y() / 2.0 + 0.03, pt[2])
        obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rng.uniform(0.0, 2.0 * math.pi)), mn.Vector3.y_axis())
        obj.motion_type = hsim.physics.MotionType.DYNAMIC
        if contact_test(sim, obj):
            rigid_mgr.remove_object_by_id(obj.object_id)
            continue
        placed.append((obj, cls, profile))

    log(f"撒入 {len(placed)} 个物体（{attempts} 次采样）")
    settle(sim, settle_seconds)

    rng.shuffle(placed)
    graspables = [(o, c) for (o, c, p) in placed if not p["furniture"]]
    n_dynamic = int(round(len(graspables) * interactive_ratio))
    dynamic_ids = {o.object_id for (o, c) in graspables[:n_dynamic]}

    instances: list[dict[str, Any]] = []
    for obj, cls, profile in placed:
        keep_dynamic = obj.object_id in dynamic_ids
        obj.motion_type = (hsim.physics.MotionType.DYNAMIC if keep_dynamic
                           else hsim.physics.MotionType.STATIC)
        t, q = obj_pose(obj)
        instances.append(make_instance(
            spec, cls, t, q,
            "DYNAMIC" if keep_dynamic else "STATIC",
            rule="random",
            extra={"graspable": keep_dynamic},
        ))
    log(f"motion_type: STATIC={len(instances) - n_dynamic}, DYNAMIC={n_dynamic}")
    return instances


# ---------------------------------------------------------------------------
# 6. Phase 5 — NavMesh 烘焙（先全 STATIC 再烤，恢复 DYNAMIC 在导出前已完成）
# ---------------------------------------------------------------------------

def _add_navramps(sim, hsim, spec: BuildSpec):
    """stage 配对的 <stem>_navramps.glb 作为 STATIC 物体参与烘焙。

    Recast 按 agent 半径侵蚀可行走面，0.28m 细踏步侵蚀后无处可走导致
    楼梯断岛；坡道面把楼梯当成斜坡（游戏行业标准做法）。烘焙后移除，
    不渲染、不进物理、不进数据集。
    """
    src = spec.out_root / "stages" / f"{spec.stage_src.stem}_navramps.glb"
    if not src.exists():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="navramps_"))
    write_json(tmp / "_navramps.object_config.json", {
        "render_asset": str(src),
        "collision_asset": str(src),
        "up": [0.0, 1.0, 0.0],
        "front": [0.0, 0.0, -1.0],
        "units_to_meters": 1.0,
        "mass": 100.0,
        "use_bounding_box_for_collision": False,
        "join_collision_meshes": True,
        "is_collidable": True,
    })
    obj_tmpl_mgr = sim.get_object_template_manager()
    obj_tmpl_mgr.load_configs(str(tmp))
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(
        str(tmp / "_navramps.object_config.json"))
    if obj is None:
        log("navramps 加载失败，楼梯 navmesh 可能断岛", "WARN")
        return None
    # habitat 0.3.1 加载物体资产时按包围盒中心重置局部原点，而 navramps
    # 用 stage 世界坐标建模——须平移回原包围盒中心，否则坡道落在原点处。
    # 注意：必须在设 STATIC 之前平移！先 STATIC 再设 translation 不生效
    # （实测 ramp 留在原点 AABB 中心，段2 坡道悬在客厅上空 y0~1.58，
    #  human 高 1.7 净空不足被拦腰切断，spot 高 0.65 能从坡下穿过故全通）。
    import trimesh
    bb = trimesh.load(str(src)).bounds
    if bb is not None:
        import magnum as _mn
        c = (bb[0] + bb[1]) / 2.0
        obj.translation = _mn.Vector3(float(c[0]), float(c[1]), float(c[2]))
    obj.motion_type = hsim.physics.MotionType.STATIC
    log(f"navmesh 烘焙辅助坡道: {src.name}")
    return obj


def bake_navmeshes(sim, hsim, spec: BuildSpec) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    ramp_obj = _add_navramps(sim, hsim, spec)
    try:
        for name in spec.embodiments:
            preset = EMBODIMENTS[name]
            ns = make_navmesh_settings(hsim, preset, include_static=True)
            ok = recompute_navmesh_compat(sim, hsim, ns, True)
            if not ok:
                log(f"navmesh 重算失败: {name}", "ERROR")
                report[name] = {"ok": False, **preset}
                continue
            path = spec.navmesh_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            sim.pathfinder.save_nav_mesh(str(path))
            area = float(sim.pathfinder.navigable_area)
            report[name] = {"ok": True, "navigable_area_m2": round(area, 3),
                            "file": str(path.relative_to(spec.out_root)), **preset}
            log(f"navmesh [{name:<8}] r={preset['radius']:.2f} h={preset['height']:.2f} "
                f"climb={preset['max_climb']:.2f} -> {area:8.2f} m^2")
    finally:
        if ramp_obj is not None:
            sim.get_rigid_object_manager().remove_object_by_id(ramp_obj.object_id)
    return report


# ---------------------------------------------------------------------------
# 7. Phase 6 — 验证：抽检渲染 / 可交互性 / 稳定性
# ---------------------------------------------------------------------------

def _look_at_yaw(cam, target) -> float:
    dx = float(target[0]) - float(cam[0])
    dz = float(target[2]) - float(cam[2])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.atan2(-dx, -dz)


def _viewpoints(sim, rng, num_views: int, near_radius: float = 2.5):
    rigid_mgr = sim.get_rigid_object_manager()
    targets = []
    for h in rigid_mgr.get_object_handles(""):
        o = rigid_mgr.get_object_by_handle(h)
        if o is not None:
            t = o.translation
            targets.append((float(t.x), float(t.y), float(t.z)))
    for _ in range(num_views):
        if targets:
            import magnum as _mn
            tgt = rng.choice(targets)
            pos = None
            try:
                pos = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=_mn.Vector3(*tgt), radius=near_radius,
                    max_tries=20)
            except Exception:
                pos = None
            if pos is not None and all(p == p for p in pos):
                yield pos, _look_at_yaw(pos, tgt)
                continue
        pos = sim.pathfinder.get_random_navigable_point()
        yield pos, rng.uniform(0.0, 2.0 * math.pi)


def validate(sim, hsim, spec: BuildSpec, num_views: int, seed: int) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        log("缺 numpy/Pillow，跳过抽检图输出", "WARN")
        return {"rendered": 0}

    rng = random.Random(seed)
    out_dir = spec.out_root / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = sim.get_agent(0)

    rendered = 0
    sem_ids_seen: set[int] = set()
    for i, (pos, yaw) in enumerate(_viewpoints(sim, rng, num_views)):
        state = agent.get_state()
        state.position = pos
        state.rotation = np.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        if "rgb" in obs:
            Image.fromarray(obs["rgb"][..., :3].astype(np.uint8)).save(
                out_dir / f"view{i:02d}_rgb.png")
        if "depth" in obs:
            d = obs["depth"]
            dmax = float(d.max()) or 1.0
            Image.fromarray((d / dmax * 255).astype(np.uint8)).save(
                out_dir / f"view{i:02d}_depth.png")
        if "semantic" in obs:
            s = obs["semantic"].astype(np.uint32)
            sem_ids_seen.update(int(v) for v in np.unique(s))
            rgb = np.stack([(s * 131) % 256, (s * 57) % 256, (s * 199) % 256], -1)
            rgb[s == 0] = 0
            Image.fromarray(rgb.astype(np.uint8)).save(
                out_dir / f"view{i:02d}_semantic.png")
        rendered += 1
    fg = sorted(v for v in sem_ids_seen if v != 0)
    log(f"抽检图 {rendered} 组 -> {out_dir}（semantic 前景 id: {fg or '无'}）")
    return {"rendered": rendered, "dir": str(out_dir),
            "semantic_ids_seen": fg}


def verify_interactive(sim, hsim, mn, spec: BuildSpec,
                       max_samples: int = 5, impulse_speed: float = 0.6,
                       test_seconds: float = 0.6) -> dict[str, Any]:
    """对 DYNAMIC 物体施加水平速度，确认能产生位移（可交互），
    测试完恢复原位姿，保证最终场景与 plan 一致。"""
    rigid_mgr = sim.get_rigid_object_manager()
    results: list[dict[str, Any]] = []
    for handle in rigid_mgr.get_object_handles(""):
        obj = rigid_mgr.get_object_by_handle(handle)
        if obj is None or obj.motion_type != hsim.physics.MotionType.DYNAMIC:
            continue
        if len(results) >= max_samples:
            break
        start_t, start_q = obj_pose(obj)
        disp = 0.0
        # 四个水平方向都试：物体可能某侧贴着墙/家具，单方向推不动会误判
        for vx, vz in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            try:
                obj.linear_velocity = mn.Vector3(impulse_speed * vx, 0.0,
                                                 impulse_speed * vz)
            except Exception as exc:  # 老版本没有 velocity setter
                log(f"无法设置速度 ({exc})，改用 contact_test 判可交互", "WARN")
                results.append({"object": handle, "interactive": True,
                                "method": "dynamic_flag_only"})
                break
            settle(sim, test_seconds)
            end_t, _ = obj_pose(obj)
            disp = max(disp, math.dist(start_t, end_t))
            if disp > 0.03:
                break
            # 没推动：回到原位再换方向，避免越推越贴死
            obj.motion_type = hsim.physics.MotionType.KINEMATIC
            obj.translation = mn.Vector3(*start_t)
            obj.rotation = mn.Quaternion(
                mn.Vector3(start_q[1], start_q[2], start_q[3]), start_q[0])
            obj.motion_type = hsim.physics.MotionType.DYNAMIC
        else:
            pass
        if results and results[-1].get("method") == "dynamic_flag_only":
            continue
        results.append({"object": handle, "interactive": disp > 0.03,
                        "displacement_m": round(disp, 4)})
        # 复位
        obj.motion_type = hsim.physics.MotionType.KINEMATIC
        obj.translation = mn.Vector3(*start_t)
        obj.rotation = mn.Quaternion(
            mn.Vector3(start_q[1], start_q[2], start_q[3]), start_q[0])
        try:
            obj.linear_velocity = mn.Vector3(0.0, 0.0, 0.0)
            obj.angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        except Exception:
            pass
        obj.motion_type = hsim.physics.MotionType.DYNAMIC
    n_ok = sum(1 for r in results if r.get("interactive"))
    log(f"可交互验证: {n_ok}/{len(results)} 个 DYNAMIC 物体受力产生位移")
    return {"tested": len(results), "passed": n_ok, "details": results}


def check_stability(sim, hsim, spec: BuildSpec) -> dict[str, Any]:
    """静置后检查 DYNAMIC 物体是否倾倒（up 轴偏离竖直方向 > 45°）。
    球形物体（apple/orange）没有稳定的 up 轴，只检查没掉下表面即可。"""
    rigid_mgr = sim.get_rigid_object_manager()
    tipped: list[str] = []
    checked = 0
    for handle in rigid_mgr.get_object_handles(""):
        obj = rigid_mgr.get_object_by_handle(handle)
        if obj is None or obj.motion_type != hsim.physics.MotionType.DYNAMIC:
            continue
        # 从对象 handle 还原类名查 profile（"objects/orange_:0002" -> "orange"）
        cls = handle.split(":")[0].rstrip("_").split("/")[-1]
        if OBJECT_PROFILES.get(cls, DEFAULT_PROFILE).get("round"):
            continue
        checked += 1
        q = obj.rotation
        rot = q.to_matrix()
        up_y = float(rot[1][1])   # 物体本地 Y 轴在世界中的 y 分量
        if up_y < math.cos(math.radians(45.0)):
            tipped.append(handle)
    if tipped:
        log(f"稳定性检查: {len(tipped)} 个物体倾倒: {tipped}", "WARN")
    else:
        log(f"稳定性检查: {checked} 个 DYNAMIC 物体均保持竖直")
    return {"checked": checked, "tipped": tipped, "ok": not tipped}


# ---------------------------------------------------------------------------
# 8. 放置方案导入/导出（人工干预）
# ---------------------------------------------------------------------------

def export_plan(spec: BuildSpec, instances: list[dict[str, Any]], path: Path,
                y_off: dict[str, float] | None = None) -> None:
    """导出人工可编辑的放置方案（底面高度惯例）。类名从 template_name 还原。"""
    plan = []
    for inst in instances:
        cls = inst["template_name"].split("/")[-1]
        t = list(inst["translation"])
        if y_off is not None:
            t[1] = t[1] + y_off.get(cls, 0.0)   # 原点(中心)高度 -> 底面高度
        plan.append({
            "class": cls,
            "translation": [round(v, 4) for v in t],
            "rotation": [round(v, 6) for v in inst["rotation"]],
            "motion_type": inst["motion_type"],
            "rule": inst.get("user_defined", {}).get("placement_rule", ""),
        })
    write_json(path, {
        "format": "habitat_scene_builder_v2_plan",
        "stage": str(spec.stage_src),
        "scene": spec.scene_name,
        "note": "人工可编辑：改 translation/rotation/motion_type 后用 --plan-in 重建",
        "placements": plan,
    })
    log(f"放置方案已导出 -> {path}（{len(plan)} 个物体，可手工编辑）")


def class_y_offsets(sim, hsim, handles: dict[str, str]) -> dict[str, float]:
    """每类物体碰撞/渲染包围盒的 bb.min.y。

    habitat 加载物体 GLB 时会把网格按 AABB 中心对齐到原点（我们的 GLB 原点在
    底面中心，加载后变成 y[-h/2, +h/2]）。plan 采用"底面高度"人类惯例，
    落位时 origin_y = plan_y - bb_min_y；导出时 plan_y = origin_y + bb_min_y。
    """
    rigid_mgr = sim.get_rigid_object_manager()
    offsets: dict[str, float] = {}
    for cls, handle in handles.items():
        obj = rigid_mgr.add_object_by_template_handle(handle)
        if obj is None:
            continue
        node = obj.root_scene_node
        bb = hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        offsets[cls] = float(bb.min.y)
        rigid_mgr.remove_object_by_id(obj.object_id)
    return offsets


def instantiate_from_plan(sim, hsim, mn, spec: BuildSpec,
                          plan_path: Path,
                          y_off: dict[str, float] | None = None
                          ) -> list[dict[str, Any]]:
    """严格按 plan 落位（plan 用底面高度惯例）。STATIC 精确放置；DYNAMIC 抬高 2cm 短静置。"""
    plan = read_json(plan_path)
    placements = plan.get("placements", [])
    if not placements:
        die(f"plan 里没有 placements: {plan_path}")
    obj_tmpl_mgr = sim.get_object_template_manager()
    handles = resolve_template_handles(spec, obj_tmpl_mgr)
    if y_off is None:
        y_off = class_y_offsets(sim, hsim, handles)
    rigid_mgr = sim.get_rigid_object_manager()

    instances: list[dict[str, Any]] = []
    for item in placements:
        cls = item["class"]
        if cls not in handles:
            log(f"plan 中的类别未注册，跳过: {cls}", "WARN")
            continue
        obj = rigid_mgr.add_object_by_template_handle(handles[cls])
        if obj is None:
            continue
        t = item["translation"]
        q = item["rotation"]  # [w, x, y, z]
        motion = item.get("motion_type", "STATIC")
        # plan_y 是"底面所在高度"——换算到 habitat 的 AABB 中心原点
        base_y = float(t[1]) - y_off.get(cls, 0.0)
        obj.translation = mn.Vector3(float(t[0]), base_y, float(t[2]))
        obj.rotation = mn.Quaternion(
            mn.Vector3(float(q[1]), float(q[2]), float(q[3])), float(q[0]))
        obj.motion_type = hsim.physics.MotionType.KINEMATIC
        if motion == "DYNAMIC":
            obj.translation = mn.Vector3(float(t[0]), base_y + 0.02, float(t[2]))
            obj.motion_type = hsim.physics.MotionType.DYNAMIC
            settle(sim, 0.5)
        else:
            obj.motion_type = hsim.physics.MotionType.STATIC
        tt, qq = obj_pose(obj)
        instances.append(make_instance(spec, cls, tt, qq, motion,
                                       rule=item.get("rule", "plan"),
                                       extra={"graspable": motion == "DYNAMIC"}))
    log(f"按 plan 落位 {len(instances)} 个物体 <- {plan_path}")
    return instances


# ---------------------------------------------------------------------------
# 9. 主流程
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> int:
    rules = SEMANTIC_RULES
    if args.rules:
        rules = read_json(Path(args.rules))
        log(f"使用自定义语义规则库 <- {args.rules}")

    spec = BuildSpec(
        stage_src=Path(args.stage).expanduser().resolve(),
        objects_src=Path(args.objects).expanduser().resolve() if args.objects else None,
        out_root=Path(args.out).expanduser().resolve(),
        dataset_name=args.name,
        scene_name=args.scene or f"{args.name}_000",
        stage_up=args.stage_up,
        object_up=args.object_up,
        units_to_meters=args.units_to_meters,
        stage_collision_asset=(
            Path(args.stage_collision).expanduser().resolve()
            if args.stage_collision else None
        ),
        embodiments=args.embodiments,
        interactive_ratio=args.interactive_ratio,
    )

    for e in spec.embodiments:
        if e not in EMBODIMENTS:
            die(f"未知具身 '{e}'，可选: {', '.join(EMBODIMENTS)}")

    t0 = time.time()
    scaffold(spec)

    report: dict[str, Any] = {
        "version": "v2",
        "mode": ("plan" if args.plan_in else
                 "semantic" if args.semantic else "random"),
        "dataset_config": str(spec.dataset_config_path),
        "scene_instance": str(spec.scene_instance_path),
        "object_classes": spec.object_classes,
        "semantic_id_map": spec.semantic_id_map,
        "interactive_ratio": spec.interactive_ratio,
    }

    if args.scaffold_only:
        log("--scaffold-only，到此为止")
        write_json(spec.out_root / "_preview" / "build_report.json", report)
        _print_summary(spec, report, time.time() - t0)
        return 0

    hsim, mn = _import_habitat()

    # ---- pass 1: 摆物体 + 烤 navmesh ----
    log("打开仿真器（pass 1: place + navmesh）")
    sim = open_sim(hsim, spec, with_sensors=False)
    y_off: dict[str, float] = {}
    try:
        if not bootstrap_navmesh(sim, hsim):
            die("stage 上算不出 navmesh。八成是坐标系不对——试试 --stage-up z，"
                "或者检查 GLB 里有没有实心地面。")

        _handles = resolve_template_handles(spec, sim.get_object_template_manager())
        y_off = class_y_offsets(sim, hsim, _handles)
        if args.plan_in:
            instances = instantiate_from_plan(sim, hsim, mn, spec,
                                              Path(args.plan_in), y_off)
        elif args.semantic:
            placer = SemanticPlacer(sim, hsim, mn, spec, _handles, rules,
                                    seed=args.seed)
            placer.place_furniture(max_per_class=args.max_furniture_per_class,
                                   chair_count=args.num_chairs,
                                   min_nav_area=args.min_nav_area)
            placer.place_small_objects(num_objects=args.num_objects,
                                       min_spacing=args.min_object_distance)
            instances = placer.finalize_instances(spec.interactive_ratio)
        else:
            instances = populate_random(
                sim, hsim, mn, spec,
                num_objects=args.num_objects, seed=args.seed,
                settle_seconds=args.settle_seconds,
                min_island_radius=args.min_island_radius,
                interactive_ratio=spec.interactive_ratio,
            )
        report["placed_objects"] = len(instances)
        report["dynamic_objects"] = sum(
            1 for i in instances if i["motion_type"] == "DYNAMIC")

        # 可交互验证（在转 STATIC 烘焙之前，DYNAMIC 物体已就位）
        if args.verify_interactive:
            report["interactive_check"] = verify_interactive(
                sim, hsim, mn, spec, max_samples=args.verify_samples)
            report["stability_check"] = check_stability(sim, hsim, spec)

        # Phase 5: 烘焙。Recast 只纳入 STATIC 物体与 stage；DYNAMIC 小物件
        # 保持 DYNAMIC 不参与烘焙——桌面上的碗/杯/书若被当成实心柱，会把
        # 桌面/台面区域从 navmesh 里挖掉，实测在客厅→书房门口切断通带。
        navmesh_report = bake_navmeshes(sim, hsim, spec)
        report["navmesh"] = navmesh_report
    finally:
        sim.close()

    baked_ok = [n for n, info in navmesh_report.items() if info.get("ok")]
    registered = register_navmeshes(spec, baked_ok)
    if not registered:
        die("没有任何 navmesh 成功落盘，数据集不可用。")

    write_scene_instance(spec, instances, primary_embodiment=registered[0])
    log(f"scene_instance 写入 {spec.scene_instance_path}")

    plan_path = Path(args.plan_out) if args.plan_out else (
        spec.out_root / "placement_plan.json")
    export_plan(spec, instances, plan_path, y_off)
    report["placement_plan"] = str(plan_path)

    # ---- pass 2: 冷启动重载 + 抽检 ----
    if args.views > 0:
        log("重新打开仿真器（pass 2: validate）")
        sim = open_sim(hsim, spec, with_sensors=True, resolution=args.resolution)
        try:
            reloaded = sim.get_rigid_object_manager().get_num_objects()
            report["validate"] = validate(sim, hsim, spec, args.views, args.seed)
            report["validate"]["reload_ok"] = True
            report["validate"]["reloaded_objects"] = int(reloaded)
            report["validate"]["navmesh_loaded"] = bool(sim.pathfinder.is_loaded)
        finally:
            sim.close()
        expected = report.get("placed_objects", 0)
        if reloaded != expected:
            log(f"重载物体数 {reloaded} != 摆放数 {expected}", "ERROR")
            report["validate"]["reload_ok"] = False
        else:
            log(f"冷启动重载校验通过: {reloaded} 个物体")

    write_json(spec.out_root / "_preview" / "build_report.json", report)
    _print_summary(spec, report, time.time() - t0)
    return 0


def _print_summary(spec: BuildSpec, report: dict[str, Any], elapsed: float) -> None:
    print()
    print("=" * 68)
    print(f"  数据集就绪: {spec.out_root}")
    print(f"  模式 {report.get('mode')}   耗时 {elapsed:.1f}s")
    print("-" * 68)
    print(f"  dataset config : {spec.dataset_config_path.name}")
    print(f"  scene          : {spec.scene_name}")
    print(f"  物体类别        : {len(spec.object_classes)}"
          f"   已摆放: {report.get('placed_objects', 0)}"
          f"   DYNAMIC: {report.get('dynamic_objects', 0)}")
    ic = report.get("interactive_check") or {}
    if ic:
        print(f"  可交互验证      : {ic.get('passed')}/{ic.get('tested')} 通过")
    sc = report.get("stability_check") or {}
    if sc:
        print(f"  稳定性检查      : {'通过' if sc.get('ok') else '有倾倒'}")
    for name, info in (report.get("navmesh") or {}).items():
        if info.get("ok"):
            print(f"  navmesh[{name:<8}]: {info['navigable_area_m2']:>8.2f} m^2")
    v = report.get("validate") or {}
    if v:
        print(f"  重载校验        : {v.get('reloaded_objects')} 个物体"
              f"   semantic id: {v.get('semantic_ids_seen') or '无'}")
    print("-" * 68)
    print("  验收:")
    print(f"    python -m habitat_sim.viewer \\")
    print(f"        --dataset {spec.dataset_config_path} \\")
    print(f"        --scene {spec.scene_name}")
    print("=" * 68)


PRESETS: dict[str, dict[str, Any]] = {}
STAGE_ROOT = os.environ.get("HSB_STAGE_ROOT", "")
OBJECT_ROOT = os.environ.get("HSB_OBJECT_ROOT", "")


def default_out_root() -> Path:
    here = Path(__file__).resolve()
    sim_root = here.parent.parent
    if (sim_root / "data").is_dir():
        return sim_root / "data" / "scene_builder"
    return Path.cwd() / "scene_builder"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一键生产可交互 Habitat SceneDataset（语义放置 + 人工可干预）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", required=True, help="stage 网格 (.glb/.gltf/.ply)")
    p.add_argument("--objects", default=None, help="物体 GLB 所在目录")
    p.add_argument("--out", default=None, help="输出数据集根目录")
    p.add_argument("--name", default="my_dataset", help="数据集名")
    p.add_argument("--scene", default=None, help="场景名，默认 <name>_000")

    p.add_argument("--stage-up", choices=list(UP_VECTORS), default="y")
    p.add_argument("--object-up", choices=list(UP_VECTORS), default="y")
    p.add_argument("--units-to-meters", type=float, default=1.0)
    p.add_argument("--stage-collision", default=None)

    # 放置模式
    p.add_argument("--semantic", dest="semantic", action="store_true", default=True,
                   help="语义规则放置（默认开）")
    p.add_argument("--no-semantic", dest="semantic", action="store_false",
                   help="回退到原版随机撒点")
    p.add_argument("--num-objects", type=int, default=20, help="小物体目标数量")
    p.add_argument("--num-chairs", type=int, default=4)
    p.add_argument("--min-nav-area", type=float, default=4.0,
                   help="家具摆完后的 navmesh 验收面积 m^2（fetch 参数）；"
                        "不足时自动撤除挡路家具。0 = 关闭验收")
    p.add_argument("--max-furniture-per-class", type=int, default=1)
    p.add_argument("--min-object-distance", type=float, default=0.12)
    p.add_argument("--interactive-ratio", type=float, default=0.7,
                   help="小物体中保持 DYNAMIC（可抓取）的比例")
    p.add_argument("--rules", default=None, help="自定义语义规则库 JSON")
    p.add_argument("--settle-seconds", type=float, default=2.0)
    p.add_argument("--min-island-radius", type=float, default=1.5)

    # 人工干预
    p.add_argument("--plan-in", default=None, help="按人工编辑过的放置方案重建")
    p.add_argument("--plan-out", default=None, help="放置方案导出路径")

    # 验证
    p.add_argument("--verify-interactive", dest="verify_interactive",
                   action="store_true", default=True)
    p.add_argument("--no-verify-interactive", dest="verify_interactive",
                   action="store_false")
    p.add_argument("--verify-samples", type=int, default=5)

    p.add_argument("--embodiments", nargs="+", default=["default"],
                   choices=list(EMBODIMENTS))
    p.add_argument("--views", type=int, default=4, help="抽检渲染张数，0 = 跳过")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scaffold-only", action="store_true")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = str(default_out_root() / args.name)
    return args


if __name__ == "__main__":
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    sys.exit(build(parse_args()))
