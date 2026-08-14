#!/usr/bin/env python3
"""
habitat_scene_builder.py
========================
一键把「一个 stage GLB + 一堆 object GLB」生产成完整的 Habitat SceneDataset。

产出：
    <out>/
      <name>.scene_dataset_config.json
      stages/      <stage>.glb  +  <stage>.stage_config.json
      objects/     *.glb        +  *.object_config.json   (含 semantic_id)
      scenes/      <scene>.scene_instance.json
      navmeshes/   <scene>__<embodiment>.navmesh          (每个具身一份)
      semantic_id_map.json                                (类别名 -> semantic_id)
      _preview/    rgb / depth / semantic 抽检图 + build_report.json

流水线：
    scaffold  -> 写全部 JSON 配置，不依赖 habitat_sim
    populate  -> 起仿真，navmesh 采样撒物体，物理静置，转 STATIC
    navmesh   -> 按每个具身的 radius/height/max_climb 重算并保存
    validate  -> 从 dataset config 重新加载，渲染 RGB-D+semantic 抽检

用法：
    # 一键预设（aslam 上现成资产）
    python habitat_scene_builder.py --preset demo

    # 完整流水线
    python habitat_scene_builder.py \
        --stage assets/room.glb \
        --objects assets/objects/ \
        --out data/my_dataset --name my_dataset \
        --num-objects 25 \
        --embodiments spot fetch human

    # 只生成配置骨架（没装 habitat_sim 也能跑）
    python habitat_scene_builder.py --stage assets/room.glb --out data/my_dataset --scaffold-only

    # 资产坐标系是 Blender 默认的 Z-up
    python habitat_scene_builder.py --stage assets/room.glb --out data/x --stage-up z

参考：
    SceneDataset JSON 规范  https://aihabitat.org/docs/habitat-sim/attributesJSON.html
    NavMesh 参数教程        habitat-sim/examples/tutorials/nb_python/ECCV_2020_Navigation.py
    物体采样摆放            habitat-sim/examples/tutorials/nb_python/ECCV_2020_Interactivity.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 具身预设
# ---------------------------------------------------------------------------
# !! 这些是起步值，不是权威参数 !!
# radius 取机器人底盘外接圆半径，height 取「能通过的最低净空」（不是机器人总高，
# 而是它需要的垂直空隙），max_climb 取能跨过的台阶高度。
# 上线前请按你真实机器人的 URDF / 规格书改掉。
EMBODIMENTS: dict[str, dict[str, float]] = {
    "default": dict(radius=0.10, height=1.50, max_climb=0.20, max_slope=45.0),
    "human":   dict(radius=0.30, height=1.70, max_climb=0.20, max_slope=45.0),
    "fetch":   dict(radius=0.30, height=1.50, max_climb=0.05, max_slope=30.0),
    "stretch": dict(radius=0.30, height=1.40, max_climb=0.05, max_slope=30.0),
    "spot":    dict(radius=0.25, height=0.65, max_climb=0.15, max_slope=35.0),
    "drone":   dict(radius=0.20, height=0.30, max_climb=1.00, max_slope=85.0),
}

# 体素化精度：cell_size 越小 navmesh 越准也越慢。0.05 是常用折中。
NAVMESH_CELL_SIZE = 0.05
NAVMESH_CELL_HEIGHT = 0.20

UP_VECTORS = {
    "y": ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),   # glTF / Habitat 原生
    "z": ([0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),   # Blender / 多数 CAD 导出
    # front 只影响绕 up 轴的朝向；模型躺倒是 up 错了，模型转向不对才是 front 错了。
}

MESH_EXTS = (".glb", ".gltf", ".ply", ".obj")

# ---------------------------------------------------------------------------
# 一键预设
# ---------------------------------------------------------------------------
# 资产根目录走环境变量，preset 里只记文件名 —— 换机器改这两个变量即可，
# 不必改代码。默认值是 aslam 上已有资产的位置。
STAGE_ROOT = os.environ.get(
    "HSB_STAGE_ROOT", "~/vlfm/data/scene_datasets/habitat-test-scenes")
OBJECT_ROOT = os.environ.get(
    "HSB_OBJECT_ROOT", "~/ANM/activeINR/habitat-sim/data/test_assets/objects")

# preset 只提供「默认值」，命令行显式传参一律覆盖它。
PRESETS: dict[str, dict[str, Any]] = {
    "demo": dict(
        stage="van-gogh-room.glb", objects="", name="demo",
        num_objects=12, embodiments=["spot", "fetch", "human"], views=6,
    ),
    "apartment": dict(
        stage="apartment_1.glb", objects="", name="apartment",
        num_objects=25, embodiments=["spot", "fetch", "human"], views=6,
    ),
    "castle": dict(
        stage="skokloster-castle.glb", objects="", name="castle",
        num_objects=30, embodiments=["spot", "human"], views=6,
    ),
}


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {level:<5} {msg}", flush=True)


def die(msg: str) -> None:
    log(msg, "FATAL")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. SCAFFOLD —— 生成全部 JSON 配置（纯文件操作，无需 habitat_sim）
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

    # 运行期填充
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
        """scene_instance 里引用 stage 用的 handle = 相对 dataset config 的路径。"""
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


def scaffold(spec: BuildSpec) -> BuildSpec:
    log(f"scaffold -> {spec.out_root}")
    for sub in ("stages", "objects", "scenes", "navmeshes", "_preview"):
        (spec.out_root / sub).mkdir(parents=True, exist_ok=True)

    # ---- stage ----
    if not spec.stage_src.exists():
        die(f"stage 资产不存在: {spec.stage_src}")
    stage_dst = spec.out_root / "stages" / spec.stage_src.name
    if stage_dst.resolve() != spec.stage_src.resolve():
        shutil.copy2(spec.stage_src, stage_dst)

    up, front = UP_VECTORS[spec.stage_up]
    stage_cfg: dict[str, Any] = {
        "render_asset": stage_dst.name,
        # 高模当碰撞体会拖慢一切。有低模就传 --stage-collision。
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
    write_json(spec.out_root / "stages" / f"{spec.stage_src.stem}.stage_config.json", stage_cfg)

    # ---- objects ----
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
        # semantic_id 从 1 开始，0 保留给「未标注 / 背景」
        for idx, mesh in enumerate(meshes, start=1):
            cls = mesh.stem
            dst = spec.out_root / "objects" / mesh.name
            if dst.resolve() != mesh.resolve():
                shutil.copy2(mesh, dst)
            write_json(
                spec.object_config_path(cls),
                {
                    "render_asset": dst.name,
                    "collision_asset": dst.name,
                    "up": up_o,
                    "front": front_o,
                    "units_to_meters": spec.units_to_meters,
                    "mass": 1.0,
                    "friction_coefficient": 0.5,
                    "restitution_coefficient": 0.2,
                    "margin": 0.01,
                    # true = 用包围盒当碰撞体。动力学快很多也稳很多，
                    # 代价是凹形物体（碗、椅子）会失真。按需改。
                    "use_bounding_box_for_collision": True,
                    "join_collision_meshes": True,
                    "semantic_id": idx,
                    "shader_type": "material",
                    "user_defined": {"class_name": cls},
                },
            )
            classes.append(cls)
            semantic_map[cls] = idx

    spec.object_classes = classes
    spec.semantic_id_map = semantic_map
    write_json(spec.out_root / "semantic_id_map.json",
               {"background": 0, **semantic_map})

    # ---- dataset config ----
    # 注意：这里**不写** navmesh_instances。navmesh 文件此刻还不存在，
    # 提前登记会让 habitat 在 pass 1 打开时刷一堆 validateMap [Error]。
    # 等 bake_navmeshes 落盘后再由 register_navmeshes() 回填。
    write_json(spec.dataset_config_path, build_dataset_config(spec, baked=[]))

    # ---- 先写一份空的 scene_instance，保证 scaffold-only 也能直接加载 ----
    if not spec.scene_instance_path.exists():
        write_scene_instance(spec, object_instances=[])

    log(f"scaffold 完成: stage=1, object 类别={len(classes)}, "
        f"具身={len(spec.embodiments)}")
    return spec


def build_dataset_config(spec: BuildSpec, baked: list[str]) -> dict[str, Any]:
    """组装 dataset config。baked 里只放**磁盘上确实存在**的 navmesh 具身名。"""
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
    """navmesh 落盘后回填 dataset config。只登记文件真的在的那些。"""
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
    """手写 scene_instance.json。

    也可以用 sim.save_current_scene_config() 让 habitat 自己导出，但那样
    navmesh_instance / semantic_scene_instance 还得回头补，且不同版本行为
    有差异，所以这里自己写，完全可控。
    """
    payload: dict[str, Any] = {
        "stage_instance": {"template_name": spec.stage_handle},
        "default_lighting": "",
        "object_instances": object_instances,
        "articulated_object_instances": [],
    }
    # primary_embodiment=None 表示还没烤 navmesh，此时不能写 navmesh_instance，
    # 否则 pass 1 打开仿真器会因为找不到 .navmesh 文件报错。
    if primary_embodiment is not None:
        payload["navmesh_instance"] = spec.navmesh_handle(primary_embodiment)
    write_json(spec.scene_instance_path, payload)


# ---------------------------------------------------------------------------
# 2. 仿真侧：撒物体 + 烤 navmesh + 抽检
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
    """物体当前位姿是否与场景/其他物体相撞。

    habitat-sim 0.3.x 把 contact_test 挪到了 ManagedRigidObject 上，
    Simulator.contact_test 被删除；0.2.x 只有 Simulator 那个。两边都兼容。
    """
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
    # 新版把 include_static_objects 挂在 settings 上，旧版是 recompute 的第三个参数。
    if hasattr(ns, "include_static_objects"):
        ns.include_static_objects = include_static
    return ns


def recompute_navmesh_compat(sim, hsim, settings, include_static: bool) -> bool:
    """兼容新旧两种 recompute_navmesh 签名。"""
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
            s.position = [0.0, 1.2, 0.0]
            # 略微俯视。物体大多贴地，纯水平视角很容易整张图只有墙。
            s.orientation = [-0.30, 0.0, 0.0]
            specs.append(s)
    agent_cfg.sensor_specifications = specs
    return hsim.Simulator(hsim.Configuration(sim_cfg, [agent_cfg]))


def bootstrap_navmesh(sim, hsim) -> bool:
    """撒物体之前得先有 navmesh 才能采样落点。用宽松参数尽量多覆盖地面。"""
    loose = dict(radius=0.05, height=0.80, max_climb=0.30, max_slope=45.0)
    ns = make_navmesh_settings(hsim, loose, include_static=False)
    ok = recompute_navmesh_compat(sim, hsim, ns, False)
    if not ok or not sim.pathfinder.is_loaded:
        return False
    log(f"bootstrap navmesh 可行走面积 = {sim.pathfinder.navigable_area:.2f} m^2")
    return True


def resolve_template_handles(spec: BuildSpec, obj_tmpl_mgr) -> dict[str, str]:
    """类别名 -> 模板 handle。

    get_template_handles() 是**子串**匹配：类别名 'box' 会同时命中
    'nested_box' 和 'transform_box'，直接取 found[0] 就是掷骰子。
    所以先按绝对路径精确比对，匹配不上才退回子串并警告。
    """
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
            # handle 不是绝对路径的老版本会走到这里
            handles[cls] = found[0]
            if len(found) > 1:
                log(f"'{cls}' 子串命中 {len(found)} 个模板，取 {found[0]}", "WARN")
    return handles


def populate(
    sim, hsim, mn, spec: BuildSpec, num_objects: int, seed: int,
    settle_seconds: float = 2.0, min_island_radius: float = 1.5,
) -> list[dict[str, Any]]:
    """navmesh 采样 -> 抬高到不穿模 -> 物理静置 -> 转 STATIC -> 导出实例列表。"""
    rng = random.Random(seed)
    if not spec.object_classes:
        log("没有可用 object 类别，跳过 populate", "WARN")
        return []

    obj_tmpl_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()

    handles_by_class = resolve_template_handles(spec, obj_tmpl_mgr)
    if not handles_by_class:
        return []

    placed: list[tuple[Any, str]] = []  # (object, class_name)
    attempts, max_attempts = 0, num_objects * 40
    while len(placed) < num_objects and attempts < max_attempts:
        attempts += 1
        cls = rng.choice(list(handles_by_class))
        obj = rigid_mgr.add_object_by_template_handle(handles_by_class[cls])
        if obj is None:
            continue

        pt = sim.pathfinder.get_random_navigable_point()
        # 过滤掉桌面顶部、阳台之类的孤立小岛
        if sim.pathfinder.island_radius(pt) < min_island_radius:
            rigid_mgr.remove_object_by_id(obj.object_id)
            continue

        node = obj.root_scene_node
        bb = hsim.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        obj.translation = mn.Vector3(pt[0], pt[1] + bb.size_y() / 2.0 + 0.03, pt[2])
        obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rng.uniform(0.0, 2.0 * math.pi)), mn.Vector3.y_axis()
        )
        obj.motion_type = hsim.physics.MotionType.DYNAMIC

        if contact_test(sim, obj):
            rigid_mgr.remove_object_by_id(obj.object_id)
            continue
        placed.append((obj, cls))

    log(f"撒入 {len(placed)} 个物体（{attempts} 次采样）")

    # 静置：让所有东西落到实处，避免导出时坐标悬空
    steps = int(settle_seconds * 60)
    for _ in range(steps):
        sim.step_physics(1.0 / 60.0)

    # 转 STATIC，这样它们才会被计入后面的 navmesh 体素化
    instances: list[dict[str, Any]] = []
    for obj, cls in placed:
        obj.motion_type = hsim.physics.MotionType.STATIC
        t = obj.translation
        q = obj.rotation
        instances.append({
            "template_name": spec.object_handle(cls),
            "translation": [float(t.x), float(t.y), float(t.z)],
            "rotation": [float(q.scalar), float(q.vector.x),
                         float(q.vector.y), float(q.vector.z)],
            "motion_type": "STATIC",
            "translation_origin": "COM",
            "uniform_scale": 1.0,
        })
    return instances


def bake_navmeshes(sim, hsim, spec: BuildSpec) -> dict[str, dict[str, Any]]:
    """按每个具身的 radius/height/max_climb 各烤一份 navmesh 并落盘。"""
    report: dict[str, dict[str, Any]] = {}
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
    return report


def _look_at_yaw(cam, target) -> float:
    """agent 朝向沿 -Z，绕 Y 转 yaw 后 forward = (-sin y, 0, -cos y)。
    要让 forward 对上 (dx, dz)，解出 yaw = atan2(-dx, -dz)。"""
    dx = float(target[0]) - float(cam[0])
    dz = float(target[2]) - float(cam[2])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.atan2(-dx, -dz)


def _viewpoints(sim, rng, num_views: int, near_radius: float = 2.5):
    """生成抽检视角。

    纯随机取点常常整张图只有墙，semantic 抽检图全 0，看起来像语义坏了。
    所以优先「挑一个已放置物体 -> 在它附近取可行走点 -> 转向它」，
    场景里没有物体时才退回纯随机。
    """
    rigid_mgr = sim.get_rigid_object_manager()
    targets = []
    for h in rigid_mgr.get_object_handles(""):
        o = rigid_mgr.get_object_by_handle(h)
        if o is not None:
            t = o.translation
            targets.append((float(t.x), float(t.y), float(t.z)))

    for _ in range(num_views):
        if targets:
            tgt = rng.choice(targets)
            pos = None
            try:
                pos = sim.pathfinder.get_random_navigable_point_near(
                    circle_center=tgt, radius=near_radius, max_tries=20)
            except Exception:
                pos = None
            # 采样失败会返回 NaN，得当作失败处理
            if pos is not None and all(p == p for p in pos):
                yield pos, _look_at_yaw(pos, tgt)
                continue
        pos = sim.pathfinder.get_random_navigable_point()
        yield pos, rng.uniform(0.0, 2.0 * math.pi)


def validate(sim, hsim, spec: BuildSpec, num_views: int, seed: int) -> dict[str, Any]:
    """从 navmesh 上取点渲染 RGB-D + semantic，确认能真的跑起来。"""
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
            # 简单调色，只为肉眼确认标签分布，不是正式可视化
            rgb = np.stack([(s * 131) % 256, (s * 57) % 256, (s * 199) % 256], -1)
            rgb[s == 0] = 0
            Image.fromarray(rgb.astype(np.uint8)).save(
                out_dir / f"view{i:02d}_semantic.png")
        rendered += 1

    fg = sorted(v for v in sem_ids_seen if v != 0)
    log(f"抽检图 {rendered} 组 -> {out_dir}（semantic 前景 id: {fg or '无'}）")
    return {"rendered": rendered, "dir": str(out_dir),
            "semantic_ids_seen": fg}


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> int:
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
    )

    for e in spec.embodiments:
        if e not in EMBODIMENTS:
            die(f"未知具身 '{e}'，可选: {', '.join(EMBODIMENTS)}")

    t0 = time.time()
    scaffold(spec)

    report: dict[str, Any] = {
        "dataset_config": str(spec.dataset_config_path),
        "scene_instance": str(spec.scene_instance_path),
        "object_classes": spec.object_classes,
        "semantic_id_map": spec.semantic_id_map,
    }

    if args.scaffold_only:
        log("--scaffold-only，到此为止")
        write_json(spec.out_root / "_preview" / "build_report.json", report)
        _print_summary(spec, report, time.time() - t0)
        return 0

    hsim, mn = _import_habitat()

    # ---- pass 1: 摆物体 + 烤 navmesh ----
    log("打开仿真器（pass 1: populate + navmesh）")
    sim = open_sim(hsim, spec, with_sensors=False)
    try:
        if not bootstrap_navmesh(sim, hsim):
            die("stage 上算不出 navmesh。八成是坐标系不对——试试 --stage-up z，"
                "或者检查 GLB 里有没有实心地面。")

        instances = populate(
            sim, hsim, mn, spec,
            num_objects=args.num_objects, seed=args.seed,
            settle_seconds=args.settle_seconds,
            min_island_radius=args.min_island_radius,
        )
        report["placed_objects"] = len(instances)
        navmesh_report = bake_navmeshes(sim, hsim, spec)
        report["navmesh"] = navmesh_report
    finally:
        sim.close()

    # navmesh 落盘后才回填 dataset config，避免 pass 1 引用不存在的文件
    baked_ok = [n for n, info in navmesh_report.items() if info.get("ok")]
    registered = register_navmeshes(spec, baked_ok)
    if not registered:
        die("没有任何 navmesh 成功落盘，数据集不可用。")

    write_scene_instance(spec, instances, primary_embodiment=registered[0])
    log(f"scene_instance 写入 {spec.scene_instance_path}")

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
        else:
            log(f"冷启动重载校验通过: {reloaded} 个物体")

    write_json(spec.out_root / "_preview" / "build_report.json", report)
    _print_summary(spec, report, time.time() - t0)
    return 0


def _print_summary(spec: BuildSpec, report: dict[str, Any], elapsed: float) -> None:
    print()
    print("=" * 68)
    print(f"  数据集就绪: {spec.out_root}")
    print(f"  耗时 {elapsed:.1f}s")
    print("-" * 68)
    print(f"  dataset config : {spec.dataset_config_path.name}")
    print(f"  scene          : {spec.scene_name}")
    print(f"  物体类别        : {len(spec.object_classes)}"
          f"   已摆放: {report.get('placed_objects', 0)}")
    for name, info in (report.get("navmesh") or {}).items():
        if info.get("ok"):
            print(f"  navmesh[{name:<8}]: {info['navigable_area_m2']:>8.2f} m^2")
    v = report.get("validate") or {}
    if v:
        print(f"  重载校验        : {v.get('reloaded_objects')} 个物体"
              f"   semantic id: {v.get('semantic_ids_seen') or '无'}")
    print("-" * 68)
    print("  验收:")
    print(f"    python examples/viewer.py \\")
    print(f"        --dataset {spec.dataset_config_path} \\")
    print(f"        --scene {spec.scene_name}")
    print("=" * 68)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """把 preset 填进没有显式传参的位置。显式参数永远优先。"""
    if args.preset:
        if args.preset not in PRESETS:
            die(f"未知 preset '{args.preset}'，可选: {', '.join(PRESETS)}")
        p = PRESETS[args.preset]
        stage_root = Path(STAGE_ROOT).expanduser()
        object_root = Path(OBJECT_ROOT).expanduser()
        if args.stage is None:
            args.stage = str(stage_root / p["stage"])
        if args.objects is None and "objects" in p:
            # preset 里 objects 为空串 = 直接用 OBJECT_ROOT 整个目录
            sub = p["objects"]
            args.objects = str(object_root / sub if sub else object_root)
        if args.name is None:
            args.name = p["name"]
        if args.num_objects is None:
            args.num_objects = p["num_objects"]
        if args.embodiments is None:
            args.embodiments = list(p["embodiments"])
        if args.views is None:
            args.views = p["views"]
        if args.out is None:
            args.out = str(default_out_root() / args.name)

    # preset 没覆盖到的，落到硬默认值
    if args.stage is None:
        die("必须给 --stage，或者用 --preset（可选: " + ", ".join(PRESETS) + "）")
    if args.name is None:
        args.name = "my_dataset"
    if args.out is None:
        args.out = str(default_out_root() / args.name)
    if args.num_objects is None:
        args.num_objects = 20
    if args.embodiments is None:
        args.embodiments = ["default"]
    if args.views is None:
        args.views = 4
    return args


def default_out_root() -> Path:
    """默认输出到 simulation_habitat/data/scene_builder/（该目录已被 gitignore）。"""
    here = Path(__file__).resolve()
    sim_root = here.parent.parent          # scripts/ -> simulation_habitat/
    if (sim_root / "data").is_dir():
        return sim_root / "data" / "scene_builder"
    return Path.cwd() / "scene_builder"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一键生产 Habitat SceneDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # preset 只提供默认值；下面这些默认全是 None，用来区分「没传」和「传了默认值」
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="一键预设，资产根目录由 HSB_STAGE_ROOT / HSB_OBJECT_ROOT 指定")
    p.add_argument("--stage", default=None, help="stage 网格 (.glb/.gltf/.ply)")
    p.add_argument("--objects", default=None, help="物体 GLB 所在目录")
    p.add_argument("--out", default=None, help="输出数据集根目录")
    p.add_argument("--name", default=None, help="数据集名")
    p.add_argument("--scene", default=None, help="场景名，默认 <name>_000")

    p.add_argument("--stage-up", choices=list(UP_VECTORS), default="y",
                   help="stage 资产的 up 轴。Blender 导出通常是 z")
    p.add_argument("--object-up", choices=list(UP_VECTORS), default="y")
    p.add_argument("--units-to-meters", type=float, default=1.0)
    p.add_argument("--stage-collision", default=None,
                   help="stage 的低模碰撞网格，强烈建议提供")

    p.add_argument("--num-objects", type=int, default=None)
    p.add_argument("--settle-seconds", type=float, default=2.0)
    p.add_argument("--min-island-radius", type=float, default=1.5,
                   help="过滤掉半径小于此值的 navmesh 孤岛")

    p.add_argument("--embodiments", nargs="+", default=None,
                   choices=list(EMBODIMENTS),
                   help="为哪些具身各烤一份 navmesh")

    p.add_argument("--views", type=int, default=None, help="抽检渲染张数，0 = 跳过")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scaffold-only", action="store_true",
                   help="只生成 JSON 配置，不启动 habitat_sim")
    return apply_preset(p.parse_args(argv))


if __name__ == "__main__":
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    sys.exit(build(parse_args()))
