# habitat_scene_builder —— 自建 Habitat 场景

把「一个 stage GLB（房间）+ 一堆 object GLB（物体）」一键生产成完整的
Habitat SceneDataset：配置、物体摆放、逐具身 navmesh、抽检图，一条命令全出。

## 一键跑（aslam）

```bash
export HABITAT_PYTHON=~/sjj_ws/probes/semantic_component_v23_trace_code_20260728_01/.envs/habitat033_clean/bin/python
cd simulation_habitat
./run_scene_builder.sh demo
```

约 30 秒出结果，产物落在 `simulation_habitat/data/scene_builder/demo/`
（该目录已被 gitignore，GLB 不会进仓库）。

可用预设：`demo`（van-gogh-room）、`apartment`、`castle`。

## 用你自己的资产

```bash
./run_scene_builder.sh \
    --stage /path/to/room.glb \
    --objects /path/to/objects_dir/ \
    --out data/scene_builder/my_scene --name my_scene \
    --num-objects 25 \
    --embodiments spot fetch human
```

**Blender 导出的 GLB 是 Z-up**，必须加 `--stage-up z`，否则整个房间会躺倒、
navmesh 直接算出 0 面积。

没装 habitat_sim 时可以只生成 JSON 骨架：加 `--scaffold-only`。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `HABITAT_PYTHON` | habitat_sim 解释器（优先级最高） | — |
| `HABITAT_VENV` | venv 目录 | `${ROOT}/.venv` |
| `HSB_STAGE_ROOT` | preset 的 stage 资产根目录 | `~/vlfm/data/scene_datasets/habitat-test-scenes` |
| `HSB_OBJECT_ROOT` | preset 的 object 资产根目录 | `~/ANM/activeINR/habitat-sim/data/test_assets/objects` |

换机器只需改后两个变量，preset 本身不用动。

## 流水线

```
scaffold   纯文件操作，不开仿真
           写 stage_config / object_config / dataset_config / 空 scene_instance
           拷 GLB，给每个物体类别分配 semantic_id（从 1 开始，0 留给背景）
   ↓
populate   开仿真（无传感器，快）
           宽松参数烤 bootstrap navmesh → 在其上随机采点
           滤掉半径 < --min-island-radius 的孤岛（桌面、阳台）
           抬高到不穿模 → 随机绕 Y 转向 → contact_test 撞了就丢弃重采
           物理静置 --settle-seconds 秒 → 转 STATIC → 导出位姿
   ↓
navmesh    每个具身按自己的 radius/height/max_climb 各烤一份
           物体已是 STATIC，会被计入体素化
           落盘后才回填 dataset config 的 navmesh_instances
   ↓
validate   关掉仿真、从 dataset config 冷启动重新加载
           挑已放置物体 → 附近取可行走点 → 转向它 → 渲 RGB-D + semantic
           校验重载物体数 == 摆放数
```

**冷启动重载是关键设计**：pass 2 不复用 pass 1 的内存状态，等于每次都自证
「产物真的能被别人加载」，而不只是「我内存里摆好了」。

## 产物结构

```
<out>/
  <name>.scene_dataset_config.json
  stages/      <stage>.glb  +  <stage>.stage_config.json
  objects/     *.glb        +  *.object_config.json   (含 semantic_id)
  scenes/      <scene>.scene_instance.json
  navmeshes/   <scene>__<embodiment>.navmesh          (每个具身一份)
  semantic_id_map.json                                (类别名 -> semantic_id)
  _preview/    rgb / depth / semantic 抽检图 + build_report.json
```

## 具身预设

`EMBODIMENTS` 里那几组 radius / height / max_climb **是起步值，不是权威参数**。
`height` 指「需要的垂直净空」，不是机器人总高。上线前请按真实机器人的
URDF / 规格书改掉。

实测 van-gogh-room 上的差异（同一场景、同样 12 个物体）：

| 具身 | radius | height | max_climb | 可行走面积 |
|---|---|---|---|---|
| spot | 0.25 | 0.65 | 0.15 | 5.89 m² |
| human | 0.30 | 1.70 | 0.20 | 5.25 m² |
| fetch | 0.30 | 1.50 | 0.05 | 3.01 m² |

fetch 最小是因为 `max_climb=0.05` 最严，跨不过门槛类小起伏。

## stage GLB 从哪来

脚本**不造几何体**，房间必须你提供。三条来源：

1. **Blender 手工建模** → 导出 GLB（记得 `--stage-up z`）
2. **真实空间扫描** → 手机 LiDAR / Polycam / 摄影测量出 GLB
3. **现成数据集** → HM3D / MP3D / Replica 里挑一个

硬要求只有一个：**网格得有实心地面**。navmesh 靠体素化地面算出来，
只有墙没有地板的模型会直接算出 0 面积。

## 已知边界

- 摆放是 **navmesh 随机撒 + 物理静置**，只保证不穿模、不悬空、不在孤岛上。
  它不做语义化布置（不会把椅子摆到桌边、杯子放到台面）。
- `use_bounding_box_for_collision: true` 让动力学又快又稳，代价是凹形物体
  （碗、椅子）的碰撞体会失真。需要精确碰撞就在 object_config 里关掉。
- stage 默认拿高模当碰撞体。有低模强烈建议传 `--stage-collision`。
