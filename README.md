# 两层别墅具身智能仿真环境（Habitat-Sim）

按真实户型工程数据参数化生成的两层别墅场景，含楼梯、电梯、真实家具布局
与可交互小物件，支持 spot / fetch / human 三种具身的 NavMesh 导航（跨层连通）。

## 环境规格与信息

- 两层，每层 10×10 m = **100 m²/层**，层高 2.9 m
- **F1 动区**：客厅（沙发贴西墙、茶几居中、电视柜贴东墙、电视对沙发）、
  餐厅（餐桌+4 椅）、厨房（操作台贴北墙）、玄关（鞋柜）、书房（书桌+书架）、
  卫生间、楼梯间、电梯井
- **F2 静区**：主卧（床+双床头柜+衣柜）、次卧、书房、茶室（桌+2椅+边柜）、
  走廊、楼梯间、电梯井
- **楼梯**：L 形直跑 19 踏步 + 90° 转角平台（踏高 0.145 m / 踏面 0.28 m，
  真实住宅标准），platform 宽 2.2 m 保证两梯段共享整条边
- **电梯**：1.4×1.4 m 井道 + 独立轿厢（1.3×1.3 m），脚本升降 F1↔F2
- **71 个物体 / 24 类**，其中 42 个 DYNAMIC 可交互（碗/杯/书/苹果/瓶子…）

布局参考安居客 94㎡ 两室两厅户型的工程尺寸（沙发 3×0.85、餐桌 1.6×0.85、
床 2×2、衣柜 1.8×0.6、床头柜 0.5×0.45 等）。

## 具身与 NavMesh

| 具身  | radius | height | max_climb | 跨层方式 |
|-------|--------|--------|-----------|----------|
| human | 0.30   | 1.70   | 0.20      | 楼梯     |
| spot  | 0.25   | 0.65   | 0.15      | 楼梯     |
| fetch | 0.30   | 1.50   | 0.05      | 电梯（楼梯台缘超过其爬升能力） |

`probe_nav.py` 全链路验收（16 条主链+支路，human/spot 全通）：
客厅→书房→走廊→楼梯→转角平台→顶步→F2 走廊→各房间，跨层爬升 ~2.9 m。

## 文件

```
habitat_scene_builder/
├── habitat_scene_builder_v2.py   # 主 builder（放置 + navmesh 烘焙 + 验证）
├── make_villa_stage.py           # 两层别墅 stage 生成器（含 navramps 坡道）
├── make_villa_plan.py            # 户型布局 plan 生成器（真实尺寸坐标）
├── make_demo_objects.py          # 参数化物体库生成器（24 类真实尺寸）
├── verify_villa.py               # 跨层 navmesh 验收 + 定点巡检渲染图
├── record_villa_tour.py          # 两层巡检 + 电梯 + 推物体演示视频
├── probe_nav.py                  # navmesh 全链路逐段验收
└── record_walkthrough.py         # 通用单层巡检视频（其他场景用）
```

## 快速开始

```bash
PY=/data/hsb/env/bin/python        # habitat-sim 0.3.1 环境

# 1) 生成 stage（含 navramps 辅助坡道）+ 物体库 + 布局 plan
$PY make_villa_stage.py --out ../demo_assets/stages/villa_2f.glb \
    --car-out ../demo_assets/objects/elevator_car.glb \
    --nav-out ../demo_assets/stages/villa_2f_navramps.glb
$PY make_demo_objects.py --out ../demo_assets/objects
$PY make_villa_plan.py --out ../demo_run/villa_2f/placement_plan.json

# 2) 重建数据集（放置 71 物体 + 烘焙三具身 navmesh）
$PY habitat_scene_builder_v2.py \
    --stage ../demo_assets/stages/villa_2f.glb \
    --objects ../demo_assets/objects \
    --out ../demo_run/villa_2f --name villa_2f \
    --plan-in ../demo_run/villa_2f/placement_plan.json \
    --embodiments spot fetch human --views 0

# 3) 验收：跨层连通 + 巡检渲染图
$PY verify_villa.py          # -> _preview/villa_check/*.png
$PY probe_nav.py             # 全链路逐段 find_path

# 4) 巡检视频（F1→爬楼→F2→电梯→推物体）
$PY record_villa_tour.py     # -> _preview/villa_tour.mp4
```

## 人工干预（改布局）

`placement_plan.json` 是人工可编辑的放置方案（底面高度惯例，四元数
[w,x,y,z]）。改任意物体的 `translation`/`rotation`/`motion_type` 后重建：

```bash
$PY habitat_scene_builder_v2.py --stage ... --objects ... \
    --out ../demo_run/villa_2f --name villa_2f \
    --plan-in <你改的 plan>.json --embodiments spot fetch human --views 0
```

## 关键技术点

- **NavRamp 坡道辅助**：Recast 按 agent 半径（0.25~0.30 m）侵蚀可行走面，
  0.28 m 细踏步被侵蚀后无处可走导致楼梯断岛。铺 27~30° 坡道面把楼梯当
  斜坡（< 各具身 max_slope），烘焙后移除（不渲染/不进物理/不进数据集）。
  游戏行业标准做法。见 `make_villa_stage.build_navramps()`。
- **坡道底端衔接板**：坡道严格过踏步前缘顶线时，其 y=0 穿出点必然落在
  踏步北侧，入口处坡道西侧面（高 0.145 m）侵蚀后与地面形成无 mesh 缝。
  加一块与入口同高（y=0.145）的水平衔接板把坡道西缘与地面缝合。
- **`_add_navramps` 平移顺序**：habitat 0.3.1 加载物体资产按包围盒中心
  重置局部原点，而 navramps 用 stage 世界坐标建模，须平移回原包围盒中心。
  **必须先设 translation 再设 STATIC**——先 STATIC 后 translation 不生效，
  坡道会留在原点（悬在客厅上空 y0~1.58，把 human 的客厅拦腰切断）。
- **navmesh cell 量化**：`NAVMESH_CELL_HEIGHT=0.05`。0.20 会把 walkableHeight
  量化成 1.8 m 使 2.1 m 门洞断连，0.10 时 spot 的 max_climb=0.15 只剩 1
  voxel 爬不上坡道接缝；0.05 时 0.15 m→3 voxel，坡道/门洞全通。

## 已验证

- 跨层连通：human/spot 16 条链路全通（客厅→楼梯→F2 各房间）
- 布局合理：沙发靠墙对电视、餐桌椅围绕贴地、小物件在台面不悬空
- 可交互：42 个 DYNAMIC 物体受力可推动（推碗演示）
- 稳定性：40 个非球形 DYNAMIC 静置保持竖直、2 个球形在表面
