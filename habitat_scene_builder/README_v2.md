# Habitat 可交互场景生成器 v2 — 使用说明

> 相对原版的改动对应《Habitat可交互语义场景生成器_技术方案》Phase 0-6：
> 修复物体物理、可抓取物体保持 DYNAMIC、射线表面检测 + 语义规则放置、
> NavMesh 烘焙流程修正、可交互/稳定性自动验证，并提供人工干预接口。

## 文件

| 文件 | 说明 |
|---|---|
| `habitat_scene_builder.py` | 同学原版（保留未动） |
| `habitat_scene_builder_v2.py` | 改进版主程序 |
| `make_demo_objects.py` | 参数化生成 18 种家具/小物体 GLB（无需下载资产） |
| `run_v2_demo.sh` | 端到端演示（真实 MP3D 扫描 + 语义放置 + 验证） |

## 快速开始

```bash
# 0. 生成演示物体资产（只需一次）
python make_demo_objects.py --out ../demo_assets/objects

# 1. 语义模式一键生成可交互场景
python habitat_scene_builder_v2.py \
    --stage /path/to/real_scan.glb \
    --objects ../demo_assets/objects \
    --out ../demo_run/my_scene --name my_scene \
    --num-objects 12 --interactive-ratio 0.7 \
    --embodiments spot fetch human --views 6

# 2. 看结果
cat ../demo_run/my_scene/_preview/build_report.json   # 验证报告
ls  ../demo_run/my_scene/_preview/                    # RGB/深度/语义抽检图
```

## 人工干预工作流（核心）

不做「纯 AI 生成不可控」——自动放置只是草稿，人可以改：

```bash
# 1) 自动生成并导出放置方案（默认也会写 <out>/placement_plan.json）
python habitat_scene_builder_v2.py --stage s.glb --objects objs/ \
    --out run1 --name run1 --plan-out my_plan.json

# 2) 手工编辑 my_plan.json：每个物体的 class / translation / rotation /
#    motion_type 都在里面，改位置、删物体、把 STATIC 改 DYNAMIC 都可以

# 3) 按改过的方案精确重建（跳过自动放置）
python habitat_scene_builder_v2.py --stage s.glb --objects objs/ \
    --out run2 --name run2 --plan-in my_plan.json
```

语义规则库同样可人工覆盖：`--rules my_rules.json`（格式见源码中
`SEMANTIC_RULES` 字典：哪个类放哪种表面、挨着谁、贴不贴墙）。

## 验证项（全部自动，写入 build_report.json）

| 检查 | 含义 |
|---|---|
| `interactive_check` | 对 DYNAMIC 物体施加速度，确认产生位移（可抓取/可推动） |
| `stability_check` | 静置后物体未倾倒 |
| `navmesh.*.navigable_area_m2` | 每个具身的可行走面积 > 0 |
| `validate.reload_ok` | 关掉仿真器从 dataset config 冷启动重载，物体数一致 |
| `validate.semantic_ids_seen` | 抽检图里出现的语义 id |

## 关键参数

- `--num-objects N`：小物体数量（家具自动：每类 1 件 + 椅子 4 把）
- `--interactive-ratio 0.7`：小物体里 70% 保持 DYNAMIC（可交互），其余 STATIC
- `--no-semantic`：回退原版随机撒点（物理修复仍生效）
- `--embodiments spot fetch human`：为每个具身单独烤 navmesh
- `--seed`：全流程确定性，同参数结果可复现

## 环境

- habitat-sim 0.3.1（with bullet，headless/EGL 或带显示均可）
- numpy / pillow / trimesh（仅生成演示物体需要 trimesh）
- 服务器无显示器时用 EGL：`CUDA_VISIBLE_DEVICES=0 python habitat_scene_builder_v2.py ...`

## 已知限制

- 表面分类靠「高度带 + 水平法线」启发式（ground/table/shelf），
  扫描场景里奇形怪状的台面可能归错类——用 `--plan-out` 检查再 `--plan-in` 修正。
- 演示物体是参数化简单网格；换真实 YCB/ReplicaCAD 资产时直接把 GLB
  放进 `--objects` 目录即可，物理参数按类名从 `OBJECT_PROFILES` 查表，
  未知名称走 `DEFAULT_PROFILE`。

## 实现要点（踩过的坑，改动时别回退）

1. **封闭扫描的射线起点**：MP3D 这类扫描是封闭网格，从场景包围盒顶部
   往下打射线会全部命中屋顶/二楼地板，永远看不到真实地面。必须先用
   navmesh 估计地面高度，再从「地面 + 1.6m」（低于天花板）开始打。
2. **多楼层地面高度取众数**：loose navmesh（max_climb=0.3）会把楼梯、
   地下室都纳入可行走区，各层面积接近时「中位数」不稳定。用 0.25m 直方图
   取面积最大的一层（`_estimate_floor_y`）。
3. **重叠检测必须 KINEMATIC**：DYNAMIC 刚体的 Bullet 投机接触边距会让
   `contact_test` 对悬空 3cm 的物体也判「接触地面」→ 100% 误杀。
   流程：KINEMATIC 位姿检测 → 转 DYNAMIC 下落静置 → 静置后定 STATIC/DYNAMIC。
4. **家具接触失败要重试**：单次采样成功率约 30%~90%（取决于与扫描里
   既有家具的重叠），每类家具需要 20+ 次重试才能稳定放下。
5. **球形物体（apple/orange）没有稳定 up 轴**，稳定性检查跳过其朝向，
   只看没滚下表面（profile 里 `round: True`）。
