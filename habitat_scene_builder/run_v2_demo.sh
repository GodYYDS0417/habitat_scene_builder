#!/usr/bin/env bash
# 端到端演示：真实 MP3D 扫描场景 + 语义放置 + 可交互验证
# 用法: bash run_v2_demo.sh [输出目录]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HSB_PYTHON:-python}"
OUT="${1:-$HERE/../demo_run/living_demo}"

# 只用 GPU0（GPU1 上有别的实验在跑）
export CUDA_VISIBLE_DEVICES="${HSB_GPU:-0}"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# 真实扫描场景（Habitat MP3D 示例公寓）
STAGE="${HSB_STAGE:-/root/rl/simulation_habitat-main/data/versioned_data/mp3d_example_scene_1.1/17DRP5sb8fy/17DRP5sb8fy.glb}"

exec "$PY" "$HERE/habitat_scene_builder_v2.py" \
    --stage "$STAGE" \
    --objects "$HERE/../demo_assets/objects" \
    --out "$OUT" --name "$(basename "$OUT")" \
    --num-objects 20 --num-chairs 4 \
    --min-nav-area "${HSB_MIN_NAV_AREA:-5.0}" \
    --interactive-ratio 0.7 \
    --embodiments spot fetch human \
    --views 6 --resolution 512 --seed 0
