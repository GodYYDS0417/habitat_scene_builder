#!/usr/bin/env bash
# 一键生产 Habitat SceneDataset。
#
#   ./run_scene_builder.sh demo                     # 用预设资产一键建场景
#   ./run_scene_builder.sh --stage room.glb ...     # 用你自己的资产
#
# 解释器解析沿用仓库约定: HABITAT_PYTHON > HABITAT_VENV > ${ROOT}/.venv
# 资产根目录: HSB_STAGE_ROOT / HSB_OBJECT_ROOT（见 SCENE_BUILDER.md）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HABITAT_VENV:-${ROOT}/.venv}"
PY="${HABITAT_PYTHON:-${VENV}/bin/python}"

if [ ! -x "${PY}" ]; then
  echo "missing ${PY}"
  echo "set HABITAT_PYTHON to a python that has habitat_sim, e.g."
  echo "  export HABITAT_PYTHON=/path/to/env/bin/python"
  exit 1
fi

if ! "${PY}" -c "import habitat_sim" >/dev/null 2>&1; then
  echo "${PY} does not have habitat_sim installed"
  echo "either point HABITAT_PYTHON at an env that does, or run with --scaffold-only"
  exit 1
fi

# 无需 DISPLAY: 建场景全程走 headless EGL 渲染。

# 第一个参数不以 - 开头时当作 preset 名
ARGS=()
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  ARGS+=(--preset "$1")
  shift
fi
ARGS+=("$@")

cd "${ROOT}"
exec "${PY}" scripts/habitat_scene_builder.py "${ARGS[@]}"
