#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export RL100_CORE_ROOT="$ROOT/rl100"
export RL100_DUAL_ROOT="$ROOT/experiments/dual_rgb"
export RL100_CONFIG_ROOT="$ROOT/configs/historical"
export PYTHONPATH="$RL100_DUAL_ROOT:$RL100_CORE_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
export MUJOCO_EGL_DEVICE_ID=${GPU_ID:-0} EGL_DEVICE_ID=${GPU_ID:-0}
export LD_LIBRARY_PATH="${MUJOCO_BIN:-$HOME/.mujoco/mujoco210/bin}:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true
cd "$ROOT"
exec "${PYTHON:-python}" -u "$@"
