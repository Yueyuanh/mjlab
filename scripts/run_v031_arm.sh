#!/usr/bin/env bash
# 跑任意一档（或几档）V031 消融训练，参数与 run_v031_symmetry_ablation.sh 一致，
# 这样任何一档都能和已有四档做**同预算**对照。
#
# 用法（在 mjlab/ 下）：
#   nohup bash scripts/run_v031_arm.sh Mjlab-Velocity-Flat-Pikachu-V031-YawHuber \
#     > /tmp/v031_yawhuber/driver.log 2>&1 &
#
# 环境变量覆盖：ITERS / ENVS / SAVE / ROOT / LOG_ROOT。
set -u

cd "$(dirname "$0")/.." || exit 1

if [ "$#" -eq 0 ]; then
  echo "用法: $0 <task_id> [<task_id> ...]" >&2
  exit 2
fi

ITERS="${ITERS:-5000}"
ENVS="${ENVS:-4096}"
SAVE="${SAVE:-500}"
ROOT="${ROOT:-/tmp/v031_ablation}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
# 本机离线，wandb 走网络会挂；tensorboard event 文件照常写。
export WANDB_MODE=disabled

mkdir -p "$ROOT"

for task in "$@"; do
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] 开始训练 $task  (${ENVS} env × ${ITERS} iter)"
  echo "=================================================================="
  uv run train "$task" \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$ITERS" \
    --agent.save-interval "$SAVE" \
    --log-root "$LOG_ROOT" \
    > "$ROOT/${task##*-}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $task 结束 rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "!! $task 失败，继续下一档"
  fi
done

echo "[$(date +%H:%M:%S)] 全部结束"
