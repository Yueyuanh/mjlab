#!/usr/bin/env bash
# V031「单侧偏航 / hip_roll 不对称」的四档对照训练（一次只改一个因素）。
#
#   1. T6        —— 原基线：无动作裁剪、无对称约束
#   2. Sym       —— 只加镜像数据增强 + mirror loss
#   3. SymClip   —— 两者都加
#   4. Clip      —— 只加动作裁剪
#
# 四档都用同一套 T6 环境、同样的 4096 env × 5000 iter 预算，所以是**同预算对照**，
# 不是与原来 30000 轮的 run 比绝对水平。
#
# 用法（在 mjlab/ 下）：
#   nohup bash scripts/run_v031_symmetry_ablation.sh > /tmp/v031_ablation/driver.log 2>&1 &
set -u

cd "$(dirname "$0")/.." || exit 1

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

run_arm() {
  local task="$1"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] 开始训练 $task  (${ENVS} env × ${ITERS} iter)"
  echo "=================================================================="
  uv run train "$task" \
    --env.scene.num-envs "$ENVS" \
    --agent.max-iterations "$ITERS" \
    --agent.save-interval "$SAVE" \
    --log-root "$LOG_ROOT" \
    > "$ROOT/${task##*-}.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] $task 结束 rc=$rc"
  return $rc
}

for task in \
  Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  Mjlab-Velocity-Flat-Pikachu-V031-Sym \
  Mjlab-Velocity-Flat-Pikachu-V031-SymClip \
  Mjlab-Velocity-Flat-Pikachu-V031-Clip
do
  run_arm "$task" || echo "!! $task 失败，继续下一档"
done

echo "[$(date +%H:%M:%S)] 全部结束"
