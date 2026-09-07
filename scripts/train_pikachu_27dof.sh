#!/usr/bin/env bash
# 统一训练启动脚本（Pikachu 27-dof velocity / tracking）。
#
# 两个任务：
#   Mjlab-Velocity-Flat-Pikachu-27dof         —— 双足平地 speed-goal（腿 10 dof，
#                                              arm 8 + ear/tail 6 随机摆动干扰）
#   Mjlab-Tracking-Flat-Pikachu-27dof-Walk    —— 双足前进步态跟踪（腿 10 dof，同上肢干扰）
#
# 两种模式：
#   smoke —— 每个任务极短迭代（默认 3 iters），只验证训练循环/初始条件能跑、不崩。
#   full  —— 每个任务完整训练（默认 30k iters），后台依次跑。
#
# 用法：
#   uv run scripts/train_pikachu_27dof.sh smoke  [iters] [num_envs]
#   uv run scripts/train_pikachu_27dof.sh full   [iters]   [num_envs]
#   nohup uv run scripts/train_pikachu_27dof.sh full > /tmp/pikachu_27dof_train.log 2>&1 &
set -euo pipefail

# 从 uv 包裹内跑，需 cd 到 mjlab 根。
MJLAB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MJLAB_DIR"

MODE="${1:-smoke}"
max_training_iters="${2:-30000}"
smoke_iters="${max_training_iters}"  # smoke 用第二个参数；下面覆盖

# 两个任务（TASK|num_envs_smoke|num_envs_full）。full 用 RTX 4060(8GB) 保守规模，
# 27dof box 碰撞 + 大观测，避免 OOM 白跑；如显存富余可手动加回 1024/512。
TASKS=(
  "Mjlab-Velocity-Flat-Pikachu-27dof|64|512"
  "Mjlab-Tracking-Flat-Pikachu-27dof-Walk|32|256"
)

run_one() {
  local tid="$1" nenv="$2" iters="$3" tag="$4" logger="$5"
  local stamp; stamp="$(date +%m%d_%H%M%S)"
  echo ">>> [$(date +%H:%M:%S)] $tag  $tid  iters=$iters nenv=$nenv run_name=${tag}_${stamp}"
  uv run train "$tid" \
    --env.scene.num-envs "$nenv" \
    --agent.max-iterations "$iters" \
    --agent.run-name "${tag}_${tid##*-}_${stamp}" \
    --agent.logger "$logger" \
    --agent.seed 42
}

if [ "$MODE" = "smoke" ]; then
  # 冒烟：验证两任务初始条件/训练循环。用 tensorboard，免 wandb 登录。
  for row in "${TASKS[@]}"; do
    tid="${row%%|*}"; rest="${row#*|}"; nv="${rest%%|*}"
    run_one "$tid" "$nv" 3 "smoke" "tensorboard"
    echo ">>> smoke OK: $tid"
  done
  echo "ALL_SMOKE_OK"
elif [ "$MODE" = "full" ]; then
  for row in "${TASKS[@]}"; do
    tid="${row%%|*}"; rest="${row#*|}"; nv="${rest#*|}"
    run_one "$tid" "$nv" "$max_training_iters" "full" "tensorboard"
    echo ">>> FULL DONE: $tid"
  done
  echo "ALL_FULL_DONE"
else
  echo "unknown mode: $MODE (smoke|full)" >&2
  exit 2
fi