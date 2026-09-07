#!/usr/bin/env bash
# action_rate_l2 权重消融：每个权重从头训练 5000 轮（4096 envs），run-name 按参数命名。
# 每轮只差 --env.rewards.action-rate-l2.weight 这一个变量，其余完全一致。
# 需在 mjlab 目录下运行：  bash run_ar_ablation.sh
# 时长预算：4096 envs 下约 1.3s/轮 → 5000 轮 ≈ 1.8h/轮，3 轮 ≈ 5.5h 总量。
set -euo pipefail
cd "$(dirname "$0")"

TASK=Mjlab-Velocity-Flat-Pikachu-Sl22-Quad
ENVS=4096
MAX_ITER=5000
VIDEO=True                       # tyro 关 bool 折叠：bool 要显式给 Python 值 True/False

run() {
  local w="$1"; local name="$2"
  echo
  echo "=================================================================="
  echo ">> action_rate_l2.weight=${w}   run-name=${name}"
  echo ">> 从头训练 ${MAX_ITER} 轮, ${ENVS} envs"
  echo "=================================================================="
  uv run train "$TASK" \
    --env.scene.num-envs "${ENVS}" \
    --agent.max-iterations "${MAX_ITER}" \
    --agent.run-name "${name}" \
    --env.rewards.action-rate-l2.weight "${w}" \
    --video "${VIDEO}"
}

run -0.25 ar_w0.25
run -0.40 ar_w0.40
run -0.60 ar_w0.60