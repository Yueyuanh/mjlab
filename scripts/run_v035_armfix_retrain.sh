#!/usr/bin/env bash
# v035 手臂默认位姿修正后的复训：walk v1 配方 5000 轮 -> V5（角速度权重 6）热启动 1500 轮
# -> 确定性评估（5 段 + 视频）。E1 站桩只做对照评估（不改配置、不重训）。
#
# 用法（仓库根 mjlab/ 下）：nohup bash scripts/run_v035_armfix_retrain.sh > logs/... 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

DOC=doc/pikachu_v035_armfix_20261029
RUNS=logs/rsl_rl/pikachu_v035/biped_walk
STAGE1_NAME=biped_walk_v035_armfix
STAGE2_NAME=walk_v5_angweight_armfix

mkdir -p "$DOC/data/walk_v1_armfix" "$DOC/data/walk_v5_armfix" \
  "$DOC/data/e1_control" "$DOC/media/clips" "$DOC/logs"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "stage 1: Walk（v1 配方）5000 轮，2048 env"
uv run train Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 5000 \
  --agent.logger tensorboard \
  --agent.run-name "$STAGE1_NAME"

RUN1=$(ls -dt "$RUNS"/*_"$STAGE1_NAME" | head -1)
RUN1_BASE=$(basename "$RUN1")
log "stage 1 done: $RUN1_BASE"

log "stage 2: Walk-V5 热启动 1500 轮（角速度权重 6.0）"
uv run train Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk-V5 \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1500 \
  --agent.logger tensorboard \
  --agent.run-name "$STAGE2_NAME" \
  --agent.resume True \
  --agent.load-run "$RUN1_BASE" \
  --agent.load-checkpoint "model_4999.pt"

RUN2=$(ls -dt "$RUNS"/*_"$STAGE2_NAME" | head -1)
RUN2_BASE=$(basename "$RUN2")
LAST=$(ls "$RUN2"/model_*.pt | sort -t_ -k2 -n | tail -1)
log "stage 2 done: $RUN2_BASE  last=$LAST"

log "eval: V5 复训模型（5 段 + 视频）"
uv run python scripts/eval_v035.py --mode walk \
  --task Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk-V5 \
  --checkpoint "$LAST" \
  --num-envs 16 --seconds 8 \
  --output "$DOC/data/walk_v5_armfix/eval.json" \
  --video-dir "$DOC/media/clips"

log "eval: walk v1@5000（同配方对照，5 段）"
uv run python scripts/eval_v035.py --mode walk \
  --task Mjlab-Velocity-Flat-Pikachu-V035-Biped-Walk \
  --checkpoint "$RUN1/model_4999.pt" \
  --num-envs 16 --seconds 8 \
  --output "$DOC/data/walk_v1_armfix/eval.json"

log "eval: E1 站桩（旧 checkpoint 在新位姿下的对照，5 段）"
uv run python scripts/eval_v035.py --mode biped \
  --task Mjlab-Velocity-Flat-Pikachu-V035-Biped-E1 \
  --checkpoint logs/rsl_rl/pikachu_v035/biped_e1/2026-10-28_17-22-25_biped_e1_v035/model_2999.pt \
  --num-envs 16 --seconds 8 \
  --output "$DOC/data/e1_control/eval.json"

log "全部完成"
