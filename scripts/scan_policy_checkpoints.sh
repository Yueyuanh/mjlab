#!/usr/bin/env bash
# Evaluate a checkpoint sweep with fixed commands, one output folder per
# checkpoint. Each run writes <out>/<stem>/metrics/<clip>_metrics.json through
# scripts/review_v033_policy.py so gait and knee metrics stay comparable.
#
# Usage:
#   bash scripts/scan_policy_checkpoints.sh \
#     --task Mjlab-Velocity-Flat-Pikachu-V034-Biped-Knee \
#     --checkpoint-dir <run>/checkpoints \
#     --out <experiment>/evaluation/ckpt_scan \
#     --clips forward --seconds 12 --num-envs 16
#
# Missing checkpoints are skipped, so the same command is safe to rerun after a
# training run adds new model_*.pt files.
set -euo pipefail

task=""
checkpoint_dir=""
checkpoints=""
out=""
clips="forward"
seconds="12"
num_envs="16"
seed="42"
log_dir=""
case_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2 ;;
    --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
    --checkpoints) checkpoints="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --clips) clips="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    --num-envs) num_envs="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --case-name) case_name="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$task" || -z "$out" ]]; then
  echo "--task and --out are required" >&2
  exit 2
fi

files=()
if [[ -n "$checkpoints" ]]; then
  read -r -a files <<< "$checkpoints"
elif [[ -n "$checkpoint_dir" ]]; then
  shopt -s nullglob
  files=("$checkpoint_dir"/model_*.pt)
  shopt -u nullglob
else
  echo "--checkpoints or --checkpoint-dir is required" >&2
  exit 2
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no checkpoints found" >&2
  exit 1
fi

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
fi

for ckpt in "${files[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "skip missing $ckpt"
    continue
  fi
  stem="$(basename "$ckpt" .pt)"
  if [[ -n "$case_name" ]]; then
    stem="$case_name"
  fi
  for clip in ${clips//,/ }; do
    metrics_dir="$out/$stem/metrics"
    mkdir -p "$metrics_dir"
    echo "== $stem / $clip"
    cmd=(
      uv run python scripts/review_v033_policy.py
      --task "$task"
      --checkpoint "$ckpt"
      --out "$metrics_dir"
      --name "${clip}_metrics"
      --clip "$clip"
      --seconds "$seconds"
      --num-envs "$num_envs"
      --seed "$seed"
    )
    if [[ -n "$log_dir" ]]; then
      mkdir -p "$log_dir/$stem"
      "${cmd[@]}" > "$log_dir/$stem/${clip}.log" 2>&1
    else
      "${cmd[@]}"
    fi
  done
done
