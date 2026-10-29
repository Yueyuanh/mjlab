#!/usr/bin/env bash
# Run on the local workstation. Passwords are requested by SSH, never saved.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
REMOTE=finnox-4090@10.20.15.181
REMOTE_ROOT=/home/finnox-4090/RL/Pikachu/mjlab
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
SSH=(ssh -o ControlMaster=auto -o ControlPersist=600
  -o ControlPath=/tmp/pikachu-4090-sync-%C)
export RSYNC_RSH='ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=/tmp/pikachu-4090-sync-%C'
cd "$ROOT"
if [[ $(hostname) == finnox-4090 ]]; then
  echo 'Run this helper on the local workstation.' >&2
  exit 2
fi
usage() {
  echo 'Usage: bash scripts/sync_4090.sh code [--dry-run]'
  echo '       bash scripts/sync_4090.sh {push-final|pull-final} RELATIVE_MODEL_PATH SHA256'
  echo 'Only specify a confirmed final model; never select a live/latest checkpoint.'
}
MODE=${1:-help}
if [[ $MODE == help || $MODE == --help ]]; then usage; exit 0; fi
# Resolve the target before writing anything, including backup directories.
"${SSH[@]}" "$REMOTE" "test \"\$(realpath '$REMOTE_ROOT')\" = '$REMOTE_ROOT'"
if [[ $MODE == code ]]; then
  OPTIONS=()
  if [[ ${2:-} == --dry-run && $# == 2 ]]; then
    OPTIONS+=(--dry-run)
  elif [[ $# != 1 ]]; then usage; exit 2; fi
  SOURCES=(src scripts tests typings pyproject.toml uv.lock .python-version
    .gitignore CLAUDE.md AGENTS.md README.md Makefile
    docs/source/changelog.rst doc/code_sync_20261022 doc/pikachu_v033*)
  SYNC_ARGS=(-rlc --relative --safe-links --itemize-changes --stats
    "${OPTIONS[@]}" --backup --backup-dir="$REMOTE_ROOT/.sync/backups/$STAMP/code"
    --exclude=__pycache__/ --exclude='*.pyc' --exclude=.git/
    --exclude=.venv/ --exclude=.pytest_cache/ --exclude=.ruff_cache/
    --exclude=logs/ --exclude=wandb/ --exclude=pre_train/
    --exclude=checkpoints/ --exclude=exports/ --exclude=provenance/
    --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt'
    --exclude='*.onnx' --exclude='*.safetensors'
    --exclude='events.out.tfevents.*' --exclude='*.log'
    --exclude='*.mp4' --exclude='*.gif' --exclude='*.npz' --exclude='*.npy'
    --exclude=MUJOCO_LOG.TXT)
  rsync "${SYNC_ARGS[@]}" "${SOURCES[@]}" "$REMOTE:$REMOTE_ROOT/"
elif [[ $MODE == push-final || $MODE == pull-final ]]; then
  [[ $# == 3 ]] || { usage; exit 2; }
  MODEL=$2
  EXPECTED=$3
  [[ $MODEL =~ ^[a-zA-Z0-9_./-]+\.(pt|pth|onnx|ckpt|safetensors)$ &&
    $MODEL != /* && /$MODEL/ != */../* && /$MODEL/ != */./* &&
    $EXPECTED =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid path/hash.' >&2; exit 2; }
  case "$MODEL" in logs/rsl_rl/pikachu_v033/*|pre_train/*) ;;
    *) echo 'Final models must be under logs/rsl_rl/pikachu_v033/ or pre_train/.' >&2; exit 2;;
  esac
  RESOLVED=$(realpath -m -- "$ROOT/$MODEL")
  [[ $RESOLVED == "$ROOT/"* ]] || exit 2
  "${SSH[@]}" "$REMOTE" "case \"\$(realpath -m '$REMOTE_ROOT/$MODEL')\" in '$REMOTE_ROOT/'*) ;; *) exit 2;; esac"
  if [[ $MODE == push-final ]]; then
    printf '%s  %s\n' "$EXPECTED" "$MODEL" | sha256sum --check
    "${SSH[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT/$(dirname -- "$MODEL")'"
    rsync -rc --itemize-changes --backup \
      --backup-dir="$REMOTE_ROOT/.sync/backups/$STAMP/models" \
      "$MODEL" "$REMOTE:$REMOTE_ROOT/$MODEL"
    "${SSH[@]}" "$REMOTE" "cd '$REMOTE_ROOT' && printf '%s  %s\n' '$EXPECTED' '$MODEL' | sha256sum --check"
  else
    "${SSH[@]}" "$REMOTE" "cd '$REMOTE_ROOT' && printf '%s  %s\n' '$EXPECTED' '$MODEL' | sha256sum --check"
    mkdir -p -- "$(dirname -- "$MODEL")"
    rsync -rc --itemize-changes --backup \
      --backup-dir="$ROOT/.sync/backups/$STAMP/models" \
      "$REMOTE:$REMOTE_ROOT/$MODEL" "$MODEL"
    printf '%s  %s\n' "$EXPECTED" "$MODEL" | sha256sum --check
  fi
else
  usage
  exit 2
fi
