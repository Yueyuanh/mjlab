#!/usr/bin/env bash
set -euo pipefail

run() {
  local task="$1"
  local iterations="$2"
  local run_name="$3"
  uv run train "$task" \
    --env.scene.num-envs 4096 \
    --agent.max-iterations "$iterations" \
    --agent.save-interval 500 \
    --agent.seed 42 \
    --agent.run-name "$run_name" \
    --agent.logger tensorboard
}

run Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip 5000 screen-s42
run Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry 5000 screen-s42
run Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber 5000 screen-s42
run Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum 30000 final-s42
