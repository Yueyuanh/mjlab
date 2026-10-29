#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

root="doc/pikachu_v031_gait_improvement_20261011"
e0_run="logs/rsl_rl/pikachu_v031_velocity/2026-10-10_20-11-53"
e1_run="logs/rsl_rl/pikachu_v031_e1_clip/2026-10-11_21-15-42_screen-s42"
e2_run="logs/rsl_rl/pikachu_v031_e2_symmetry/2026-10-11_21-29-49_screen-s42"
e3_run="logs/rsl_rl/pikachu_v031_e3_yaw_huber/2026-10-11_21-45-39_screen-s42"
e4_run="logs/rsl_rl/pikachu_v031_e4_curriculum/2026-10-11_22-01-49_final-s42"
clips="stand,fwd,fwd_fast,back,lat,turn"

record_nominal() {
  local task="$1"
  local checkpoint="$2"
  local tag="$3"
  uv run python scripts/record_v03_metrics.py \
    --task "$task" --checkpoint "$checkpoint" \
    --out "$root/metrics/$tag" --tag "$tag" \
    --clips "$clips" --seconds 8 --seed 42 --nominal
}

# Equal-budget causal screen: all checkpoints are model_1000.pt.
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  "$e0_run/model_1000.pt" E0_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E1-Clip \
  "$e1_run/model_1000.pt" E1_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E2-Symmetry \
  "$e2_run/model_1000.pt" E2_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E3-YawHuber \
  "$e3_run/model_1000.pt" E3_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  "$e4_run/model_1000.pt" E4_nominal_s42

# Practical comparison: old 30k final model against the 5k E4 candidate.
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  "$e0_run/model_29999.pt" E0final_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  "$e4_run/model_2500.pt" E4pre_full_nominal_s42
record_nominal Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  "$e4_run/model_5000.pt" E4final_nominal_s42

# E4 robustness slices: startup DR only, then startup DR + full push + full noise.
uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint "$e4_run/model_5000.pt" \
  --out "$root/metrics/E4final_startupdr_s42" --tag E4final_startupdr_s42 \
  --clips "$clips" --seconds 8 --seed 42
uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint "$e4_run/model_5000.pt" \
  --out "$root/metrics/E4final_pushnoise_s42" --tag E4final_pushnoise_s42 \
  --clips "$clips" --seconds 8 --seed 42 --with-push --obs-noise
uv run python scripts/record_v03_metrics.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint "$e4_run/model_2500.pt" \
  --out "$root/metrics/E4pre_full_pushnoise_s42" \
  --tag E4pre_full_pushnoise_s42 --clips "$clips" \
  --seconds 8 --seed 42 --with-push --obs-noise

# Same fwd command, camera, duration, seed, and nominal plant for visual comparison.
uv run python scripts/record_v03_video.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-T6 \
  --checkpoint "$e0_run/model_29999.pt" --out "$root/videos" \
  --tag E0_model29999_nominal --clips fwd --views side,front \
  --seconds 8 --seed 42 --nominal
uv run python scripts/record_v03_video.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint "$e4_run/model_2500.pt" --out "$root/videos" \
  --tag E4_model2500_nominal --clips fwd --views side,front \
  --seconds 8 --seed 42 --nominal
uv run python scripts/record_v03_video.py \
  --task Mjlab-Velocity-Flat-Pikachu-V031-E4-Curriculum \
  --checkpoint "$e4_run/model_5000.pt" --out "$root/videos" \
  --tag E4_model5000_nominal --clips fwd --views side,front \
  --seconds 8 --seed 42 --nominal

uv run python "$root/analyze_experiments.py"
