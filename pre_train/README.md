# pre_train — Deploy Corpus Policy

This tree holds **stable, deployable** training artifacts for Pikachu tasks so a
pinned version stays reproducible without re-training. It is committed on
purpose (see `CLAUDE.md`).

## What belongs here (keep)

Per run directory under `pre_train/rsl_rl/<experiment>/<run>/`:

| Keep | Why |
|------|-----|
| **Final** `model_<N>.pt` (highest N only) | Deployable policy weights |
| `*.onnx` (if exported) | Runtime / embedded deploy |
| Metrics dashboard PDF | Human-readable verification |
| **One** primary metrics CSV (prefer `*_with_target.csv`) | Plot/compare against reference |
| Small `params/` / config snapshots if present | Reproduce the run |

## What must NOT be committed

| Drop | Why |
|------|-----|
| Intermediate `model_*.pt` (`model_0`, `model_500`, …) | Multiplies repo size ×10 |
| `events.out.tfevents.*` | 50MB+ tensorboard noise |
| `torchrunx/**/*.log` | Multi-MB stdout dumps |
| Duplicate metrics CSVs (same run, many timestamps) | Keep one primary |
| Large `git/*.diff` full-tree dumps | Not deploy artifacts |

`.gitignore` enforces the noisy patterns. When promoting a new deploy run:

1. Train as usual; only copy/export the **final** checkpoint (+ ONNX + metrics).
2. Place under `pre_train/rsl_rl/<task_family>/<run_name>/`.
3. Run a quick `uv run play <task-id> --wandb-run-path ...` or local play against the checkpoint.
4. Update the Pretrain column in `docs/task_lineage.md`.

## Mapping

Task IDs ↔ run directories: see `docs/task_lineage.md` §2–§3.