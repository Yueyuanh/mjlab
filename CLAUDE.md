# Canonical Training Specs

Before working on model import, task design, training, experiment artifacts, or
evaluation, read `spec/agent-defaults.md` completely and then read every topic
spec routed there for the current task. The `spec/` directory is canonical for
new training workflow rules. The fork, Git, server, and code-quality rules below
remain active.

# Branch Strategy (Pikachu fork)

> Fork-specific convention. The upstream project (mujocolab/mjlab) follows the
> standard fork → feature branch → PR flow; this fork adds the following rules
> for developing Pikachu training tasks on the internal `origin`
> (git.finnoxcorp.com/yuanhao.yue/mjlab).

## Principles

- **Single mainline.** `main` tracks the internal `origin/main` and must stay
  trainable at all times. Day-to-day development happens directly on `main`.
- **Isolate by directory, not by branch.** Different training tasks live in
  separate config trees and are developed on the same `main`:
  - velocity → `src/mjlab/tasks/velocity/config/pikachu/`
  - tracking → `src/mjlab/tasks/tracking/config/pikachu_transition/`
  Shared framework changes (mdp, rl, assets) go straight to `main` so both
  tasks benefit immediately.

## When to branch

- **`task/<name>`** — long-lived work touching shared code across multiple
  iterations (e.g. refactoring Pikachu collision bodies, mocap recording).
  Integrate `main` after every milestone; never let it drift far behind.
- **`feat/<name>`** — experimental work that may break `main`'s trainability
  (e.g. an experiment mid-training). Merge back to `main` or delete when done.

A task branch is merged back to `main` and deleted as soon as it is complete.
Do not keep branches whose content has already been absorbed into `main`
(e.g. `pikachu_transition` is fully absorbed — its content lives on `main` now).

## Dependencies between tasks

If tracking builds on velocity, strictly sequence the work: velocity merges to
`main` first, then tracking continues from the updated `main`. Never grow a
dependent task out of an old task branch.

## Remotes

- `origin` → internal git.finnoxcorp.com (default; `main` tracks `origin/main`).
- `github` → mujocolab/mjlab upstream (only for syncing upstream updates;
  syncing it into `main` re-introduces divergence).

## Out of scope for git

Training artifacts — checkpoint weights, run logs, ad-hoc `metrics_*.csv`,
`.pyc` files — stay out of version control (`.gitignore` or git-lfs). They
must not ride along on branch switches.

**Exception — deploy corpus (`pre_train/`)**: `pre_train/rsl_rl/<task>/<run>/`
holds the trained, stable models and their recorded metrics used for
**deployment** (reference motions plus the metrics CSVs / dashboard PDFs used
to verify them). These are committed to git on purpose so the pinned
deployable version stays reproducible — do not add them to an ignore rule and
do not delete them. Treatment differs from transient scratch (recordings under
`mocap/tmp/`, ad-hoc plots), which may be left untracked.

# Development Workflow

## 4090 server synchronization (user requirement, 2026-10-22)

- Server: `finnox-4090@10.20.15.181`. All remote modifications, including
  backups and reports, must stay inside `/home/finnox-4090/RL/Pikachu/mjlab`.
- Use `bash scripts/sync_4090.sh code` on the workstation to push source,
  V033 robot assets, tests and documentation. It backs up overwritten files
  inside the remote `.sync/backups/` and never deletes remote-only files.
- Keep each machine's environment, Git state, active training logs and
  intermediate checkpoints local. Do not mirror training output directories.
- Transfer only explicitly identified final models after training completes,
  using `push-final` / `pull-final` with the final model's relative path and
  verified SHA-256 from `config/run.json`. Never infer completion from the
  highest checkpoint number or transfer a live checkpoint automatically.
- Do not store SSH passwords in scripts, documentation or Git.
- Details and commands: `doc/code_sync_20261022/README.md`.

## Training experiment artifact layout

Historical note: the 2026-10-22 artifact-type layout was superseded on
2026-10-29. Do not use `training/<run>`, experiment-level `evaluation/`, or
parallel `reports/metadata/provenance` trees for new runs. The canonical layout
is now stage-first and run-local:

```text
logs/rsl_rl/<robot>/<task>/<experiment>/
  1-sanity/<run>/
  2-smoke/<run>/
  3-pilot/<run>/
  4-full/<run>/
```

Read and follow `spec/experiment-logging.md` for the complete run layout,
`E<experiment>-R<repeat>-<YY-MM-DD>` naming, automatic full evaluation, and
migration rules. Historical experiments remain where they are unless a current
task explicitly migrates them.

## Evaluation video standard (Pikachu)

Historical note: the earlier morphology-specific side/front camera rules were
superseded on 2026-10-29. Every new locomotion final video uses a 1280x720,
50 FPS third-person three-quarter view with the velocity response overlay at
bottom-right. Read and follow `spec/evaluation-report.md` and
`spec/experiment-logging.md`; those files are the canonical video contract.

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

# W&B experiment tracking

- Every registered task must pass **distinct** `wandb_project` and `wandb_tags`
  to its `*_ppo_runner_cfg()` factory. Name the project `<robot>-<task>-<variant>`
  (e.g. `pikachu-transition-down`, `pikachu-velocity-flat`) and tag runs with the
  task semantics (e.g. `("transition", "down")`). Never let multiple tasks share
  the default `mjlab` project — otherwise their training curves mix together and
  cannot be told apart by task.
- These two parameters are **required** (the factory signature has no defaults), so
  adding a new task forces an explicit value.
- You can still override at train time with `--agent.wandb-project` /
  `--agent.wandb-tags`.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
