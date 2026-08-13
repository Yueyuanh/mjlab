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

Training artifacts — weights, logs, `.pyc` files — stay out of version control
(`.gitignore` or git-lfs). They must not ride along on branch switches.

# Development Workflow

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
