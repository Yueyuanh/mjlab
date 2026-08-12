# Mocap Dynamic Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mjlab.mocap` record floating-base dynamics from joint-only JSON by default.

**Architecture:** Keep `MocapEnv` as the low-level simulation wrapper. Change `record_to_csv()` orchestration so dynamic mode reads actual root state from `env.get_state()` and guided mode preserves the old root-target CSV behavior. Update preview modes and docs to make free-root PD the normal path.

**Tech Stack:** Python, NumPy, tyro CLI, pytest, MuJoCo/mjlab simulation.

---

## File Structure

- `src/mjlab/mocap/recorder.py`: load JSON and record CSV; owns dynamic vs guided CSV semantics.
- `src/mjlab/mocap/mocap_env.py`: low-level reset, PD stepping, and state reads; default PD stepping should be free-root.
- `src/mjlab/mocap/preview.py`: viewer playback modes.
- `src/mjlab/mocap/preview_cli.py`: CLI literal modes and docs.
- `src/mjlab/mocap/record_cli.py`: default `write_root=False` and CLI docs.
- `src/mjlab/mocap/README.md`: user-facing workflow.
- `src/mjlab/mocap/configs/*.json`: examples should show joint-only dynamic configs.
- `tests/test_mocap_recorder.py`: fast tests for dynamic/guided recording semantics.

### Task 1: Recorder Semantics Tests

**Files:**
- Create: `tests/test_mocap_recorder.py`

- [ ] **Step 1: Write failing tests**

Create tests that monkeypatch `mjlab.mocap.recorder.MocapEnv` with a fake env.
The fake exposes `joint_names`, `reset()`, `pd_step()`, `get_state()`, and
`check_joint_limits()`.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_mocap_recorder.py -q`

Expected: at least one failure showing default `write_root` or CSV root source
does not match dynamic semantics.

### Task 2: Dynamic Recorder Implementation

**Files:**
- Modify: `src/mjlab/mocap/recorder.py`
- Modify: `src/mjlab/mocap/mocap_env.py`
- Modify: `src/mjlab/mocap/record_cli.py`

- [ ] **Step 1: Change defaults**

Set `write_root=False` in `record_to_csv()`, `MocapEnv.pd_step()`, and
`record_cli.main()`.

- [ ] **Step 2: Build trajectory with simulator initial root**

Before `build_trajectory()`, call `env.get_state()` and pass its root pose as
`initial_root_pos` and `initial_root_quat_xyzw`.

- [ ] **Step 3: Write dynamic CSV rows from actual state**

After each `pd_step()`, read `root_pos, root_quat_xyzw, dof_pos = env.get_state()`.
If `write_root` is false, concatenate actual root and actual joints. If
`write_root` is true, concatenate trajectory root and actual joints.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `uv run pytest tests/test_mocap_recorder.py -q`

Expected: all tests pass.

### Task 3: Preview Modes

**Files:**
- Modify: `src/mjlab/mocap/preview.py`
- Modify: `src/mjlab/mocap/preview_cli.py`

- [ ] **Step 1: Update mode literal**

Allow `mode` values `kinematic`, `pd`, and `pd_guided`.

- [ ] **Step 2: Make `pd` dynamic**

In `pd`, call `env.pd_step(..., write_root=False)`. In `pd_guided`, call
`env.pd_step(..., write_root=True)`.

- [ ] **Step 3: Run a syntax/type smoke check**

Run: `uv run python -m compileall src/mjlab/mocap`

Expected: compileall succeeds.

### Task 4: Docs And Examples

**Files:**
- Modify: `src/mjlab/mocap/README.md`
- Modify: `src/mjlab/mocap/__init__.py`
- Modify: `src/mjlab/mocap/configs/transition_down.json`
- Modify: `src/mjlab/mocap/configs/transition_up.json`
- Modify: `src/mjlab/mocap/configs/NOTES.md`

- [ ] **Step 1: Update README semantics**

Describe dynamic recording as default, explain `--write-root True` as guided
compatibility/debug mode, and show joint-only JSON.

- [ ] **Step 2: Update example configs**

Remove per-keyframe `root_pos` and `root_quat` from example JSON so they only
configure joint angles and timing.

- [ ] **Step 3: Update module docs**

Refresh package and file docstrings to avoid claiming root is always kinematic.

### Task 5: Final Verification

**Files:**
- No production file changes.

- [ ] **Step 1: Run targeted tests**

Run: `uv run pytest tests/test_mocap_recorder.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run formatter/linter on touched files**

Run: `uv run ruff format src/mjlab/mocap tests/test_mocap_recorder.py`

Run: `uv run ruff check src/mjlab/mocap tests/test_mocap_recorder.py`

Expected: both commands exit 0.

- [ ] **Step 3: Inspect diff**

Run: `git diff -- src/mjlab/mocap tests/test_mocap_recorder.py docs/superpowers`

Expected: diff only contains the intended dynamic-root mocap changes.
