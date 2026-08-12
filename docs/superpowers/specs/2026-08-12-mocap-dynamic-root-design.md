# Mocap Dynamic Root Design

## Goal

`mjlab.mocap` should let users configure only joint angles in JSON, then record
motions whose floating base evolves from MuJoCo dynamics under the robot's own
contacts and PD joint actuators.

## Current Problem

Pikachu is a floating-base robot in the MJCF, and the joints are driven by
position actuators. The mocap pipeline currently overrides that behavior:

- `MocapEnv.pd_step()` defaults to writing root pose every physics substep.
- `record_to_csv()` writes interpolated root targets into CSV instead of the
  simulated root state.
- `preview_keyframes(mode="pd")` also writes the interpolated root pose.
- Example JSON and docs imply every keyframe must define `root_pos` and
  `root_quat`.

This produces root-guided kinematic reference data, not natural floating-base
dynamics.

## Design

Dynamic recording is the default. `record_to_csv()` defaults `write_root=False`.
It resets the robot once to the first frame, using configured root fields when
provided and the robot's current/default root otherwise. Each subsequent frame
writes only joint PD targets, steps physics, reads `env.get_state()`, and writes
the actual root pose and joint positions to CSV.

The old guided behavior remains available with `write_root=True`. In that mode
the recorder keeps writing root targets during stepping and writes those guided
root targets to CSV for backwards compatibility with existing root-authored
motions.

JSON root fields are optional. Keyframes may contain only `duration`/`time` and
`joint_pos`. When no root is configured, trajectory building uses the simulator's
current root as the initial and carried root value. Missing joint keys still
carry forward from previous keyframes; the first keyframe must provide all
robot joints unless an explicit initial joint map is passed.

Preview semantics are updated to match recording. `mode="pd"` shows free
floating-base dynamics. A separate `mode="pd_guided"` keeps the old root-guided
debug view, and `mode="kinematic"` remains the direct target-pose preview.

## Testing

Add unit tests with a fake `MocapEnv` so the core recording semantics can be
verified without launching MuJoCo:

- JSON with joint-only keyframes loads successfully.
- Default recording calls `pd_step(..., write_root=False)` and writes actual
  simulated root state to CSV.
- Guided recording calls `pd_step(..., write_root=True)` and writes trajectory
  root targets to CSV.

Run the targeted mocap tests, then run formatting/lint/type checks where
available.
