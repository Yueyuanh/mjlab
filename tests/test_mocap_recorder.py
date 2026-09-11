"""Tests for mocap recording semantics."""

from __future__ import annotations

import numpy as np

from mjlab.mocap.keyframes import InterpConfig, PoseKeyframe, build_trajectory
from mjlab.mocap.recorder import load_keyframes_json, record_to_csv

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
)


def _joint_map(value: float = 0.0) -> dict[str, float]:
  return {name: value for name in JOINT_NAMES}


class FakeMocapEnv:
  """Small recorder test double that exposes changing simulated root states."""

  instances: list["FakeMocapEnv"] = []

  def __init__(
    self,
    device: str = "cpu",
    physics_dt: float = 0.005,
    pd_gain_multiplier: float = 1.0,
  ):
    del device, physics_dt, pd_gain_multiplier
    self.joint_names = JOINT_NAMES
    self.soft_joint_pos_limits = np.tile(np.array([-10.0, 10.0]), (len(JOINT_NAMES), 1))
    self.reset_calls: list[dict[str, np.ndarray | None]] = []
    self.pd_step_calls: list[dict[str, object]] = []
    self._state_index = 0
    self._root_states = [
      (
        np.array([0.0, 0.0, 0.20]),
        np.array([0.0, 0.0, 0.0, 1.0]),
      ),
      (
        np.array([0.1, 0.0, 0.19]),
        np.array([0.0, 0.0, 0.1, 0.995]),
      ),
      (
        np.array([0.2, 0.0, 0.18]),
        np.array([0.0, 0.0, 0.2, 0.980]),
      ),
      (
        np.array([0.3, 0.0, 0.17]),
        np.array([0.0, 0.0, 0.3, 0.954]),
      ),
    ]
    FakeMocapEnv.instances.append(self)

  def reset(
    self,
    joint_pos: np.ndarray | dict[str, float] | None = None,
    root_pos: np.ndarray | None = None,
    root_quat_xyzw: np.ndarray | None = None,
  ) -> None:
    self.reset_calls.append(
      {
        "joint_pos": None if joint_pos is None else np.asarray(joint_pos).copy(),
        "root_pos": None if root_pos is None else np.asarray(root_pos).copy(),
        "root_quat_xyzw": (
          None if root_quat_xyzw is None else np.asarray(root_quat_xyzw).copy()
        ),
      }
    )

  def pd_step(
    self,
    joint_target: dict[str, float] | np.ndarray,
    root_pos: np.ndarray | None = None,
    root_quat_xyzw: np.ndarray | None = None,
    n_steps: int = 1,
    write_root: bool = False,
  ) -> None:
    self.pd_step_calls.append(
      {
        "joint_target": np.asarray(joint_target).copy(),
        "root_pos": None if root_pos is None else np.asarray(root_pos).copy(),
        "root_quat_xyzw": (
          None if root_quat_xyzw is None else np.asarray(root_quat_xyzw).copy()
        ),
        "n_steps": n_steps,
        "write_root": write_root,
      }
    )
    self._state_index = min(self._state_index + 1, len(self._root_states) - 1)

  def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pos, root_quat = self._root_states[self._state_index]
    dof_pos = np.full(len(JOINT_NAMES), float(self._state_index))
    return root_pos.copy(), root_quat.copy(), dof_pos

  def check_joint_limits(
    self, dof_pos: np.ndarray, joint_names: str = ""
  ) -> tuple[bool, np.ndarray]:
    del joint_names
    return True, np.ones_like(np.asarray(dof_pos, dtype=float))


def test_build_trajectory_defaults_unlisted_joints_to_zero():
  # 只在关键帧里指定个别关节，未提及的（如 27dof 的 head_pitch_joint）
  # 应兜底为 0 弧度，而不是 KeyError。
  kf = [
    PoseKeyframe(duration=0.1, joint_pos={"left_hip_pitch_joint": -0.3}),
    PoseKeyframe(duration=0.1, joint_pos={"left_hip_pitch_joint": -0.4}),
  ]
  traj = build_trajectory(kf, JOINT_NAMES)

  assert traj.dof_pos.shape[1] == len(JOINT_NAMES)
  idx_pitch = JOINT_NAMES.index("left_hip_pitch_joint")
  np.testing.assert_allclose(traj.dof_pos[-1, idx_pitch], -0.4)
  for j, name in enumerate(JOINT_NAMES):
    if name != "left_hip_pitch_joint":
      np.testing.assert_allclose(traj.dof_pos[-1, j], 0.0, atol=1e-9)


def test_build_trajectory_loop_closes_track_continuously():
  # 非周期动作：loop=True 应在末尾补一段回退段，且末帧精确等于首帧，
  # 使循环折回时不跳变。
  full = _joint_map()
  kf = [
    PoseKeyframe(
      duration=0.1,
      root_pos=(0.0, 0.0, 0.20),
      root_quat=(0.0, 0.0, 0.0, 1.0),
      joint_pos={**full, "left_hip_pitch_joint": 0.0},
    ),
    PoseKeyframe(
      duration=0.1,
      root_pos=(1.0, 0.0, 0.20),
      root_quat=(0.0, 0.0, 0.0, 1.0),
      joint_pos={**full, "left_hip_pitch_joint": 0.5},
    ),
  ]
  cfg = InterpConfig(fps=20.0, easing="linear")
  linear = build_trajectory(kf, JOINT_NAMES, cfg=cfg, loop=False)
  looped = build_trajectory(kf, JOINT_NAMES, cfg=cfg, loop=True)

  assert looped.num_frames > linear.num_frames
  np.testing.assert_allclose(looped.dof_pos[-1], looped.dof_pos[0])
  np.testing.assert_allclose(looped.root_pos[-1], looped.root_pos[0])
  np.testing.assert_allclose(looped.root_quat_xyzw[-1], looped.root_quat_xyzw[0])
  # 首帧 → 末帧单调走到 1，回退段再退到 0。
  xs = looped.root_pos[:, 0]
  assert xs[0] == 0.0
  assert xs[linear.num_frames - 1] == 1.0
  assert xs[-1] == 0.0


def test_load_keyframes_json_allows_joint_only_keyframes(tmp_path):
  config = tmp_path / "joint_only.json"
  config.write_text(
    """
{
  "fps": 30.0,
  "keyframes": [
    {"duration": 0.1, "joint_pos": {"left_hip_pitch_joint": -0.1}},
    {"joint_pos": {"left_hip_pitch_joint": -0.2}}
  ]
}
""",
    encoding="utf-8",
  )

  keyframes, cfg = load_keyframes_json(config)

  assert cfg.fps == 30.0
  assert keyframes[0].root_pos is None
  assert keyframes[0].root_quat is None
  assert keyframes[0].joint_pos == {"left_hip_pitch_joint": -0.1}


def test_record_to_csv_defaults_to_dynamic_root(monkeypatch, tmp_path):
  FakeMocapEnv.instances.clear()
  monkeypatch.setattr("mjlab.mocap.recorder.MocapEnv", FakeMocapEnv)
  out_csv = tmp_path / "dynamic.csv"
  keyframes = [
    PoseKeyframe(duration=0.1, joint_pos=_joint_map(0.0)),
    PoseKeyframe(joint_pos=_joint_map(0.5)),
  ]

  record_to_csv(keyframes, out_csv, fps=10.0, physics_dt=0.1, verbose=False)

  env = FakeMocapEnv.instances[-1]
  rows = np.loadtxt(out_csv, delimiter=",")
  assert rows.shape == (2, 21)
  assert [call["write_root"] for call in env.pd_step_calls] == [False, False]
  np.testing.assert_allclose(rows[0, :3], [0.1, 0.0, 0.19])
  np.testing.assert_allclose(rows[1, :3], [0.2, 0.0, 0.18])
  np.testing.assert_allclose(rows[0, 7:], np.ones(len(JOINT_NAMES)))
  np.testing.assert_allclose(rows[1, 7:], np.full(len(JOINT_NAMES), 2.0))


def test_record_to_csv_guided_root_preserves_target_root(monkeypatch, tmp_path):
  FakeMocapEnv.instances.clear()
  monkeypatch.setattr("mjlab.mocap.recorder.MocapEnv", FakeMocapEnv)
  out_csv = tmp_path / "guided.csv"
  keyframes = [
    PoseKeyframe(
      duration=0.1,
      root_pos=(1.0, 0.0, 0.30),
      root_quat=(0.0, 0.0, 0.0, 1.0),
      joint_pos=_joint_map(0.0),
    ),
    PoseKeyframe(
      root_pos=(2.0, 0.0, 0.40),
      root_quat=(0.0, 0.0, 0.0, 1.0),
      joint_pos=_joint_map(0.5),
    ),
  ]

  record_to_csv(
    keyframes,
    out_csv,
    fps=10.0,
    physics_dt=0.1,
    write_root=True,
    verbose=False,
  )

  env = FakeMocapEnv.instances[-1]
  rows = np.loadtxt(out_csv, delimiter=",")
  assert [call["write_root"] for call in env.pd_step_calls] == [True, True]
  np.testing.assert_allclose(rows[0, :3], [1.0, 0.0, 0.30])
  np.testing.assert_allclose(rows[1, :3], [2.0, 0.0, 0.40])
  np.testing.assert_allclose(rows[0, 7:], np.ones(len(JOINT_NAMES)))
