"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.arm_swing_viewer import ArmSwingViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "frozen", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "arm-swing"] = "auto"
  com: bool = False
  """开启实时稳定性叠加层：整机质心 CoM、绕其惯量主轴椭球、足底接触支撑
  多边形（viser 的 Visualization tab 里也有开关）。"""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  sampling_mode: Literal["adaptive", "uniform", "start"] | None = None
  """Override the tracking motion command's sampling mode.

  ``start`` fixes every episode's initial frame to reference frame 0 (useful to
  match the training initial state). Defaults to the env cfg value (adaptive).
  Ignored for non-tracking tasks.
  """
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class _BaseBalanceWrapper(RslRlVecEnvWrapper):
  """Frozen 静稳测试用的基座平衡器（仅 GPU/warp 下 frozen 分支启用）。

  已反复验证：warp（float32）求解器对「自由 base + 软 PD 锁 deep-squat 站姿」这
  个本身无平衡点的系统，任何求参（iter/tolerance/solref/Kp/脚底面积）都无法让
  base 静态承重——base 会缓慢下沉穿地（z->-0.15）+ 蠕动，被用户看成“打滑”。
  CPU（float64）同配置靠更稳的收敛能短暂立在 z~0.095。唯一能在 GPU 上也站住的，
  是给 base 加一个姿态/高度平衡器（等效真实足式机器人的站桩平衡反馈）：

  - 高度/水平位置 spring + damper（把 base 拉回站姿态，压住下沉穿地）
  - 姿态对齐力矩（把 base 体上轴拉回世界 +z，压住前翻/侧翻）

  仅作用于 base body，不改关节、不改接触/求解器，不影响 velocity 训练 sim。
  """

  # 平衡增益。z_ref 不再写死：从 default_root_state 读当前 keyframe 高度
  # （YRP 屈膝约 0.180，PRY 约 0.168，旧 deep-squat 约 0.14）。
  _KZ, _DZ, _KXY, _DXY, _KAL, _DAL = 900.0, 90.0, 55.0, 35.0, 6.0, 1.2
  _UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

  def __init__(
    self, env: Any, clip_actions: float | None = None, body_name: str = "base_link"
  ):
    super().__init__(env, clip_actions=clip_actions)
    robot = self.unwrapped.scene["robot"]
    self._robot = robot
    # 平衡器把力/力矩作用在机器人的躯干(body)上。body 名逐任务不同：
    # Pikachu 叫 base_link，G1 叫 torso_link —— 不能硬编码，从调用方传
    # (env_cfg.viewer.body_name)，默认 base_link 兼容旧调用。
    self._body_name = body_name
    default_root = robot.data.default_root_state
    z_ref = float(default_root[0, 2].item()) if default_root is not None else 0.168
    self._Z_REF = z_ref
    print(f"[INFO]: BaseBalance z_ref={z_ref:.4f} (from default_root_state)")
    model = self.unwrapped.sim.mj_model
    names = [
      model.names[
        int(model.name_bodyadr[i]) : model.names.find(
          b"\x00", int(model.name_bodyadr[i])
        )
      ].decode()
      for i in range(model.nbody)
    ]
    self._base_id = names.index(f"robot/{self._body_name}")
    self._device = self.unwrapped.device

  def step(self, actions):
    self._apply()
    return super().step(actions)

  def _apply(self) -> None:
    data = self._robot.data
    pose = data.root_link_pose_w[0]  # (7,) = [pos(3), quat_wxyz(4)]
    pos = pose[:3]
    quat = pose[3:7]
    vel = data.root_link_lin_vel_w[0]
    ang_vel = data.root_link_ang_vel_w[0]
    up = quat_apply(quat, self._UP.detach().clone().to(quat.device))
    err = torch.cross(up, self._UP.to(up.device), dim=-1)
    force = torch.empty(3, device=quat.device, dtype=torch.float32)
    force[0] = self._KXY * (-pos[0]) - self._DXY * vel[0]
    force[1] = self._KXY * (-pos[1]) - self._DXY * vel[1]
    force[2] = self._KZ * (self._Z_REF - pos[2]) - self._DZ * vel[2]
    torque = self._KAL * err - self._DAL * ang_vel
    self._robot.write_external_wrench_to_sim(
      force.reshape(1, 1, 3),
      torque.reshape(1, 1, 3),
      env_ids=torch.tensor([0], device=self._device),
      body_ids=[self._base_id],
    )


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random", "frozen"}
  TRAINED_MODE = not DUMMY_MODE

  if DUMMY_MODE:
    # Dummy agents are for static pose / pipeline checks, not robustness demos.
    # Strip upper-body flail and startup domain randomization so the robot is
    # evaluated on the calibrated keyframe, not a randomized model.
    env_cfg.actions.pop("arm_flail", None)
    env_cfg.actions.pop("extremity_flail", None)
    # Pin velocity command to zero: nonzero sampled twists only affect reward/obs,
    # but viewers often read them as "the robot is being told to walk".
    if "twist" in env_cfg.commands:
      twist = env_cfg.commands["twist"]
      if hasattr(twist, "ranges"):
        twist.ranges.lin_vel_x = (0.0, 0.0)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)
        if getattr(twist, "ranges", None) is not None and hasattr(
          twist.ranges, "heading"
        ):
          twist.ranges.heading = (0.0, 0.0)
      if hasattr(twist, "rel_standing_envs"):
        twist.rel_standing_envs = 1.0
        twist.rel_heading_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.rel_backward_envs = 0.0
        twist.rel_lateral_envs = 0.0
    for ev_name in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
      env_cfg.events.pop(ev_name, None)
    # Pin root reset offsets to zero (keep default_root_state pose).
    if "reset_base" in env_cfg.events:
      env_cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      }
    print(f"[INFO]: Dummy agent '{cfg.agent}' — flail/DR disabled, reset pinned")

  if cfg.agent in {"frozen", "zero"}:
    # Hold the calibrated default stance on every actuated joint.
    # JointPositionAction has use_default_offset=True, so zero action =
    # default_joint_pos for all joints in actuator_names. Entity.reset also
    # writes non-action joint targets to default after clear_state.
    joint_pos_action = env_cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = (".*",)
    if cfg.agent == "zero":
      print("[INFO]: zero agent holding default_joint_pos on all joints")

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Optionally fix the initial frame (e.g. --sampling-mode start to match the
    # training initial state). Defaults to the env cfg value (adaptive).
    if cfg.sampling_mode is not None:
      motion_cmd.sampling_mode = cfg.sampling_mode

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # zero/frozen 静稳测试：GPU/warp + 软 PD + 自由 base 在站姿下无静态平衡点，
  # base 会下沉/蠕动，用户常看成“向后打滑/滑倒”。与 frozen 相同，给 zero 也
  # 挂基座姿态/高度平衡器（只作用 base，不改关节 PD / 训练路径）。
  if cfg.agent in {"frozen", "zero"}:
    env = _BaseBalanceWrapper(
      env,
      clip_actions=agent_cfg.clip_actions,
      body_name=(env_cfg.viewer.body_name or "base_link"),
    )
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent in {"zero", "frozen"}:

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  metrics_output_dir = (
    log_dir or (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
  )

  def metrics_source_name() -> str:
    checkpoint_name = (
      ckpt_manager.current_name
      if ckpt_manager is not None
      else (resume_path.name if resume_path is not None else cfg.agent)
    )
    return f"{task_id}__{Path(checkpoint_name).stem}"

  if resolved_viewer == "native":
    NativeMujocoViewer(
      env,
      policy,
      com_visualizer=cfg.com,
      freeze_joints_default=DUMMY_MODE,
    ).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(
      env,
      policy,
      checkpoint_manager=ckpt_manager,
      metrics_output_dir=metrics_output_dir,
      metrics_source_name=metrics_source_name,
      com_visualizer=cfg.com,
      freeze_joints_default=DUMMY_MODE,
    ).run()
  elif resolved_viewer == "arm-swing":
    ArmSwingViewer(
      env,
      policy,
      checkpoint_manager=ckpt_manager,
      metrics_output_dir=metrics_output_dir,
      metrics_source_name=metrics_source_name,
    ).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
