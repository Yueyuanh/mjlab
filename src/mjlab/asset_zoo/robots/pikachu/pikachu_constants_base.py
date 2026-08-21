"""Pikachu 关节消融实验(手臂摆动力学)专用常量。

集中本消融实验特有的、与 ``pikachu_constants.py`` 不重复的关节/控制参数:
网络策略为**纯腿 10 关节**模型,手臂由零维副作用 ActionTerm ``ArmSineAction``
独立施加正弦摆动(见 ``envs/mdp/actions/arm_sine.py``)。这里提供腿/臂关节名、
左右反相约定与滑条安全幅值上限,以供 npz 裁剪、env 配置与 viewer 使用。
"""

# 腿部 10 个关节名(顺序 = ``PIKACHU_JOINT_NAMES[0:10]``,即前 10 列)。
LEG_JOINT_NAMES: tuple[str, ...] = (
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
)

# 手臂动作维对应的关节名(顺序 = XML 定义序,即 left 先、right 后)。
ARM_PITCH_JOINT_NAMES: tuple[str, str] = (
  "left_arm_pitch_joint",
  "right_arm_pitch_joint",
)
ARM_ROLL_JOINT_NAMES: tuple[str, str] = (
  "left_arm_roll_joint",
  "right_arm_roll_joint",
)

# 左臂相位、右臂 +π(相位差 π),制造非对称力矩扰动。
ARM_RIGHT_PHASE_SHIFT: float = 3.141592653589793


def arm_pitch_amp_upper_limit() -> float:
  """pitch 摆幅滑条的安全上限(rad),低于关节限位,避免摆动越过硬限位。

  默认姿态 arm_pitch ≈ ±1.2,关节限位 [-3.14, 1.05]——向正方向可摆 ~1.05,向负方向可
  摆 ~1.94,取保守公共裕度的上限。实际值在运行时还会用 ``joint_pos_limits`` 复核。
  """
  return 1.2


def arm_roll_amp_upper_limit() -> float:
  """roll 摆幅滑条的安全上限(rad)。默认姿态 ±0.2,对称,给更大裕度。"""
  return 1.0
