"""V035 直立行走 v4：给膝关节接上真实的 DC 电机扭矩-转速模型。

v1/v2 用的是 ``BuiltinPositionActuatorCfg``——MuJoCo 内置位置执行器，
**只认 ``effort_limit``，不认 ``velocity_limit``**。也就是说 v035 常量表里写的
「4340 额定 36 rpm」在仿真中根本没有生效：膝关节可以任意速度摆动，而且任何
速度下都能拿到 40 N·m。

评估 v1 时测得膝关节峰值角速度 8.0 rad/s（前进命令 0.2）到 10.3 rad/s（命令 0.4），
是额定 3.77 rad/s 的 2.1–2.7 倍，而力矩只有 2 N·m 出头。这是当前最大的
sim2real 差距：真实 4340 在 8 rad/s 处早已没有力矩。

本档把膝关节换成 ``DcMotorActuatorCfg``（``IdealPdActuator`` 的子类，
只多一层线性扭矩-转速削顶）：

  * ``saturation_effort = 40``：堵转/峰值力矩；
  * ``velocity_limit = 7.5``：**空载转速**，按额定 36 rpm 的 2 倍假设
    （DM 电机的额定转速一般是连续工作点，空载通常 1.5–2 倍，但没有官方数据，
    这是本档唯一的外部假设）；
  * ``effort_limit = 12``：额定连续力矩。

其余关节（4310 / 3507）不动，奖励与命令全部继承 v2（含偏航修正）。
注意：v2 的偏航修正实测是**退步**（转向从 0.093 掉到 0.027，见报告 §4.3），
所以本档只用来验证执行器模型，不代表最终策略。
动作缩放是固定的 0.5，不随 ``effort_limit`` 变化，所以可以从 v2 的末轮热启动。
"""

from dataclasses import replace

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.pikachu.pikachu_v035_constants import (
  ARMATURE_J4340,
  DAMPING_J4340,
  STIFFNESS_J4340,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.pikachu_v035_biped_walk_v2.env_cfgs import (
  v035_biped_walk_v2_env_cfg,
)


def v035_biped_walk_v4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = v035_biped_walk_v2_env_cfg(play=play)
  robot = cfg.scene.entities["robot"]
  knee = DcMotorActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=STIFFNESS_J4340,
    damping=DAMPING_J4340,
    effort_limit=12.0,
    saturation_effort=40.0,
    velocity_limit=7.5,
    armature=ARMATURE_J4340,
  )
  actuators = tuple(
    knee if ".*_knee_joint" in a.target_names_expr else a
    for a in robot.articulation.actuators
  )
  cfg.scene.entities["robot"] = replace(
    robot, articulation=replace(robot.articulation, actuators=actuators)
  )
  return cfg
