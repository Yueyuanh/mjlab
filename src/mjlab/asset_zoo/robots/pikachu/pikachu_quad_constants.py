"""Pikachu 四足(quadruped)全向 velocity 专用常量。

独立于共享 ``pikachu_constants.py``:这里只 **import 其只读值引用**(碰撞、
执行器、脚正则),但**模型 XML 用四足专属版** —— 不碰共享 cycle 用的
``Pikachu_V025_flat_*``。保证双足/其余任务完全不受影响。

四足构型:后腿(``hip_pitch/roll/yaw/knee/ankle``,5 dof×2)+ 前腿 = 手臂
(``arm_pitch/arm_roll``,2 dof×2)。模型用四足专属 XML
``Pikachu_V025_quad_14dof_capsule.xml``;起步姿态仍复用共享 ``QUAD_KEYFRAME``
(手脚四点支撑、关节摆出四足趴姿)。

碰撞:复用共享 ``FULL_COLLISION_WITHOUT_SELF`` —— 脚(foot1~7)与手
(``arm_yaw_link_collision``)均配 condim=3 + friction、且无自碰撞,正是四足
手脚撑地所需,避免握拳式点接触打滑(新 XML 保留了同名的这些 geom/site)。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants import (
  FULL_COLLISION_WITHOUT_SELF,
  PIKACHU_ARTICULATION,
  PIKACHU_FOOT_GEOM_NAMES,
  QUAD_KEYFRAME,
)
from mjlab.entity import EntityCfg

__all__ = [
  "QUAD_XML",
  "QUAD_START",
  "QUAD_TORSO_BODY",
  "QUAD_HAND_GEOM_NAMES",
  "QUAD_SUPPORT_GEOM_NAMES",
  "QUAD_ACTION_SCALE",
  "get_quad_spec",
  "get_pikachu_quad_robot_cfg",
]

# 四足专属 MJCF:碰撞体为 capsule。
QUAD_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pikachu"
  / "xmls"
  / "mjcf_14dof_lite"
  / "Pikachu_V025_quad_14dof_capsule.xml"
)
assert QUAD_XML.exists()


def get_quad_spec() -> mujoco.MjSpec:
  """从四足 XML 编译出 MuJoCo 规格(每次调用返回新实例)。"""
  return mujoco.MjSpec.from_file(str(QUAD_XML))


# 起步姿态:复用共享 QUAD_KEYFRAME(手脚四点支撑、关节摆出四足趴姿)。
# 落地配平的微调项(rot/arm_pitch/hip_pitch/pos.z)若需改,copy 一份到这里改,
# 不动共享 keyframe,保证双足/其余任务不受影响。
QUAD_START = QUAD_KEYFRAME

# 四足躯干(基座)body 名(与共享集中不同)。
QUAD_TORSO_BODY = "base_link"

# 四足前腿 = 手的支撑几何:手掌(arm_yaw_link)撑地。
QUAD_HAND_GEOM_NAMES = r"^(left|right)_arm_yaw_link_collision$"

# 四足 4 点支撑的末端接地 geom 集:后脚胶囊 + 双手手掌。供脚接地传感器
# (feet_ground_contact) 与 foot_friction 事件匹配。
QUAD_SUPPORT_GEOM_NAMES = (
  PIKACHU_FOOT_GEOM_NAMES,  # 后脚(left/right foot1~7 capsule)
  QUAD_HAND_GEOM_NAMES,  # 双手撑地
)

# 四足动作缩放:双足 velocity 已验证的放宽值(hip_pitch/knee ±0.5、
# roll/yaw/ankle/arm ±0.35)是足够的迈步幅度;四足时前腿(arm)作为前腿要
# 能抬/落,arm_pitch 放宽到 0.5 与 hip_pitch 同档。
QUAD_ACTION_SCALE: dict[str, float] = {
  r".*_hip_pitch_joint": 1.0,
  r".*_knee_joint": 1.0,
  r".*_hip_roll_joint": 0.35,
  r".*_hip_yaw_joint": 0.35,
  r".*_ankle_joint": 0.5,
  r".*_arm_pitch_joint": 1.0,  # 前腿(手)抬脚幅度
  r".*_arm_roll_joint": 0.35,
}


def get_pikachu_quad_robot_cfg() -> EntityCfg:
  """获取四足 velocity 机器人配置(四足 XML 默认姿态,可安全修改)。"""
  return EntityCfg(
    init_state=QUAD_START,
    collisions=(FULL_COLLISION_WITHOUT_SELF,),
    spec_fn=get_quad_spec,
    articulation=PIKACHU_ARTICULATION,
  )
