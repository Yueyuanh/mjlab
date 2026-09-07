"""构造 27dof 布局的 walk reference npz。

从 walk_base 纯腿前进参考(14dof body 布局)生成一个按 27-dof 机器人 body 顺序
存储的 npz,使得 tracking 27dof 的 ``MotionLoader``(用当前机器人 body id 索引)
能把 base+双腿的参考正确切片到 27dof 布局的对应行。头/耳/尾/手臂 body 不确定,
填 base 值(不被观测/奖励引用);腿关节填 walk 参考,其余关节(头/耳/尾/臂,由
RandomFlail 干扰)填 0(限位中位、正位)。

用法: uv run python <此脚本> <输出目录>
"""
import sys

import numpy as np

SRC = "pikachu_walk_0.185_base.npz"
SRC_DIR = "src/mjlab/mocap/npz/pikachu_walk_base_npz"

# 27dof 机器人 body 顺序(索引 = 布局列)。来自 pakachu_27dof_train.xml。
BODY_27 = [
  "base_link", "head_pitch_link", "head_yaw_link", "head_roll_link",
  "left_ear_pitch_link", "left_ear_roll_link", "right_ear_pitch_link",
  "right_ear_roll_link", "tail_pitch_link", "tail_yaw_link",
  "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
  "left_knee_link", "left_ankle_link", "right_hip_pitch_link",
  "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
  "right_ankle_link", "left_arm_pitch_link", "left_arm_roll_link",
  "left_arm_yaw_link", "left_elbow_link", "right_arm_pitch_link",
  "right_arm_roll_link", "right_arm_yaw_link", "right_elbow_link",
]

# 27dof 关节顺序(索引 = dof 列)。
JOINT_27 = [
  "head_pitch_joint", "head_yaw_joint", "head_roll_joint",
  "left_ear_pitch_joint", "left_ear_roll_joint", "right_ear_pitch_joint",
  "right_ear_roll_joint", "tail_pitch_joint", "tail_yaw_joint",
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_joint", "right_hip_pitch_joint",
  "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_joint", "left_arm_pitch_joint", "left_arm_roll_joint",
  "left_arm_yaw_joint", "left_elbow_joint", "right_arm_pitch_joint",
  "right_arm_roll_joint", "right_arm_yaw_joint", "right_elbow_joint",
]

# 13dof 源 body 顺序(flat_14dof_lite):base + 双腿5 + 双臂3。
BODY_SRC = [
  "base_link",
  "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
  "left_knee_link", "left_ankle_link",
  "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
  "right_knee_link", "right_ankle_link",
  "left_arm_pitch_link", "left_arm_roll_link", "left_arm_yaw_link",
  "right_arm_pitch_link", "right_arm_roll_link", "right_arm_yaw_link",
]

# 腿 10(顺序与 LEG_JOINT_NAMES 一致)。
LEG_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_joint",
]

# 从 27dof body 中复用 walk 参考的 body:base + legs。
# 27dof 布局中 id,对应 SRC 布局中的 id。
_27_of = {b: i for i, b in enumerate(BODY_27)}
_src_of = {b: i for i, b in enumerate(BODY_SRC)}
# LEG body(7):base + 双腿 hip_pitch/knee/ankle(与 walk_base env_cfg 一致)。
LEG_BODIES = (
  "base_link",
  "left_hip_pitch_link", "left_knee_link", "left_ankle_link",
  "right_hip_pitch_link", "right_knee_link", "right_ankle_link",
)
BODY_MAP = {_27_of[b]: _src_of[b] for b in LEG_BODIES}  # 27dof col -> src col
JOINT_MAP = {JOINT_27.index(j): LEG_JOINTS.index(j) for j in LEG_JOINTS}


def main(out_dir: str) -> None:
  import os
  os.makedirs(out_dir, exist_ok=True)
  d = np.load(f"{SRC_DIR}/{SRC}")
  n_frames = d["joint_pos"].shape[0]
  n_body = len(BODY_27)
  n_joint = len(JOINT_27)

  bp = np.zeros((n_frames, n_body, 3), dtype=np.float32)
  bq = np.zeros((n_frames, n_body, 4), dtype=np.float32)
  bl = np.zeros((n_frames, n_body, 3), dtype=np.float32)
  ba = np.zeros((n_frames, n_body, 3), dtype=np.float32)
  for col27, col_src in BODY_MAP.items():
    bp[:, col27] = d["body_pos_w"][:, col_src]
    bq[:, col27] = d["body_quat_w"][:, col_src]
    bl[:, col27] = d["body_lin_vel_w"][:, col_src]
    ba[:, col27] = d["body_ang_vel_w"][:, col_src]
  # 非 base+腿 body 填 base 值(不参与跟踪,仅补齐布局)。
  base_col = BODY_27.index("base_link")
  for col27 in range(n_body):
    if col27 not in BODY_MAP:
      bp[:, col27] = d["body_pos_w"][:, BODY_MAP.get(base_col, 0)]
      bq[:, col27] = d["body_quat_w"][:, BODY_MAP.get(base_col, 0)]
      bl[:, col27] = 0.0
      ba[:, col27] = d["body_ang_vel_w"][:, BODY_MAP[base_col]]

  # 参考只有腿上肢？不，只跟踪腿 10 关节：npz 关节直接取 walk_base 的纯腿 10 列
  # （列序即 LEG_JOINT_NAMES 顺序，与 robot.find_joints(LEG_JOINT_NAMES) 的
  # joint_ids 一一对应）。上肢关节由 RandomFlail 独立干扰，不进参考。
  jp = d["joint_pos"].astype(np.float32)
  jv = d["joint_vel"].astype(np.float32)

  out = f"{out_dir}/pikachu_walk_0.185_27dof.npz"
  np.savez(
    out,
    fps=d["fps"],
    joint_pos=jp, joint_vel=jv,
    body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=bl, body_ang_vel_w=ba,
  )
  print(
    f"written {out}\n  frames={n_frames} body={n_body} "
    f"joint={jp.shape[1]} (legs-only {len(LEG_JOINTS)})  vel={jv.shape[1]}\n"
    f"  body_map(col27->src): {BODY_MAP}"
  )


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "src/mjlab/mocap/npz/pikachu_walk_27dof_npz")