"""生成 walk-base 无臂 npz(腿 10 dof)。

读 ``npz/pikachu_walk_npz/pikachu_walk_0.185_0.0_0.0.npz``(14 dof),把
``joint_pos`` / ``joint_vel`` 裁到前 10 列(腿序 = ``PIKACHU_JOINT_NAMES[0:10]``),
``body_*`` 与 ``fps`` 原样保留,存为 ``npz/pikachu_walk_base_npz/pikachu_walk_0.185_base.npz``。
一次生成,手动在需要重建时运行。
"""

import numpy as np

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pikachu.pikachu_constants_base import (
  LEG_JOINT_NAMES,
)

SRC = (
  MJLAB_SRC_PATH
  / "mocap"
  / "npz"
  / "pikachu_walk_npz"
  / "pikachu_walk_0.185_0.0_0.0.npz"
)
DST_DIR = MJLAB_SRC_PATH / "mocap" / "npz" / "pikachu_walk_base_npz"
DST = DST_DIR / "pikachu_walk_0.185_base.npz"


def main() -> None:
  with np.load(SRC) as data:
    n_legs = len(LEG_JOINT_NAMES)
    out = {
      "fps": data["fps"],
      "joint_pos": data["joint_pos"][:, :n_legs],
      "joint_vel": data["joint_vel"][:, :n_legs],
      "body_pos_w": data["body_pos_w"],
      "body_quat_w": data["body_quat_w"],
      "body_lin_vel_w": data["body_lin_vel_w"],
      "body_ang_vel_w": data["body_ang_vel_w"],
    }
  DST_DIR.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(DST, **out)
  print(f"wrote {DST}")
  print(
    f"joint_pos/vel shape = {out['joint_pos'].shape} "
    f"(expected T,{n_legs}); fps = {out['fps']}"
  )


if __name__ == "__main__":
  main()
