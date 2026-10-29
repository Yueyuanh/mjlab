"""Serialize physical and control parameters for the V033 review report."""

import json
from pathlib import Path

import mujoco

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main():
  results = {}
  for variant in ("E0", "Crawl"):
    task = f"Mjlab-Velocity-Flat-Pikachu-V033-{variant}"
    cfg = load_env_cfg(task)
    runner = load_rl_cfg(task)
    robot = Entity(cfg.scene.entities["robot"])
    model = robot.spec.compile()
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    results[task] = {
      "body_mass_kg": float(model.body_mass.sum()),
      "num_joints_excluding_free": model.njnt - 1,
      "num_actuators": model.nu,
      "physics_dt": cfg.sim.mujoco.timestep,
      "control_dt": cfg.sim.mujoco.timestep * cfg.decimation,
      "nominal_episode_s": cfg.episode_length_s,
      "registered_num_envs": cfg.scene.num_envs,
      "registered_iterations": runner.max_iterations,
      "action_scale": action.scale,
      "joint_limits_rad": {
        model.joint(j).name: model.jnt_range[j].tolist()
        for j in range(1, model.njnt)
        if model.jnt_limited[j]
      },
      "actuator_limits_nm": {
        model.actuator(a).name: model.actuator_forcerange[a].tolist()
        for a in range(model.nu)
      },
      "actuator_gain": model.actuator_gainprm[:, :3].tolist(),
      "actuator_bias": model.actuator_biasprm[:, :3].tolist(),
      "enabled_contact_geoms": [
        model.geom(g).name
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
      ],
      "reset_position": cfg.scene.entities["robot"].init_state.pos,
      "reset_quaternion_wxyz": cfg.scene.entities["robot"].init_state.rot,
      "reset_joint_positions": cfg.scene.entities["robot"].init_state.joint_pos,
      "sensors": {
        s.name: {
          "primary": s.primary.pattern,
          "body_order": robot.find_bodies(s.primary.pattern)[1],
        }
        for s in cfg.scene.sensors
        if isinstance(s, ContactSensorCfg) and s.primary.mode in ("body", "subtree")
      },
      "mujoco_version": mujoco.__version__,
    }
  path = Path("doc/pikachu_v033_experiments_20261019/config_audit/asset_audit.json")
  path.write_text(json.dumps(results, indent=2) + "\n")
  print(path)


if __name__ == "__main__":
  main()
