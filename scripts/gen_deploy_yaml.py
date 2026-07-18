"""Generate a deploy.yaml from the env config a policy was actually trained with.

The deployment runtime reads gains, action scales, the joint mapping and the
observation layout from deploy.yaml. Those numbers exist already in the env cfg
dumped next to every training run (``logs/<run>/params/env.yaml``), so deriving
them is strictly better than maintaining a hand-written template that silently
desyncs the moment the training config changes.

The failure this prevents is quiet: the observation vector keeps a plausible
length while its contents are re-ordered, and the policy simply behaves badly.

Usage:
    python scripts/gen_deploy_yaml.py \
        --run-dir logs/rsl_rl/g1_23dof_tracking/<run> \
        --output  deploy/robots/g1_23dof/config/policy/mimic/<name>/v0/params/deploy.yaml
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tyro
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Unitree G1 SDK motor order. The 29-DoF deploy config maps policy index to
# motor index with the identity, so this list *is* the SDK ordering, and any
# reduced robot's joint_ids_map is just its joint names looked up here.
G1_SDK_MOTOR_ORDER = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Training observation function -> name registered in the deploy runtime
# (REGISTER_OBSERVATION in deploy/). A term whose function is not listed here
# has no C++ implementation and cannot be deployed.
FUNC_TO_DEPLOY_TERM = {
  "generated_commands": "motion_command",
  "motion_anchor_ori_b": "motion_anchor_ori_b",
  "joint_pos_rel": "joint_pos_rel",
  "joint_vel_rel": "joint_vel_rel",
  "last_action": "last_action",
  "projected_gravity": "projected_gravity",
  "velocity_commands": "velocity_commands",
  "gait_phase": "gait_phase",
}
# builtin_sensor terms are distinguished by which sensor they read.
SENSOR_TO_DEPLOY_TERM = {
  "robot/imu_ang_vel": "base_ang_vel",
}

# Params the deploy runtime actually consumes. Training-side params (sensor_name,
# biased, ...) are meaningless there and are dropped. The params key itself is
# always emitted: ObservationManager decides whether the config is grouped by
# testing whether the first term defines it.
DEPLOY_PARAM_KEYS = {"command_name"}


class _TolerantLoader(yaml.SafeLoader):
  """Reads the dumped cfg without importing the python objects it references."""


def _unknown(loader, suffix, node):  # noqa: ANN001
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node)
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node)
  # mjlab dumps callables as `!!python/name:module.func ''` - the identity is in
  # the tag, and the scalar body is empty.
  value = loader.construct_scalar(node)
  return value if value else suffix


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _unknown)
_TolerantLoader.add_multi_constructor("!", _unknown)


_UNSET = object()


def _match(patterns: dict[str, Any], joint: str, what: str, default: Any = _UNSET) -> Any:
  """Resolve a regex-keyed map (as mjlab dumps them) for one joint."""
  for pattern, value in patterns.items():
    if re.fullmatch(pattern, joint):
      return value
  if default is not _UNSET:
    return default
  raise KeyError(f"no {what} matches joint {joint!r}")


def _actuator_for(actuators: list[dict], joint: str) -> dict:
  for group in actuators:
    for pattern in group["target_names_expr"]:
      if re.fullmatch(pattern, joint):
        return group
  raise KeyError(f"no actuator group matches joint {joint!r}")


def joint_order_from_xml(xml_path: Path) -> list[str]:
  """Policy joint order is the order the joints appear in the compiled model."""
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  names = []
  for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
      names.append(name)
  return names


def build(run_dir: Path, robot_xml: Path) -> dict:
  env = yaml.load((run_dir / "params" / "env.yaml").read_text(), Loader=_TolerantLoader)

  robot = env["scene"]["entities"]["robot"] if "entities" in env["scene"] else None
  if robot is None:  # older dumps nest the robot directly
    robot = next(v for k, v in env["scene"].items() if isinstance(v, dict) and "articulation" in v)
  actuators = robot["articulation"]["actuators"]
  default_pos_patterns = robot["init_state"]["joint_pos"]

  joints = joint_order_from_xml(robot_xml)

  action_cfg = env["actions"]["joint_pos"]
  scale_patterns = action_cfg["scale"]

  stiffness, damping, default_pos, scale = [], [], [], []
  for joint in joints:
    act = _actuator_for(actuators, joint)
    stiffness.append(float(act["stiffness"]))
    damping.append(float(act["damping"]))
    default_pos.append(float(_match(default_pos_patterns, joint, "default joint_pos", 0.0)))
    scale.append(float(_match(scale_patterns, joint, "action scale")))

  missing = [j for j in joints if j not in G1_SDK_MOTOR_ORDER]
  if missing:
    raise KeyError(f"joints absent from the SDK motor order: {missing}")
  joint_ids_map = [G1_SDK_MOTOR_ORDER.index(j) for j in joints]

  observations, gym_history = {}, None
  for name, term in env["observations"]["actor"]["terms"].items():
    func = str(term["func"]).rsplit(".", 1)[-1].strip(" '")
    if func == "builtin_sensor":
      sensor = term.get("params", {}).get("sensor_name")
      deploy_name = SENSOR_TO_DEPLOY_TERM.get(sensor)
      if deploy_name is None:
        raise KeyError(
          f"observation {name!r} reads sensor {sensor!r}, which has no deploy "
          f"implementation. Train a variant that does not use it."
        )
    else:
      deploy_name = FUNC_TO_DEPLOY_TERM.get(func)
      if deploy_name is None:
        raise KeyError(
          f"observation {name!r} uses {func!r}, which is not registered in the "
          f"deploy runtime (see REGISTER_OBSERVATION in deploy/)."
        )

    history = int(term.get("history_length") or 0) or 1
    if history > 1:
      # mjlab flattens each term's own history and then concatenates terms;
      # the deploy runtime does the same only when use_gym_history is false.
      flat = bool(term.get("flatten_history_dim", True))
      gym_history = not flat if gym_history is None else gym_history

    params = term.get("params") or {}
    entry: dict[str, Any] = {
      "params": {k: v for k, v in params.items() if k in DEPLOY_PARAM_KEYS}
    }
    entry["clip"] = term.get("clip")
    entry["scale"] = term.get("scale")
    entry["history_length"] = history
    observations[deploy_name] = entry

  if gym_history is not None:
    observations = {"use_gym_history": gym_history, **observations}

  return {
    "joint_ids_map": joint_ids_map,
    "step_dt": float(env["decimation"]) * float(env["sim"]["mujoco"]["timestep"]),
    "stiffness": stiffness,
    "damping": damping,
    "default_joint_pos": default_pos,
    "commands": {},
    "actions": {
      "JointPositionAction": {
        "clip": action_cfg.get("clip"),
        "joint_names": [".*"],
        "scale": scale,
        # use_default_offset=True in training, so the offset is the default pose.
        "offset": default_pos,
        "joint_ids": None,
      }
    },
    "observations": observations,
  }


def main(
  run_dir: str,
  output: str | None = None,
  robot_xml: str = "src/assets/robots/unitree_g1/xmls/g1_23dof.xml",
) -> None:
  """Write a deploy.yaml derived from a training run's env cfg.

  Args:
    run_dir: Training run directory containing params/env.yaml.
    output: Destination file. Prints to stdout when omitted.
    robot_xml: Robot model defining the policy joint order.
  """
  cfg = build(Path(run_dir).expanduser().resolve(), REPO_ROOT / robot_xml)
  text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=100)
  if output:
    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")
  else:
    print(text)


if __name__ == "__main__":
  tyro.cli(main)
