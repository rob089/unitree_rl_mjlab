"""Package a G1 23-DoF tracking run for the native mimic controller."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import tyro
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "deploy" / "robots" / "g1_23dof"
CONFIG_PATH = ROBOT_DIR / "config" / "config.yaml"
TEMPLATE_DIR = ROBOT_DIR / "config" / "policy" / "mimic" / "frogger" / "v0"


def _slug(value: str) -> str:
  slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
  if not slug:
    raise ValueError("The bundle name must contain at least one letter or number.")
  return slug


def _next_state_id(states: dict) -> int:
  used_ids = {
    config["id"]
    for config in states.values()
    if isinstance(config, dict) and isinstance(config.get("id"), int)
  }
  state_id = 1
  while state_id in used_ids:
    state_id += 1
  return state_id


def _ask_for_trigger() -> str:
  print("Enter the Velocity-mode trigger for this mimic policy.")
  print("Examples: RB + A.on_pressed, RB + X.on_pressed, LT(2s) + up.on_pressed")
  print("Supported buttons: A B X Y LB RB LT RT LS RS start back up down left right.")
  trigger = input("Trigger [RB + A.on_pressed]: ").strip()
  return trigger or "RB + A.on_pressed"


def main(
  log_dir: str,
  motion_file: str,
  name: str | None = None,
  trigger: str | None = None,
  overwrite: bool = False,
) -> None:
  """Create a deployable mimic-policy bundle and register it in config.yaml.

  The controller loads ``policy.onnx`` and ``deploy.yaml`` from the bundle and
  the reference NPZ from the same bundle. Entering the mimic state is possible
  from Velocity mode through the supplied joystick DSL expression.

  Args:
    log_dir: Directory containing the trained tracking policy ``policy.onnx``.
    motion_file: Reference motion NPZ used to train the policy.
    name: Bundle name. Defaults to the motion filename without ``.npz``.
    trigger: Joystick DSL expression, such as ``RB + A.on_pressed``. If omitted,
      the script asks interactively.
    overwrite: Replace an existing bundle and FSM state of the same name.
  """
  run_dir = Path(log_dir).expanduser().resolve()
  source_policy = run_dir / "policy.onnx"
  source_motion = Path(motion_file).expanduser().resolve()
  if not source_policy.is_file():
    raise FileNotFoundError(f"Actor policy not found: {source_policy}")
  if not source_motion.is_file():
    raise FileNotFoundError(f"Motion NPZ not found: {source_motion}")
  if source_motion.suffix != ".npz":
    raise ValueError("motion_file must be an exported .npz motion file.")
  if not TEMPLATE_DIR.is_dir():
    raise FileNotFoundError(f"Deployment template not found: {TEMPLATE_DIR}")

  bundle_name = _slug(name or source_motion.stem)
  state_name = f"Mimic_{bundle_name}"
  trigger = trigger.strip() if trigger is not None else _ask_for_trigger()
  if not trigger:
    raise ValueError("The trigger expression cannot be empty.")

  bundle_dir = ROBOT_DIR / "config" / "policy" / "mimic" / bundle_name / "v0"
  if bundle_dir.exists() and not overwrite:
    raise FileExistsError(
      f"Bundle already exists: {bundle_dir}. Re-run with --overwrite True to replace it."
    )

  with CONFIG_PATH.open() as file:
    config = yaml.safe_load(file)
  states = config["FSM"]["_"]
  if state_name in states and not overwrite:
    raise ValueError(
      f"FSM state {state_name!r} already exists. Re-run with --overwrite True to update it."
    )

  if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
  (bundle_dir / "exported").mkdir(parents=True)
  (bundle_dir / "params").mkdir(parents=True)
  shutil.copy2(source_policy, bundle_dir / "exported" / "policy.onnx")
  source_policy_data = source_policy.with_name("policy.onnx.data")
  if source_policy_data.is_file():
    shutil.copy2(source_policy_data, bundle_dir / "exported" / "policy.onnx.data")
  shutil.copy2(TEMPLATE_DIR / "params" / "deploy.yaml", bundle_dir / "params")
  deployed_motion = bundle_dir / "params" / source_motion.name
  shutil.copy2(source_motion, deployed_motion)

  velocity = config["FSM"].setdefault("Velocity", {})
  transitions = velocity.setdefault("transitions", {})
  transitions[state_name] = trigger
  config["FSM"][state_name] = {
    "transitions": {
      "Passive": "LT + B.on_pressed",
      "Velocity": "RT + A.on_pressed",
    },
    "motion_file": str(deployed_motion.relative_to(ROBOT_DIR)),
    "policy_dir": str(bundle_dir.relative_to(ROBOT_DIR)),
    "time_start": 0.0,
    "time_end": 500.0,
  }
  states[state_name] = {"id": _next_state_id(states), "type": "Mimic"}

  backup_path = CONFIG_PATH.with_suffix(".yaml.bak")
  shutil.copy2(CONFIG_PATH, backup_path)
  with CONFIG_PATH.open("w") as file:
    yaml.safe_dump(config, file, sort_keys=False)

  print(f"[INFO] Bundle created: {bundle_dir}")
  print(f"[INFO] FSM state: {state_name} (id={states[state_name]['id']})")
  print(f"[INFO] Enter from Velocity with: {trigger}")
  print(f"[INFO] Config backup: {backup_path}")


if __name__ == "__main__":
  tyro.cli(main)