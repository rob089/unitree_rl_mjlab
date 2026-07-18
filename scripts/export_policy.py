"""Export a policy.onnx from a specific checkpoint without running the full environment."""

import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

# ---------------------------------------------------------------------------
# Bootstrap: ensure the workspace src/ is importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab.tasks  # noqa: F401  – populates the task registry
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main(
    task_id: str,
    checkpoint: str,
    output_dir: str | None = None,
    output_name: str = "policy.onnx",
    num_envs: int = 1,
    device: str = "cpu",
) -> None:
    """Export a policy ONNX from an arbitrary checkpoint.

    Args:
        task_id:     Registered task name, e.g. ``Unitree-G1-23Dof-Flat``.
        checkpoint:  Path to the ``.pt`` checkpoint file.
        output_dir:  Directory where the ONNX file is written.
                     Defaults to the same directory as the checkpoint.
        output_name: Filename for the exported model (default: ``policy.onnx``).
        num_envs:    Number of parallel envs used to build the runner (1 is fine).
        device:      Torch device string (``cpu`` or ``cuda:0`` etc.).
    """
    configure_torch_backends()

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(output_dir).resolve() if output_dir else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task        : {task_id}")
    print(f"[INFO] Checkpoint  : {checkpoint_path}")
    print(f"[INFO] Output dir  : {out_dir}")
    print(f"[INFO] Output name : {output_name}")

    # ------------------------------------------------------------------
    # Build a minimal environment (needed to define network input dims).
    # ------------------------------------------------------------------
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = num_envs

    agent_cfg = load_rl_cfg(task_id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # Build runner, load checkpoint weights.
    # ------------------------------------------------------------------
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )

    # ------------------------------------------------------------------
    # Export ONNX.
    # ------------------------------------------------------------------
    runner.export_policy_to_onnx(str(out_dir), output_name)
    print(f"[INFO] Exported: {out_dir / output_name}")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
