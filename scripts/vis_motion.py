#!/usr/bin/env python3
"""
Visualize a single LAFAN1-retargeted motion CSV as a 3D MuJoCo replay.

Reuses the repo's own MotionLoader + G1 scene config + mjlab OffscreenRenderer,
so the mesh, joint mapping, and fps interpolation are identical to csv_to_npz.py
-- but it ONLY renders a video, it does not write an npz.

Place this in the repo's scripts/ folder and run from the repo root, e.g.:

    python scripts/vis_motion.py \
        --input-file /workspace/lafan1_g1/g1/run1_subject2.csv \
        --robot g1 \
        --output run1_subject2.mp4

Then scp the mp4 to your local machine to watch it.

Headless note: uses EGL offscreen rendering. If you hit a GL init error, run
    export MUJOCO_GL=egl
before invoking (your container already has the NVIDIA EGL libs mounted).
"""
import os
import sys
from pathlib import Path

# Make the repo importable no matter where this is launched from:
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))   # so `import src...` resolves
sys.path.insert(0, str(_SCRIPTS))     # so `import csv_to_npz` resolves

# Default to EGL offscreen GL before mujoco is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import (
    unitree_g1_23dof_flat_tracking_env_cfg,
)

# Reuse the exact loader from the conversion script (identical CSV parsing,
# wxyz reorder, and fps interpolation).
from csv_to_npz import MotionLoader


G1_29DOF_JOINTS = [
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

G1_23DOF_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
]


def main(
    input_file: str,
    robot: str = "g1",
    output: str = "motion_preview.mp4",
    input_fps: float = 30.0,
    output_fps: float | None = None,
    max_seconds: float | None = None,
    device: str = "cuda:0",
    width: int = 640,
    height: int = 480,
    distance: float = 2.0,
    elevation: float = -5.0,
    azimuth: float = 20.0,
):
    """Render a motion CSV or exported NPZ to an mp4 using the G1 MuJoCo model.

    Args:
        input_file: Path to a LAFAN1-format motion CSV or exported motion NPZ.
        robot: "g1" (29 DoF) or "g1_23dof" (23 DoF).
        output: Output video path (.mp4).
        input_fps: Frame rate of the CSV (LAFAN1 dataset is 30).
        output_fps: Playback/interpolation frame rate. Defaults to the input rate.
        device: Torch device.
        width, height: Render resolution.
        distance, elevation, azimuth: Tracking-camera placement.
    """
    if robot == "g1":
        scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
        joint_names = G1_29DOF_JOINTS
    elif robot == "g1_23dof":
        scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
        joint_names = G1_23DOF_JOINTS
    else:
        raise ValueError(f"Unsupported robot: {robot!r} (use 'g1' or 'g1_23dof')")

    is_npz = Path(input_file).suffix.lower() == ".npz"
    if is_npz:
        motion_data = np.load(input_file)
        input_rate = float(np.asarray(motion_data["fps"]).item())
    else:
        motion_data = None
        input_rate = input_fps
    render_fps = output_fps or input_rate

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / render_fps

    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    viewer_cfg = ViewerConfig(
        height=height,
        width=width,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=distance,
        elevation=elevation,
        azimuth=azimuth,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

    robot_entity: Entity = scene["robot"]
    joint_idx = robot_entity.find_joints(joint_names, preserve_order=True)[0]
    if is_npz:
        assert motion_data is not None
        joint_positions = torch.as_tensor(
            motion_data["joint_pos"], device=sim.device, dtype=torch.float32
        )
        joint_velocities = torch.as_tensor(
            motion_data["joint_vel"], device=sim.device, dtype=torch.float32
        )
        body_positions = torch.as_tensor(
            motion_data["body_pos_w"], device=sim.device, dtype=torch.float32
        )
        body_orientations = torch.as_tensor(
            motion_data["body_quat_w"], device=sim.device, dtype=torch.float32
        )
        body_linear_velocities = torch.as_tensor(
            motion_data["body_lin_vel_w"], device=sim.device, dtype=torch.float32
        )
        body_angular_velocities = torch.as_tensor(
            motion_data["body_ang_vel_w"], device=sim.device, dtype=torch.float32
        )
        if joint_positions.shape[1] != len(joint_idx):
            raise SystemExit(
                f"NPZ has {joint_positions.shape[1]} joint positions, but "
                f"{robot} expects {len(joint_idx)}."
            )
        n_frames = joint_positions.shape[0]
    else:
        motion = MotionLoader(
            motion_file=input_file,
            input_fps=input_fps,
            output_fps=render_fps,
            device=sim.device,
            line_range=None,
        )
        n_csv_dof = motion.motion_dof_poss.shape[1]
        if n_csv_dof != len(G1_29DOF_JOINTS):
            raise SystemExit(
                f"Expected a 29-DoF LAFAN1 CSV ({len(G1_29DOF_JOINTS)} joint cols) "
                f"but got {n_csv_dof}. Is this the right dataset?"
            )
        col_sel_t = torch.as_tensor(
            [G1_29DOF_JOINTS.index(name) for name in joint_names],
            device=sim.device,
            dtype=torch.long,
        )
        n_frames = motion.output_frames

    scene.reset()
    frames = []
    if max_seconds is not None:
        n_frames = min(n_frames, int(max_seconds * render_fps))
    print(f"Rendering {n_frames} frames of "
          f"{os.path.basename(input_file)} ...")

    from tqdm import tqdm
    for frame_index in tqdm(range(n_frames), unit="frame", ncols=80):
        if is_npz:
            base_pos = body_positions[frame_index, 0].unsqueeze(0)
            base_rot = body_orientations[frame_index, 0].unsqueeze(0)
            base_lin_vel = body_linear_velocities[frame_index, 0].unsqueeze(0)
            base_ang_vel = body_angular_velocities[frame_index, 0].unsqueeze(0)
            dof_pos = joint_positions[frame_index].unsqueeze(0)
            dof_vel = joint_velocities[frame_index].unsqueeze(0)
        else:
            (base_pos, base_rot, base_lin_vel, base_ang_vel,
             dof_pos, dof_vel), _ = motion.get_next_state()

        root_states = robot_entity.data.default_root_state.clone()
        root_states[:, 0:3] = base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
        robot_entity.write_root_state_to_sim(root_states)

        joint_pos = robot_entity.data.default_joint_pos.clone()
        joint_vel = robot_entity.data.default_joint_vel.clone()
        if is_npz:
            joint_pos[:, joint_idx] = dof_pos
            joint_vel[:, joint_idx] = dof_vel
        else:
            joint_pos[:, joint_idx] = dof_pos[:, col_sel_t]
            joint_vel[:, joint_idx] = dof_vel[:, col_sel_t]
        robot_entity.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        renderer.update(sim.data)
        frames.append(renderer.render())

    if not output.endswith(".mp4"):
        output += ".mp4"
    imageio.mimsave(output, frames, fps=int(render_fps), macro_block_size=1)
    print(f"Saved -> {output}  ({len(frames)} frames @ {int(render_fps)} fps)")


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)