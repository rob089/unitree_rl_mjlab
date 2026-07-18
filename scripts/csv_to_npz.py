from typing import Any

import numpy as np
import torch
import tyro
import os
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
    transition_time: float = 0.0,
    default_dof_pos: dict[str, float] | None = None,
    default_root_height: float | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self.transition_time = transition_time
    self._load_motion()
    self._interpolate_motion()
    if transition_time > 0.0:
      self._prepend_stand_transition(default_dof_pos, default_root_height)
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _prepend_stand_transition(
    self,
    default_dof_pos: dict[str, float] | None,
    default_root_height: float | None,
  ):
    """Prepends a synthetic transition from a neutral stand into frame 0.

    LAFAN1 clips are cut out of continuous performances, so frame 0 is almost
    never a standstill. Training therefore only ever initialises the policy in
    poses it cannot reach on hardware, where the motion has to be entered from
    whatever pose the robot is already holding. Prepending the transition makes
    phase 0 the stand pose, so the policy learns the entry itself.

    Uses a cubic Hermite: zero velocity at the stand, and the clip's own initial
    velocity at the seam, so no discontinuity is introduced at either end. The
    segment is kinematic, not dynamically feasible - it only has to funnel the
    state into the reference motion.
    """
    n = int(round(self.transition_time * self.output_fps))
    if n <= 0:
      return
    if default_dof_pos is None or default_root_height is None:
      raise ValueError("transition_time > 0 requires default_dof_pos and default_root_height")

    dt = self.output_dt
    duration = n * dt

    # Endpoint state: the clip's first frame, with its own initial velocity.
    p1_dof = self.motion_dof_poss[0]
    p1_pos = self.motion_base_poss[0]
    q1 = self.motion_base_rots[0]
    if self.output_frames > 1:
      v1_dof = (self.motion_dof_poss[1] - p1_dof) / dt
      v1_pos = (self.motion_base_poss[1] - p1_pos) / dt
    else:
      v1_dof = torch.zeros_like(p1_dof)
      v1_pos = torch.zeros_like(p1_pos)

    # Start state: standing at the clip's starting ground position, facing the
    # same way. Joints not covered by the robot keep their frame-0 value.
    p0_dof = p1_dof.clone()
    for name, value in default_dof_pos.items():
      p0_dof[G1_29DOF_JOINTS.index(name)] = value
    p0_pos = p1_pos.clone()
    p0_pos[2] = default_root_height

    # Upright, yaw-only version of the clip's starting orientation.
    w, x, y, z = q1[0], q1[1], q1[2], q1[3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    q0 = torch.stack(
      [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)]
    ).to(q1.device)

    s = torch.arange(n, device=self.device, dtype=torch.float32) / n  # [0, 1)
    s = s.unsqueeze(1)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    dof = h00 * p0_dof + h10 * duration * 0.0 + h01 * p1_dof + h11 * duration * v1_dof
    pos = h00 * p0_pos + h10 * duration * 0.0 + h01 * p1_pos + h11 * duration * v1_pos

    # Rotation: smoothstep slerp, which is smooth enough for the entry.
    blend = (3 * s**2 - 2 * s**3).squeeze(1)
    rots = torch.zeros((n, 4), device=self.device, dtype=self.motion_base_rots.dtype)
    for i in range(n):
      rots[i] = quat_slerp(q0, q1, float(blend[i]))

    self.motion_dof_poss = torch.cat([dof.to(self.motion_dof_poss.dtype), self.motion_dof_poss])
    self.motion_base_poss = torch.cat([pos.to(self.motion_base_poss.dtype), self.motion_base_poss])
    self.motion_base_rots = torch.cat([rots, self.motion_base_rots])
    self.output_frames += n
    self.duration = (self.output_frames - 1) * self.output_dt

    print(
      f"Prepended {n} stand-transition frames ({self.transition_time:.2f}s); "
      f"total frames: {self.output_frames}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


G1_29DOF_JOINTS = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_path,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
  transition_time: float = 0.0,
):
  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # The stand pose the transition starts from is the robot's own default, so it
  # matches what FixStand holds at deploy time.
  default_joint_pos = robot.data.default_joint_pos[0, robot_joint_indexes]
  default_dof_pos = {
    name: float(default_joint_pos[i]) for i, name in enumerate(joint_names)
  }
  default_root_height = float(robot.data.default_root_state[0, 2])

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
    transition_time=transition_time,
    default_dof_pos=default_dof_pos,
    default_root_height=default_root_height,
  )
  if motion.motion_dof_poss.shape[1] != len(G1_29DOF_JOINTS):
    raise ValueError(
      f"Expected {len(G1_29DOF_JOINTS)} CSV joint columns, got "
      f"{motion.motion_dof_poss.shape[1]}"
    )
  source_joint_indexes = torch.tensor(
    [G1_29DOF_JOINTS.index(name) for name in joint_names], device=sim.device
  )

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos[:, source_joint_indexes]
    joint_vel[:, robot_joint_indexes] = motion_dof_vel[:, source_joint_indexes]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        np.savez(output_path, **log)  # type: ignore[arg-type]


def main(
  robot: str,
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  transition_time: float = 0.75,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
    transition_time: Seconds of synthetic stand-to-first-frame motion to prepend
      (0 disables). Makes phase 0 a pose the robot can actually be in at deploy.
  """
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  if robot == "g1":    # 29 Dof
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    joint_names=[
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]
    output_dir = "./src/assets/motions/g1"
  elif robot == "g1_23dof":
    scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
    joint_names=[    # 23 Dof
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
    ]
    output_dir = "./src/assets/motions/g1_23dof"
  else:
    raise ValueError(f"Unsupported robot: {robot}")

  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()
  os.makedirs(output_dir, exist_ok=True)
  if not output_name.endswith(".npz"):
    output_name += ".npz"
  output_path = os.path.join(output_dir, output_name)

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_path=output_path,
    render=render,
    line_range=line_range,
    renderer=renderer,
    transition_time=transition_time,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
