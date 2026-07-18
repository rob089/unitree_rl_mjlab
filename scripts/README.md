# Scripts

Every script is a [tyro](https://brentyi.github.io/tyro/) CLI — run it with
`--help` for the full argument list. Run them from the repo root so `src` is
importable.

The end-to-end path from mocap to robot is:

```
LAFAN1 csv → csv_to_npz.py → train.py → export_policy.py → package_g1_23dof_mimic.py → g1_ctrl
                   ↓                        ↓                         ↓
             vis_motion.py              play.py            simulate/run_headless.sh
```

---

## Motion preparation

### `csv_to_npz.py`

Retargeted LAFAN1 CSV → the `.npz` the tracking task trains on. Resamples to
the training rate, computes velocities by finite differences, and replays the
result through MuJoCo so the stored body poses come from real forward
kinematics.

```bash
python scripts/csv_to_npz.py \
  --robot g1_23dof \
  --input-file /workspace/lafan1_g1/g1/fight1_subject3.csv \
  --output-name fight1_subject3_frames_220_260.npz \
  --input-fps 30 --output-fps 50 \
  --line-range "(220,260)"
```

Output goes to `src/assets/motions/<robot>/`.

- `--line-range` needs the parenthesised form `"(220,260)"`; two bare integers
  are rejected by tyro.
- `--transition-time` (default **0.75 s**) prepends a synthetic stand-to-first-frame
  transition, so phase 0 is the robot's default stand pose. Without it a clip cut
  from mid-performance starts in a pose the robot cannot be in at deploy time,
  and the policy is initialised far outside its training distribution — this was
  measured at 1.63 rad on the punch clip, diverging within 160 ms. Pass `0` to
  disable.
- `--render` also writes a video of the replay.

### `vis_motion.py`

Renders a motion CSV **or** an exported `.npz` to mp4, reusing the same
`MotionLoader` and scene as `csv_to_npz.py` — so what you watch is what training
consumes. Use it to sanity-check a cut before spending GPU hours on it.

```bash
python scripts/vis_motion.py \
  --input-file src/assets/motions/g1_23dof/fight1_subject3_frames_220_260.npz \
  --robot g1_23dof --output punch.mp4
```

Useful flags: `--max-seconds`, `--width/--height`, `--distance/--elevation/--azimuth`
for the tracking camera.

---

## Training

### `list_envs.py`

Lists registered task IDs. Takes an optional keyword filter.

```bash
python scripts/list_envs.py                      # everything
python scripts/list_envs.py --keyword tracking   # filtered
```

The two tracking variants for this robot:

| task | actor observations |
|---|---|
| `Unitree-G1-23Dof-Tracking` | includes `base_lin_vel` and `motion_anchor_pos_b` — requires a state estimator at deploy, **not implemented in `deploy/`** |
| `Unitree-G1-23Dof-Tracking-No-State-Estimation` | proprioception only; this is what the shipped policies use |

### `train.py`

The task ID is a **positional** argument, not a flag — everything after it is
parsed against that task's config:

```bash
python scripts/train.py Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion-file src/assets/motions/g1_23dof/fight1_subject3_frames_220_260.npz
```

Notable options: `--gpu-ids` (list, or `all`, for multi-GPU via torchrunx),
`--video` with `--video-length` / `--video-interval`, `--enable-nan-guard`.
Checkpoints and an ONNX export land in `logs/rsl_rl/<task>/<timestamp>/`.

The whole env config is reachable from the CLI, so most experiments need no
code edit. Two families worth knowing about, both currently left at defaults:

```bash
# observation history - lets the policy infer what a single frame cannot
--env.observations.actor.terms.joint-pos.history-length 5

# sensing latency - the real stack has DDS plus the motor loop
--env.observations.actor.terms.joint-pos.delay-max-lag 2
```

Changing the observation layout also changes the ONNX input width; the deploy
side follows automatically, since `deploy.yaml` is generated from the run's env
cfg rather than a template.

### `play.py`

Replays a trained checkpoint in the viewer.

Task ID is positional here too:

```bash
python scripts/play.py <TaskId> --checkpoint-file logs/rsl_rl/.../model_2000.pt
```

Loads from a local file, a `--wandb-run-path`, or a `--registry-name`. Also
`--num-envs`, `--video`, `--camera`, `--viewer {auto,native,viser}`,
`--no-terminations`. Note it does **not** export ONNX — use `export_policy.py`.

### `visualize_terrain.py`

Standalone terrain preview. Not used by the tracking tasks, which train on a
plane.

---

## Deployment

### `export_policy.py`

Exports `policy.onnx` from **any** checkpoint. Training only exports at the end,
so this is how you recover a policy from a specific iteration — useful when the
final iterations regressed or a run was interrupted.

```bash
python scripts/export_policy.py \
  --task-id Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --checkpoint logs/rsl_rl/g1_23dof_tracking/<run>/model_2000.pt \
  --output-dir deploy/robots/g1_23dof/config/policy/mimic/<name>/v0/exported
```

Builds a minimal env only to establish input dimensions, so `--device cpu` is
fine. The observation normalizer is part of the actor and is baked into the
exported graph as leading `Sub`/`Div` nodes.

### `package_g1_23dof_mimic.py`

Assembles a deployable mimic bundle (ONNX + `deploy.yaml` + reference `.npz`)
and registers the FSM state in `deploy/robots/g1_23dof/config/config.yaml`.

```bash
python scripts/package_g1_23dof_mimic.py \
  --log-dir logs/rsl_rl/g1_23dof_tracking/<run> \
  --motion-file src/assets/motions/g1_23dof/fight1_subject3_frames_220_260.npz \
  --name fight_punch \
  --trigger "RB + X.on_pressed"
```

`--trigger` is a joystick DSL expression (see `deploy/include/unitree_joystick_dsl.hpp`);
omit it and the script prompts. `--overwrite` replaces an existing bundle.

`deploy.yaml` is generated from the run's own `params/env.yaml` (see
`gen_deploy_yaml.py`), so gains, action scales and the observation layout always
match what was trained. A run without a dumped env cfg is rejected rather than
falling back to a template.

### `gen_deploy_yaml.py`

Builds a `deploy.yaml` from a training run. Called automatically by the
packaging script; run it directly to inspect what a run would deploy as:

```bash
python scripts/gen_deploy_yaml.py --run-dir logs/rsl_rl/g1_23dof_tracking/<run>
```

An observation term whose function has no C++ implementation (`base_lin_vel`
and `motion_anchor_pos_b`, i.e. the state-estimation variant) fails here with a
clear error instead of at the robot.

---

## Testing in simulation

Not a script, but the other half of the loop — see `simulate/`:

```bash
./simulate/run_headless.sh -p 8080 --paused        # terminal 1
./deploy/robots/g1_23dof/build/g1_ctrl -n lo       # terminal 2
```

Then in terminal 1: `1` FixStand → `g` resume → `2` Velocity → `r` reset (drops
the robot standing with the policy live) → `w/a/s/d`, `q/e` to drive, `7/8/9` for
the mimic motions. View at `http://localhost:8080` through
`ssh -L 8080:localhost:8080 <server>`.

Run only one simulator per DDS domain — several publishing `rt/lowstate`
simultaneously will silently clobber each other's joystick bytes.
