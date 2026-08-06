# Chapter 02 — Setup and Configuration

**Previous:** [01-project-overview.md](01-project-overview.md) | **Next:** [03-mujoco-and-physics.md](03-mujoco-and-physics.md)

---

## Installation

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r carolline_control/requirements.txt
```

Dependencies (`requirements.txt`):

| Package | Role |
|---------|------|
| `mujoco>=3.0` | Physics simulation |
| `numpy`, `PyYAML` | Core numerics and config |
| `matplotlib` | Trajectory plots |
| `imageio`, `imageio-ffmpeg` | Course video recording |

---

## Running simulations

```powershell
# Default hybrid mission (headless)
python -m carolline_control.main --no-viewer

# Autonomous course
python carolline_control/scripts/simulate_autonomous_course.py

# With video + charts
python carolline_control/scripts/simulate_autonomous_course.py --record
```

Use `--sensor-only` on any script that supports it for IMU-only state estimation (harder).

---

## Config loading pipeline

```
YAML file  →  config_loader.load_config()  →  ControllerConfig dataclass
```

**File:** [config_loader.py](../../config_loader.py)

- `load_config(path)` — typed `ControllerConfig` for controllers
- `load_raw_config(path)` — full YAML dict (sim-only keys like `timestep`, `log_path`)
- `build_mission_waypoints()` — auto-generates cardinal flight legs if `waypoints` omitted

Model paths in YAML are relative to **repository root** (`E:\CAROLLINE_PROJECT`), not `carolline_control/`.

---

## Main config: `config.yaml`

Grouped by purpose:

### Physical / model

| Key | Meaning |
|-----|---------|
| `model_path` | MuJoCo scene (`scene_cage.xml` with cage) |
| `cage_radius` | 0.40 m — spawn height offset, rolling geometry |
| `mass`, `inertia` | Set at runtime from MuJoCo if null |
| `rotor_positions` | Body-frame motor sites for mixer |
| `motor_min`, `motor_max` | ±13 N bidirectional thrust limits |

### Flight position control (Lee geometric)

| Key | Default | Role |
|-----|---------|------|
| `kx`, `kv` | 4.5, 6.0 | Horizontal position/velocity gains |
| `kx_z`, `kv_z` | 8.0, 5.5 | Vertical gains |
| `kR`, `kOmega` | 5.5, 1.6 | Attitude tracking (airborne) |
| `kR_pre`, `kOmega_pre` | 3.5, 2.0 | Attitude during recovery/rolling |

### Rolling / ground

| Key | Role |
|-----|------|
| `rolling_max_speed` | Speed cap (m/s) |
| `rolling_kp`, `rolling_kd` | Outer velocity loop |
| `rolling_omega_kp` | Contact-frame rate tracking |
| `ground_omega_kp`, `ground_omega_weights` | Pre-takeoff recovery |
| `upright_cos_threshold` | cos(angle) for "upright" (~0.92) |
| `motor_slew_rate` | Max thrust change per second per motor |

### Mission (default `main.py` mission)

| Key | Role |
|-----|------|
| `mission.roll_target` | Ground roll goal XY |
| `mission.leg_distance` | Flight leg length (m) |
| `segment_duration` | Seconds per planner segment |
| `waypoint_dwell_time` | Hold at each flight waypoint |
| `hover_before_flight_time` | Hover stability before FLIGHT |

---

## Course config: `navigation/course_config.yaml`

Used by `simulate_autonomous_course.py`:

| Section | Contents |
|---------|----------|
| `course.spawn`, `checkpoints`, `goal` | Patrol path |
| `course.walls` | Wall positions per `leg_index` |
| `course.patrol_speed` | 2.5 m/s rolling cruise |
| `course.terrain_fly_*` | When to fly over unrollable terrain |
| `terrain` | Heightfield size, hills, friction |
| `mobility` | Overrides rolling gains for terrain |
| `perception` | Rangefinder cone, max range |
| `simulation.duration` | Max sim time (320 s) |

---

## Other YAML files

| File | Used by |
|------|---------|
| [terrain_config.yaml](../../terrain_config.yaml) | `simulate_terrain_mobility.py` |
| [lqr_config.yaml](../../lqr_config.yaml) | LQR comparison studies |
| [mpc_config.yaml](../../mpc_config.yaml) | MPC comparison studies |
| [validation/validation.yaml](../../validation/validation.yaml) | Monte Carlo campaigns |

---

## `ControllerConfig` dataclass

All runtime parameters live in [utils/types.py](../../utils/types.py) `ControllerConfig`. Controllers receive this object—not raw YAML—so tuning is type-safe and centralized.

Mission scripts often **mutate** config at runtime:

```python
config.roll_target = checkpoint_xy.copy()
config.landing_height = terrain_z + cage_radius
```

---

## Checkpoint

```powershell
python -c "from carolline_control.config_loader import load_config; c=load_config('carolline_control/config.yaml'); print(c.hover_height, c.motor_max)"
```

Expected: `1.0 13.0` (approx).

**Next:** [Chapter 03 — MuJoCo and physics](03-mujoco-and-physics.md)
