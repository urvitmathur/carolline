# Chapter 04 — Math and Types

**Previous:** [03-mujoco-and-physics.md](03-mujoco-and-physics.md) | **Next:** [05-state-estimation.md](05-state-estimation.md)

---

## File: `utils/so3.py`

Implements **SO(3) geometric control** math (Lee et al. 2010). Used everywhere attitude matters.

### Key functions

| Function | Purpose |
|----------|---------|
| `hat(v)` / `vee(S)` | Skew-symmetric ↔ vector |
| `quat_to_rot(q)` | MuJoCo [w,x,y,z] → rotation matrix R |
| `attitude_error(R, Rd)` | e_R via matrix logarithm (stable near 180°) |
| `desired_rotation_from_thrust_direction(b3, yaw)` | Build desired attitude from thrust axis |
| `body_z_world(R)` | Third column of R — which way "up" points on body |
| `rolling_omega_world(v, r_contact, yaw_rate)` | No-slip rolling kinematics |

**Why logarithm for e_R?** Pre-takeoff recovery can start near inverted; Euler angles would singular.

---

## File: `utils/types.py` — line-by-line

### Type aliases (lines 11–13)

```python
Vector3 = np.ndarray   # shape (3,)
Matrix3 = np.ndarray   # shape (3,3)
Quaternion = np.ndarray  # [w, x, y, z]
```

### `ControlMode` enum (lines 16–26)

Each value is one **controller regime**:

| Mode | Typical use |
|------|-------------|
| `PRETAKEOFF` | Recover from arbitrary ground pose |
| `UPRIGHT` | Brief settle before takeoff |
| `TAKEOFF` | Climb to hover height |
| `HOVER` | Hold position before flight |
| `FLIGHT` | Track planner waypoints |
| `LANDING` | Descend to ground |
| `ROLLING` | Ground locomotion |
| `IDLE` | Motors off, mission done |

### `RobotState` (lines 29–49)

Everything controllers read about the robot:

| Field | Meaning |
|-------|---------|
| `position`, `velocity` | World-frame COM |
| `quaternion`, `rotation` | Pose (MuJoCo convention) |
| `omega_body`, `omega_world` | Angular rates |
| `accel_body` | IMU linear acceleration |
| `on_ground` | Contact heuristic |
| `ground_contact_z` | Terrain height under robot |
| `time` | Simulation time |
| `contact_point_world` | Rolling contact location |
| `contact_normal_world` | Surface normal at contact |
| `contact_force`, `contact_confidence`, `contact_valid` | Contact quality |

### `ControlCommand` (lines 52–60)

High-level wrench **before** motor mixing:

- `thrust` — collective scalar along body-z
- `moment_body` — roll/pitch/yaw torque request
- `desired_omega_body` — rate feedforward
- `desired_rotation` — Rd target attitude
- `mode` — echo of active mode

### `MotorCommand` (lines 63–67)

Four thrust values → `data.ctrl[:4]` after ESC mapping and slew limiting.

### `TrajectoryTarget` (lines 70–78)

Planner output for flight modes:

- `position`, `velocity`, `acceleration` — desired trajectory
- `yaw`, `yaw_rate` — heading

### `ControlDiagnostics` (lines 81–101)

Logged every step: errors, euler angles, motor spread, saturation flags, contact info.

### `ControllerConfig` (lines 104–176)

Large dataclass — all YAML parameters. See [Chapter 02](02-setup-and-config.md).

---

## Checkpoint

In Python REPL:

```python
from carolline_control.utils.types import ControlMode
list(ControlMode)  # all 8 modes
```

**Next:** [Chapter 05 — State estimation](05-state-estimation.md)
