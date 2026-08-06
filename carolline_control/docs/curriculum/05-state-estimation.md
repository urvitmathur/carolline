# Chapter 05 — State Estimation

**Previous:** [04-math-and-types.md](04-math-and-types.md) | **Next:** [06-control-stack-overview.md](06-control-stack-overview.md)

---

## Role in the stack

Every control step starts with:

```python
state = estimator.estimate(data)  # MjData → RobotState
```

Controllers never read `qpos` directly—they use `RobotState`.

---

## `controllers/state_estimator.py`

**Class:** `StateEstimator`

### Oracle mode (default)

`sensor_only=False`:

- **Position & orientation** from MuJoCo `qpos` (perfect ground truth).
- **Velocity** from IMU integration or body velocity sensors.
- Used for algorithm development and repeatable missions.

Tracks `last_oracle_position_error` when comparing to integrated state.

### Sensor-only mode

`sensor_only=True` (`--sensor-only` CLI flag):

- Pose initialized from spawn; **dead-reckoned** from gyro + accel.
- Realistic for testing estimator drift and recovery.

### SLAM odometry mode

`slam_odom=True` (`--slam-nav` on exploration scripts):

- Orientation from IMU (`body_quat`); planar position from the navigation SLAM stack.
- `apply_slam_pose()` injects mapper/localizer output each control step.
- Oracle `qpos` is still available for evaluation error only.

### Contact estimation

- Scans MuJoCo contacts for cage geoms vs terrain.
- Fills `contact_point_world`, `contact_normal_world`, `contact_valid`.
- Used by [rolling_controller.py](../../controllers/rolling_controller.py) and ground allocation.

---

## `sim_estimator.py`

Wiring layer for scripts:

| Function | Role |
|----------|------|
| `build_estimator(model, config, sensor_only=...)` | Factory |
| `seed_estimator_from_sim(estimator, data)` | Align after spawn |
| `add_sensor_only_argument(parser)` | Adds `--sensor-only` to argparse |
| `add_slam_nav_argument(parser)` | Adds `--slam-nav` to argparse |
| `print_estimator_mode(...)` | Console banner |

---

## Building `RobotState` from `MjData`

Conceptual steps each `estimate()` call:

1. Read free joint qpos → position, quaternion → rotation matrix.
2. Read gyro/accel sensors → omega, accel.
3. Integrate or read velocity.
4. Run contact detection → on_ground, normal, contact point.
5. Set `time = data.time`.

---

## Checkpoint

```powershell
python -m carolline_control.main --no-viewer --sensor-only
```

Compare console oracle error line at end vs default run.

**Next:** [Chapter 06 — Control stack overview](06-control-stack-overview.md)
