# Chapter 03 — MuJoCo and Physics

**Previous:** [02-setup-and-config.md](02-setup-and-config.md) | **Next:** [04-math-and-types.md](04-math-and-types.md)

---

## MuJoCo model assets

Located in `mujoco_menagerie-main/skydio_x2/` (parent repo):

| File | Use |
|------|-----|
| `x2.xml` | Skydio X2 drone body, rotors, IMU site |
| `scene.xml` | Drone + ground plane |
| `scene_cage.xml` | Drone inside CAROLLINE spherical cage (default in `config.yaml`) |

The **free joint** (7 qpos: xyz + quaternion) represents the whole cage+drone assembly rolling and flying.

---

## Timestep

Default in `config.yaml`: `timestep: 0.002` (500 Hz) for `main.py`.

Course sim uses `course_config.yaml` → `simulation.timestep: 0.004` (250 Hz).

Control runs **once per physics step**: `controller.compute(state, dt)` then `mujoco.mj_step()`.

---

## Contacts and ground control

Rolling depends on MuJoCo **contact forces** between cage geoms and terrain:

- [ground_dynamics.py](../../controllers/ground_dynamics.py) finds support contacts, estimates contact point and surface normal.
- [state_estimator.py](../../controllers/state_estimator.py) sets `RobotState.on_ground`, `contact_valid`, `contact_normal_world`.

Without valid contact, rolling torque allocation is disabled or reduced.

---

## Scene compilation

### Default mission

[visualization/markers.py](../../visualization/markers.py) `compile_model_with_markers()` injects colored mocap spheres:

- Green — spawn
- Red — roll target
- Blue — flight waypoints

### Terrain course

[navigation/course_scene.py](../../navigation/course_scene.py) `compile_course_scene()`:

1. Loads base model from `course_config` `model_path`
2. Adds **heightfield terrain** (see [Chapter 13](13-terrain-and-scenes.md))
3. Places **wall boxes** on terrain surface via `sample_terrain_height()`
4. Attaches **rangefinder sites** on the cage (`RANGE_SENSOR_PREFIX`)
5. Returns `(model, layout, heights, sensor_names)`

---

## Live target marker

During autonomous course, `set_live_target_marker()` moves an orange mocap body to the director's current target (checkpoint or landing point).

---

## Viewer vs headless

| Mode | API |
|------|-----|
| Interactive | `mujoco.viewer.launch_passive(model, data)` |
| Headless | Loop `mj_step` only; optional `mujoco.Renderer` for video ([sim_recorder.py](../../navigation/sim_recorder.py)) |

Tracking camera logic is shared with teleop: [manual_teleop.py](../../scripts/manual_teleop.py) `update_tracking_camera()`.

---

## Checkpoint

Open `mujoco_menagerie-main/skydio_x2/scene_cage.xml` and find the cage collision geoms and free joint. Note how rotor `<actuator>` elements map to `data.ctrl[0:4]`.

**Next:** [Chapter 04 — Math and types](04-math-and-types.md)
