# Chapter 3 — System Modelling and Simulation

## 3.1 Introduction

This chapter describes the MuJoCo plant model, coordinate frames, state estimation modes, simulation scenes, software architecture, and actuator characterisation used throughout the project.

## 3.2 MuJoCo plant model

The vehicle model combines the Skydio X2 quadrotor from MuJoCo Menagerie with a **protective cage** body (`scene_cage.xml`). Key parameters loaded at runtime include:

| Parameter | Value |
|-----------|-------|
| Total mass | 2.125 kg |
| Cage radius | 0.40 m |
| Motor thrust range | −13 N to +13 N |
| Simulation timestep | 0.002 s (main mission) |

Four thrust actuators attach to rotor sites with gear vectors `[0, 0, 1, 0, 0, μ_z]` where μ_z = ±0.0154 sets yaw reaction torque per newton of thrust. Collision uses a **single sphere** for the cage to stabilise rolling contact; ribbed visual geometry is optional.

Ground interaction uses MuJoCo’s soft contact model with configurable friction coefficients (default `[0.88, 0.010, 0.005]`). The floor is a plane geom; walls and maze segments are box geoms with wood-coloured visuals.

## 3.3 Coordinate frames

Three frames dominate control design:

1. **World frame** — Inertial axes; position \(\mathbf{p} \in \mathbb{R}^3\), velocity \(\mathbf{v}\), yaw about \(z\).
2. **Body frame** — Fixed to the drone centre; angular rate \(\boldsymbol{\omega}\) measured by gyro.
3. **Contact frame** — Used for rolling control: contact point on the cage surface, contact normal from MuJoCo `cfrc_ext` and touch sensors (`ground_dynamics.py`).

Rolling velocity commands are planar in world \(xy\) but tracking torques are computed relative to contact normal to handle ramps and platform edges.

## 3.4 State estimation

Three estimation modes were implemented (`StateEstimator`, `sim_estimator.py`):

| Mode | Pose source | Use case |
|------|-------------|----------|
| Oracle | MuJoCo `qpos` | Controller development, SLAM map debugging |
| Sensor-only | IMU quaternion + integrated velocity | Hardware-realistic testing |
| SLAM odom | Planar SLAM correction + IMU attitude | Navigation without oracle |

Sensor-only mode dead-reckons position from a known spawn point; drift accumulates without global corrections. SLAM mode updates **\(x, y\)** only and retains IMU attitude—preventing yaw-only replacements that destabilised rolling.

## 3.5 Simulation scenes

Multiple scene compilers extend the base cage model:

- **Default mission** — Flat floor, spawn at origin, roll target (5, 5) m, cardinal waypoints.
- **Terrain heightfield** — Hills and V-ditch for mobility and course missions.
- **Autonomous course** — Vertical walls on checkpoint legs, checkpoint markers.
- **Maze arena** — 9 m × 9 m perimeter, nine internal orthogonal walls, 1.5 m north entrance gap (`maze_scene.py`).

Rangefinders (7–25 sensors) attach to cage sites: a forward fan (−75° to +75°) and one upward ray for fly-over clearance checks.

## 3.6 Software architecture

Figure 3.1 shows the layered architecture. MuJoCo advances physics; the state estimator produces `RobotState`; mission/navigation modules set mode requests and velocity hooks; `CarollineController` computes motor thrusts each timestep. See Figure 3.1.


## 3.7 Actuator characterisation

Figure 3.2 reproduces the simulation ESC model: piecewise-linear thrust–PWM mapping with mid-band dead zone (1450–1550 µs → 0 N), and linear thrust–torque lines from `yaw_drag_coeff`. See Figure 3.2.

## 3.8 Chapter discussion

The model prioritises **controller fidelity and repeatability** over visual or aerodynamic detail. Single-sphere cage collision trades geometric accuracy for stable rolling—an explicit limitation for ribbed-cage realism. Oracle estimation enabled distinction-level validation of control and planning separately from SLAM drift; sensor-only and SLAM modes expose integration risks before hardware deployment.
