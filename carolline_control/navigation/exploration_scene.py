"""Compile exploration scenes with random obstacles and SLAM rangefinders."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from carolline_control.navigation.course_scene import (
    RANGE_SENSOR_PREFIX,
    _site_quat_for_ray,
)

EXPLORATION_SENSOR_PREFIX = "slam_rf_"


def _attach_rangefinders(
    spec: mujoco.MjSpec,
    *,
    mount_r: float = 0.36,
    dense: bool = True,
    dense_ray_count: int = 24,
) -> list[str]:
    cage = spec.body("protective_cage")
    if cage is None:
        raise RuntimeError("protective_cage body not found in robot model")

    sensor_names: list[str] = []
    pitch_up = math.radians(12.0)

    if dense:
        fan_angles_deg = np.linspace(-75.0, 75.0, int(dense_ray_count))
    else:
        fan_angles_deg = np.array([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])

    for idx, angle_deg in enumerate(fan_angles_deg):
        angle = math.radians(float(angle_deg))
        direction = np.array(
            [
                math.cos(pitch_up) * math.cos(angle),
                math.cos(pitch_up) * math.sin(angle),
                math.sin(pitch_up),
            ],
            dtype=float,
        )
        pos = direction * mount_r
        site_name = f"{EXPLORATION_SENSOR_PREFIX}fwd_{idx}"
        cage.add_site(
            name=site_name,
            pos=pos.tolist(),
            quat=_site_quat_for_ray(direction),
            size=[0.012, 0.012, 0.012],
            rgba=[0.2, 0.85, 0.95, 0.75],
        )
        sensor_name = f"{EXPLORATION_SENSOR_PREFIX}range_{idx}"
        sensor = spec.add_sensor(
            name=sensor_name,
            type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site_name,
        )
        sensor.intprm[0] = 10
        sensor_names.append(sensor_name)

    up_dir = np.array([0.25, 0.0, 0.97], dtype=float)
    up_dir /= float(np.linalg.norm(up_dir))
    up_site = f"{EXPLORATION_SENSOR_PREFIX}up"
    cage.add_site(
        name=up_site,
        pos=[mount_r * 0.5, 0.0, mount_r * 0.35],
        quat=_site_quat_for_ray(up_dir),
        size=[0.012, 0.012, 0.012],
        rgba=[0.95, 0.75, 0.2, 0.75],
    )
    up_sensor = f"{EXPLORATION_SENSOR_PREFIX}range_up"
    up = spec.add_sensor(
        name=up_sensor,
        type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=up_site,
    )
    up.intprm[0] = 10
    sensor_names.append(up_sensor)

    down_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    down_site = f"{EXPLORATION_SENSOR_PREFIX}down"
    cage.add_site(
        name=down_site,
        pos=[0.0, 0.0, -mount_r * 0.35],
        quat=_site_quat_for_ray(down_dir),
        size=[0.012, 0.012, 0.012],
        rgba=[0.95, 0.35, 0.2, 0.75],
    )
    down_sensor = f"{EXPLORATION_SENSOR_PREFIX}range_down"
    down = spec.add_sensor(
        name=down_sensor,
        type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=down_site,
    )
    down.intprm[0] = 10
    sensor_names.append(down_sensor)
    return sensor_names


def _random_obstacles(
    rng: np.random.Generator,
    *,
    count: int,
    arena_half_x: float,
    arena_half_y: float,
    min_size: float,
    max_size: float,
    spawn_xy: np.ndarray,
    goal_xy: np.ndarray,
    clearance: float = 2.5,
) -> list[dict]:
    obstacles: list[dict] = []
    attempts = 0
    while len(obstacles) < count and attempts < count * 40:
        attempts += 1
        width = float(rng.uniform(min_size, max_size))
        depth = float(rng.uniform(min_size * 0.6, max_size * 0.8))
        height = float(rng.uniform(0.45, 0.95))
        x = float(rng.uniform(-arena_half_x + 1.5, arena_half_x - 1.5))
        y = float(rng.uniform(-arena_half_y + 1.0, arena_half_y - 1.0))
        center = np.array([x, y], dtype=float)
        if float(np.linalg.norm(center - spawn_xy)) < clearance:
            continue
        if float(np.linalg.norm(center - goal_xy)) < clearance:
            continue
        if any(float(np.linalg.norm(center - np.array([o["x"], o["y"]])) ) < 2.0 for o in obstacles):
            continue
        yaw_deg = float(rng.uniform(-25.0, 25.0))
        obstacles.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "depth": depth,
                "height": height,
                "yaw_deg": yaw_deg,
            }
        )
    return obstacles


def compile_exploration_scene(
    base_scene_path: str | Path,
    raw: dict,
) -> tuple[mujoco.MjModel, list[dict], np.ndarray, np.ndarray]:
    """Build flat arena with random box obstacles and dense rangefinders."""
    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody

    exp = raw.get("exploration", {})
    sim = raw.get("simulation", {})
    perception = raw.get("perception", {})
    seed = int(exp.get("seed", 11))
    rng = np.random.default_rng(seed)

    spawn_xy = np.asarray(exp.get("spawn", [-12.0, 0.0]), dtype=float)
    goal_xy = np.asarray(exp.get("goal", [12.0, 0.0]), dtype=float)
    arena_half_x = float(exp.get("arena_half_x", 14.0))
    arena_half_y = float(exp.get("arena_half_y", 10.0))
    friction = raw.get("terrain", {}).get("friction", [0.88, 0.010, 0.005])

    for geom in spec.geoms:
        if geom.name == "floor":
            geom.contype = 1
            geom.conaffinity = 1
            geom.friction = friction
            geom.condim = 3

    wall_h = 0.55
    wall_t = 0.25
    span_x = arena_half_x + 0.8
    span_y = arena_half_y + 0.8
    for name, pos, half in [
        ("explore_wall_xmin", [-arena_half_x - wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("explore_wall_xmax", [arena_half_x + wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("explore_wall_ymin", [0.0, -arena_half_y - wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
        ("explore_wall_ymax", [0.0, arena_half_y + wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
    ]:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=half,
            rgba=[0.30, 0.34, 0.38, 0.85],
            contype=1,
            conaffinity=1,
            condim=3,
            friction=friction,
        )

    obstacles = _random_obstacles(
        rng,
        count=int(exp.get("num_obstacles", 6)),
        arena_half_x=arena_half_x,
        arena_half_y=arena_half_y,
        min_size=float(exp.get("obstacle_min_size", 0.8)),
        max_size=float(exp.get("obstacle_max_size", 2.2)),
        spawn_xy=spawn_xy,
        goal_xy=goal_xy,
    )

    for idx, obs in enumerate(obstacles):
        yaw = math.radians(float(obs["yaw_deg"]))
        quat = [float(math.cos(yaw * 0.5)), 0.0, 0.0, float(math.sin(yaw * 0.5))]
        half_x = 0.5 * float(obs["width"])
        half_y = 0.5 * float(obs["depth"])
        half_z = 0.5 * float(obs["height"])
        world.add_geom(
            name=f"explore_obstacle_{idx}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(obs["x"]), float(obs["y"]), half_z],
            quat=quat,
            size=[half_x, half_y, half_z],
            rgba=[0.62, 0.38, 0.28, 0.92],
            contype=1,
            conaffinity=1,
            condim=3,
            friction=[0.95, 0.012, 0.006],
        )

    def add_marker(name: str, xy: np.ndarray, rgba: list[float], size: float, z: float = 0.45) -> None:
        body = world.add_body(name=name, mocap=True, pos=[float(xy[0]), float(xy[1]), z])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    cage_radius = 0.40
    add_marker("explore_spawn", spawn_xy, [0.20, 0.85, 0.30, 0.90], 0.08, z=cage_radius + 0.05)
    add_marker("explore_goal", goal_xy, [0.95, 0.20, 0.15, 0.95], 0.11, z=cage_radius + 0.05)
    add_marker("explore_target_live", spawn_xy, [0.15, 0.85, 0.95, 0.95], 0.09, z=cage_radius + 0.12)

    _attach_rangefinders(
        spec,
        dense=bool(perception.get("dense_rays", True)),
        dense_ray_count=int(perception.get("dense_ray_count", 24)),
    )

    model = spec.compile()
    model.opt.timestep = float(sim.get("timestep", 0.004))
    return model, obstacles, spawn_xy, goal_xy
