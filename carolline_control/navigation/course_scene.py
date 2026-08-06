"""Compile autonomous course scene: heightfield terrain + walls + rangefinder sensors."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from carolline_control.navigation.course_layout import CourseLayout, WallSpec
from carolline_control.terrain.heightmap import TerrainLayout, sample_terrain_height, write_heightmap_png
from carolline_control.terrain.scene import TERRAIN_ASSET_DIR, terrain_layout_from_config

RANGE_SENSOR_PREFIX = "course_rf_"


def _yaw_quat(angle_rad: float) -> list[float]:
    half = angle_rad * 0.5
    return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]


def _add_wall_box(
    world,
    wall: WallSpec,
    index: int,
    friction: list[float],
    *,
    ground_z: float,
) -> None:
    yaw = wall.yaw_rad
    quat = _yaw_quat(yaw)
    half_thickness = 0.5 * wall.length
    half_span = 0.5 * wall.width
    half_height = 0.5 * wall.height
    center_z = float(ground_z) + half_height
    world.add_geom(
        name=f"course_wall_{index}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[wall.x, wall.y, center_z],
        quat=quat,
        size=[half_thickness, half_span, half_height],
        rgba=[0.62, 0.38, 0.28, 0.92],
        friction=friction,
        contype=1,
        conaffinity=1,
        condim=3,
    )


def _site_quat_for_ray(direction: np.ndarray) -> list[float]:
    """Orient site so MuJoCo rangefinder (-local Z) points along direction."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    z_world = np.array([0.0, 0.0, 1.0])
    z_axis = -direction
    if abs(float(np.dot(z_axis, z_world))) > 0.99:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = np.cross(z_world, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack((x_axis, y_axis, z_axis))
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return quat.tolist()


def _attach_rangefinders(
    spec: mujoco.MjSpec,
    *,
    cage_radius: float,
    dense: bool = False,
    dense_ray_count: int = 24,
) -> list[str]:
    """Add forward fan + upward rangefinder sites on the protective cage."""
    cage = spec.body("protective_cage")
    if cage is None:
        raise RuntimeError("protective_cage body not found in robot model")

    sensor_names: list[str] = []
    mount_r = 0.36
    if dense:
        fan_angles_deg = np.linspace(-75.0, 75.0, int(dense_ray_count))
    else:
        fan_angles_deg = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
    pitch_up = math.radians(12.0)
    for idx, angle_deg in enumerate(fan_angles_deg):
        angle = math.radians(angle_deg)
        direction = np.array(
            [
                math.cos(pitch_up) * math.cos(angle),
                math.cos(pitch_up) * math.sin(angle),
                math.sin(pitch_up),
            ],
            dtype=float,
        )
        pos = direction * mount_r
        site_name = f"{RANGE_SENSOR_PREFIX}fwd_{idx}"
        cage.add_site(
            name=site_name,
            pos=pos.tolist(),
            quat=_site_quat_for_ray(direction),
            size=[0.015, 0.015, 0.015],
            rgba=[0.2, 0.85, 0.95, 0.8],
        )
        sensor_name = f"{RANGE_SENSOR_PREFIX}range_{idx}"
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
    up_site = f"{RANGE_SENSOR_PREFIX}up"
    cage.add_site(
        name=up_site,
        pos=[mount_r * 0.5, 0.0, mount_r * 0.35],
        quat=_site_quat_for_ray(up_dir),
        size=[0.015, 0.015, 0.015],
        rgba=[0.95, 0.75, 0.2, 0.8],
    )
    up_sensor = f"{RANGE_SENSOR_PREFIX}range_up"
    up = spec.add_sensor(
        name=up_sensor,
        type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=up_site,
    )
    up.intprm[0] = 10
    sensor_names.append(up_sensor)
    return sensor_names


def compile_course_scene(
    base_scene_path: str | Path,
    layout: CourseLayout,
    raw: dict,
    *,
    cache_heightmap: bool = True,
) -> tuple[mujoco.MjModel, TerrainLayout, np.ndarray, list[str]]:
    """Build heightfield course with internal walls, markers, and rangefinders."""
    terrain_raw = raw.get("terrain", {})
    terrain_cfg = {"terrain": terrain_raw, "simulation": raw.get("simulation", {})}
    t_layout = terrain_layout_from_config(terrain_cfg)
    png_name = f"heightmap_seed{t_layout.seed}_{t_layout.nrow}x{t_layout.ncol}.png"
    png_path = TERRAIN_ASSET_DIR / png_name
    if cache_heightmap and png_path.exists():
        from PIL import Image

        gray = np.asarray(Image.open(png_path).convert("L"), dtype=float) / 255.0
        heights = gray * t_layout.max_height
    else:
        heights, png_path = write_heightmap_png(png_path, t_layout)

    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody

    for geom in spec.geoms:
        if geom.name == "floor":
            geom.contype = 0
            geom.conaffinity = 0

    spec.add_hfield(
        name="course_terrain_hf",
        file=str(png_path),
        size=[t_layout.half_x, t_layout.half_y, t_layout.max_height, t_layout.base_thickness],
    )
    friction = terrain_raw.get("friction", [0.88, 0.010, 0.005])
    world.add_geom(
        name="course_terrain",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="course_terrain_hf",
        pos=[0.0, 0.0, 0.0],
        friction=friction,
        rgba=[0.38, 0.48, 0.36, 1.0],
        contype=1,
        conaffinity=1,
        condim=6,
    )

    world.add_geom(
        name="course_void_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, -3.0],
        size=[t_layout.half_x + 2.0, t_layout.half_y + 2.0, 0.05],
        rgba=[0.15, 0.18, 0.22, 0.4],
        contype=1,
        conaffinity=1,
        condim=3,
    )

    wall_h = 0.55
    wall_t = 0.25
    span_x = t_layout.half_x + 0.6
    span_y = t_layout.half_y + 0.6
    for name, pos, half in [
        ("course_wall_xmin", [-t_layout.half_x - wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("course_wall_xmax", [t_layout.half_x + wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("course_wall_ymin", [0.0, -t_layout.half_y - wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
        ("course_wall_ymax", [0.0, t_layout.half_y + wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
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
        )

    wall_friction = [0.95, 0.012, 0.006]
    for idx, wall in enumerate(layout.walls):
        ground_z = sample_terrain_height(wall.x, wall.y, heights, t_layout)
        _add_wall_box(world, wall, idx, wall_friction, ground_z=ground_z)

    def add_marker(name: str, xy: np.ndarray, rgba: list[float], size: float) -> None:
        z = sample_terrain_height(float(xy[0]), float(xy[1]), heights, t_layout) + layout.cage_radius + 0.05
        body = world.add_body(name=name, mocap=True, pos=[float(xy[0]), float(xy[1]), z])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    add_marker("course_spawn", layout.spawn_xy, [0.20, 0.85, 0.30, 0.90], 0.08)
    for idx, pt in enumerate(layout.checkpoints):
        add_marker(f"course_checkpoint_{idx}", pt, [0.95, 0.55, 0.15, 0.90], 0.10)
    add_marker("course_goal", layout.goal_xy, [0.95, 0.20, 0.15, 0.95], 0.11)
    add_marker("course_target_live", layout.spawn_xy, [0.15, 0.85, 0.95, 0.95], 0.09)

    range_sensor_names = _attach_rangefinders(
        spec,
        cage_radius=layout.cage_radius,
        dense=bool(raw.get("perception", {}).get("dense_rays", False)),
        dense_ray_count=int(raw.get("perception", {}).get("dense_ray_count", 24)),
    )

    model = spec.compile()
    sim = raw.get("simulation", {})
    model.opt.timestep = float(sim.get("timestep", 0.004))
    return model, t_layout, heights, range_sensor_names


def set_live_target_marker(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray, z: float) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "course_target_live")
    if bid < 0:
        return
    mid = int(model.body_mocapid[bid])
    if mid >= 0:
        data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), float(z)]
