"""Build MuJoCo scene with smooth overlapping ramp track segments."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from carolline_control.rolling_ramp.path import RampGeometry


def _yaw_quat(angle_rad: float) -> list[float]:
    half = angle_rad * 0.5
    return [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]


def _add_segment_box(world, name, center, half_size, quat, rgba, friction) -> None:
    world.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=center,
        quat=quat,
        size=half_size,
        rgba=rgba,
        friction=friction,
        contype=1,
        conaffinity=1,
        condim=6,
    )


def _ramp_segments(geometry: RampGeometry, world, friction, rgba, *, direction: str) -> None:
    """Split a ramp into sub-boxes for smoother contact transitions."""
    a = geometry.angle_rad
    yc = geometry.y_center
    w = geometry.width * 0.5
    t = 0.05
    n = 5
    seg = geometry.ramp_length / n
    if direction == "up":
        for i in range(n):
            ds = (i + 0.5) * seg
            x = geometry.flat_start + ds * np.cos(a)
            z = ds * np.sin(a) - t * np.cos(a)
            _add_segment_box(
                world,
                f"track_ramp_up_{i}",
                [float(x), yc, float(z)],
                [seg * 0.55, w, t],
                _yaw_quat(-a),
                rgba,
                friction,
            )
    else:
        # The decline starts after the full top deck, not at its leading edge.
        x_top0 = geometry.flat_start + geometry.ramp_run + geometry.flat_top
        for i in range(n):
            ds = (i + 0.5) * seg
            x = x_top0 + ds * np.cos(a)
            z = geometry.rise - ds * np.sin(a) - t * np.cos(a)
            _add_segment_box(
                world,
                f"track_ramp_down_{i}",
                [float(x), yc, float(z)],
                [seg * 0.55, w, t],
                _yaw_quat(a),
                rgba,
                friction,
            )


def compile_ramp_scene(
    base_scene_path: str | Path,
    geometry: RampGeometry,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody

    for geom in spec.geoms:
        if geom.name == "floor":
            geom.contype = 0
            geom.conaffinity = 0

    yc = geometry.y_center
    width = geometry.width
    thickness = 0.05
    overlap = 0.35
    friction = [0.95, 0.012, 0.006]
    rgba_flat = [0.35, 0.45, 0.55, 1]
    rgba_up = [0.45, 0.55, 0.35, 1]
    rgba_top = [0.55, 0.50, 0.35, 1]
    rgba_down = [0.45, 0.40, 0.55, 1]

    l0 = geometry.flat_start + overlap
    _add_segment_box(
        world,
        "track_flat_start",
        [l0 * 0.5 - overlap * 0.5, yc, -thickness * 0.5],
        [l0 * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_flat,
        friction,
    )

    _ramp_segments(geometry, world, friction, rgba_up, direction="up")
    _ramp_segments(geometry, world, friction, rgba_down, direction="down")

    lt = geometry.flat_top + overlap
    x_top = geometry.flat_start + geometry.ramp_run + lt * 0.5
    _add_segment_box(
        world,
        "track_flat_top",
        [float(x_top), yc, geometry.rise - thickness * 0.5],
        [lt * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_top,
        friction,
    )

    le = geometry.flat_end + overlap
    x_end = geometry.flat_start + 2 * geometry.ramp_run + geometry.flat_top + le * 0.5
    _add_segment_box(
        world,
        "track_flat_end",
        [float(x_end), yc, -thickness * 0.5],
        [le * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_flat,
        friction,
    )

    total_x = geometry.flat_start + 2 * geometry.ramp_run + geometry.flat_top + geometry.flat_end
    for side, y_sign in (("left", -1), ("right", 1)):
        world.add_geom(
            name=f"wall_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[total_x * 0.5, yc + y_sign * (width * 0.5 + 0.05), 0.35],
            size=[total_x * 0.55, 0.04, 0.35],
            rgba=[0.25, 0.25, 0.25, 0.35],
            friction=friction,
            contype=1,
            conaffinity=1,
            condim=3,
        )

    world.add_geom(
        name="safety_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[total_x * 0.5, yc, -1.0],
        size=[total_x * 0.8, width, 0.05],
        rgba=[0.1, 0.1, 0.1, 0.15],
        friction=[0.5, 0.01, 0.005],
        contype=1,
        conaffinity=1,
        condim=3,
    )

    for name, s, rgba in (
        ("marker_start", 0.0, [0.1, 0.85, 0.2, 0.9]),
        ("marker_ramp_top", geometry.flat_start + geometry.ramp_length + 0.5 * geometry.flat_top, [0.9, 0.5, 0.1, 0.9]),
        ("marker_end", geometry.total_length, [0.9, 0.2, 0.15, 0.9]),
    ):
        p = geometry.position_at_s(s)
        body = world.add_body(name=name, mocap=True, pos=p.tolist())
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.07, 0, 0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    model = spec.compile()
    model.opt.timestep = 0.004
    return model
