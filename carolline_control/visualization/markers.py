"""Add visual mission markers to the MuJoCo model."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from carolline_control.utils.types import ControllerConfig


def compile_model_with_markers(model_path: str | Path, config: ControllerConfig) -> mujoco.MjModel:
    """Load scene XML and add mocap markers for spawn, roll target, and waypoints."""
    spec = mujoco.MjSpec.from_file(str(model_path))

    def add_marker(
        name: str,
        pos: list[float],
        rgba: list[float],
        sphere_size: float,
        *,
        ground_dot: bool = True,
    ) -> None:
        body = spec.worldbody.add_body(name=name, mocap=True, pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[sphere_size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )
        if ground_dot:
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[sphere_size * 0.55, 0.004, 0.0],
                pos=[0.0, 0.0, -pos[2] + 0.004],
                rgba=[rgba[0], rgba[1], rgba[2], min(1.0, rgba[3] + 0.02)],
                contype=0,
                conaffinity=0,
            )

    spawn = config.spawn_xy
    roll = config.roll_target
    z_mark = 0.10
    z_ground = 0.015

    add_marker(
        "marker_spawn",
        [float(spawn[0]), float(spawn[1]), z_mark],
        [0.15, 0.85, 0.25, 0.95],
        0.08,
    )
    add_marker(
        "marker_spawn_ground",
        [float(spawn[0]), float(spawn[1]), z_ground],
        [0.15, 0.85, 0.25, 0.75],
        0.045,
        ground_dot=False,
    )
    add_marker(
        "marker_roll_target",
        [float(roll[0]), float(roll[1]), z_mark],
        [0.95, 0.2, 0.15, 0.95],
        0.11,
    )
    add_marker(
        "marker_roll_target_ground",
        [float(roll[0]), float(roll[1]), z_ground],
        [0.95, 0.2, 0.15, 0.80],
        0.055,
        ground_dot=False,
    )

    if config.waypoints:
        for idx, wp in enumerate(config.waypoints):
            add_marker(
                f"marker_wp_{idx}",
                [float(wp[0]), float(wp[1]), float(wp[2])],
                [0.25, 0.45, 0.95, 0.55],
                0.045,
                ground_dot=False,
            )

    return spec.compile()
