"""Compile Gazebo-style maze arena for rolling SLAM."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from carolline_control.navigation.exploration_scene import _attach_rangefinders
from carolline_control.navigation.maze_layout import MazeLayout, MazeSegment


WOOD_RGBA = [0.68, 0.48, 0.30, 1.0]


def _add_wall_box(world, name: str, seg: MazeSegment, half_z: float, friction: list[float]) -> None:
    world.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[seg.x, seg.y, half_z],
        size=[seg.half_x, seg.half_y, half_z],
        rgba=WOOD_RGBA,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=friction,
    )


def _perimeter_walls(
    world,
    layout: MazeLayout,
    half_z: float,
    friction: list[float],
) -> None:
    """Square perimeter with a single entrance gap (default: north / +Y side)."""
    a = layout.arena_half
    t = layout.wall_thickness
    gap = layout.entrance_width
    off = layout.entrance_offset

    # South (−Y), West (−X), East (+X) — full spans
    for name, pos, half in [
        ("maze_wall_s", [0.0, -a - t, half_z], [a + t, t, half_z]),
        ("maze_wall_w", [-a - t, 0.0, half_z], [t, a + t, half_z]),
        ("maze_wall_e", [a + t, 0.0, half_z], [t, a + t, half_z]),
    ]:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=half,
            rgba=WOOD_RGBA,
            contype=1,
            conaffinity=1,
            condim=3,
            friction=friction,
        )

    # North (+Y) — two segments with gap (entrance)
    gap_center = off
    half_left = (a + gap_center - gap * 0.5) * 0.5
    cx_left = -a + half_left
    half_right = (a - gap_center - gap * 0.5) * 0.5
    cx_right = a - half_right
    for name, cx, hw in [
        ("maze_wall_nw", cx_left, half_left),
        ("maze_wall_ne", cx_right, half_right),
    ]:
        if hw > 0.05:
            world.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[cx, a + t, half_z],
                size=[hw, t, half_z],
                rgba=WOOD_RGBA,
                contype=1,
                conaffinity=1,
                condim=3,
                friction=friction,
            )


def compile_maze_scene(
    base_scene_path: str | Path,
    layout: MazeLayout,
    raw: dict,
) -> tuple[mujoco.MjModel, MazeLayout]:
    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody
    sim = raw.get("simulation", {})
    perception = raw.get("perception", {})
    friction = raw.get("terrain", {}).get("friction", [0.88, 0.010, 0.005])

    for geom in spec.geoms:
        if geom.name == "floor":
            # Keep groundplane material/texture; only tune contact + a mild tint.
            geom.rgba = [1.0, 1.0, 1.0, 1.0]
            geom.contype = 1
            geom.conaffinity = 1
            geom.friction = friction
            geom.condim = 3

    half_z = 0.5 * layout.wall_height
    _perimeter_walls(world, layout, half_z, friction)
    for idx, seg in enumerate(layout.segments):
        _add_wall_box(world, f"maze_seg_{idx}", seg, half_z, friction)

    cage_r = 0.40

    def add_marker(name: str, xy: np.ndarray, rgba: list[float], size: float) -> None:
        body = world.add_body(name=name, mocap=True, pos=[float(xy[0]), float(xy[1]), cage_r + 0.06])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    add_marker("maze_spawn", layout.spawn_xy, [0.20, 0.85, 0.30, 0.95], 0.08)
    add_marker("maze_goal", layout.goal_xy, [0.20, 0.90, 0.25, 0.95], 0.10)
    add_marker("maze_target_live", layout.spawn_xy, [0.15, 0.85, 0.95, 0.95], 0.08)

    _attach_rangefinders(
        spec,
        dense=bool(perception.get("dense_rays", True)),
        dense_ray_count=int(perception.get("dense_ray_count", 24)),
    )

    model = spec.compile()
    model.opt.timestep = float(sim.get("timestep", 0.004))
    return model, layout
