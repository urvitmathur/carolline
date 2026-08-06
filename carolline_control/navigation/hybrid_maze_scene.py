"""Compile the hybrid roll/fly maze MuJoCo scene."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from carolline_control.navigation.exploration_scene import _attach_rangefinders
from carolline_control.navigation.hybrid_maze_layout import HybridMazeLayout, WallBox

FLOOR_RGBA = [0.78, 0.76, 0.70, 1.0]
WALL_RGBA = [0.62, 0.60, 0.55, 1.0]
LOW_WALL_RGBA = [0.92, 0.62, 0.18, 1.0]
PLATFORM_RGBA = [0.55, 0.54, 0.50, 1.0]
VOID_RGBA = [0.15, 0.16, 0.18, 1.0]


def _add_box(
    world,
    *,
    name: str,
    pos: list[float],
    size: list[float],
    rgba: list[float],
    friction: list[float],
    contype: int = 1,
    conaffinity: int = 1,
) -> None:
    world.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=rgba,
        contype=contype,
        conaffinity=conaffinity,
        condim=3,
        friction=friction,
    )


def _add_wall(world, name: str, wall: WallBox, friction: list[float], rgba: list[float]) -> None:
    half_z = 0.5 * wall.height
    _add_box(
        world,
        name=name,
        pos=[wall.x, wall.y, half_z],
        size=[wall.half_x, wall.half_y, half_z],
        rgba=rgba,
        friction=friction,
    )


def _floor_tiles(layout: HybridMazeLayout) -> list[tuple[str, float, float, float, float]]:
    """Axis-aligned floor rectangles that leave a hole for the gap trench."""
    ax, ay = layout.arena_half_x, layout.arena_half_y
    gx, gy = layout.gap.x, layout.gap.y
    ghx, ghy = layout.gap.half_x, layout.gap.half_y
    # Split arena into strips around the gap AABB.
    x0, x1 = -ax, ax
    y0, y1 = -ay, ay
    gl, gr = gx - ghx, gx + ghx
    gb, gt = gy - ghy, gy + ghy
    tiles: list[tuple[str, float, float, float, float]] = []
    # Bottom strip (full width)
    if gb > y0 + 0.05:
        cy = 0.5 * (y0 + gb)
        hy = 0.5 * (gb - y0)
        tiles.append(("floor_s", 0.0, cy, ax, hy))
    # Top strip (full width)
    if y1 > gt + 0.05:
        cy = 0.5 * (gt + y1)
        hy = 0.5 * (y1 - gt)
        tiles.append(("floor_n", 0.0, cy, ax, hy))
    # Middle-left beside gap
    if gl > x0 + 0.05:
        cx = 0.5 * (x0 + gl)
        hx = 0.5 * (gl - x0)
        cy = 0.5 * (gb + gt)
        hy = 0.5 * (gt - gb)
        tiles.append(("floor_w", cx, cy, hx, hy))
    # Middle-right beside gap
    if x1 > gr + 0.05:
        cx = 0.5 * (gr + x1)
        hx = 0.5 * (x1 - gr)
        cy = 0.5 * (gb + gt)
        hy = 0.5 * (gt - gb)
        tiles.append(("floor_e", cx, cy, hx, hy))
    return tiles


def compile_hybrid_maze_scene(
    base_scene_path: str | Path,
    layout: HybridMazeLayout,
    raw: dict,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody
    sim = raw.get("simulation", {})
    perception = raw.get("perception", {})
    friction = raw.get("terrain", {}).get("friction", [0.88, 0.010, 0.005])

    # Hide default infinite floor; we build tiled floors with a gap.
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.rgba = [0.2, 0.2, 0.22, 0.0]
            geom.contype = 0
            geom.conaffinity = 0

    floor_half_z = 0.04
    for name, cx, cy, hx, hy in _floor_tiles(layout):
        if hx < 0.05 or hy < 0.05:
            continue
        _add_box(
            world,
            name=name,
            pos=[cx, cy, -floor_half_z],
            size=[hx, hy, floor_half_z],
            rgba=FLOOR_RGBA,
            friction=friction,
        )

    # Catch plane deep under the gap.
    _add_box(
        world,
        name="void_catch",
        pos=[layout.gap.x, layout.gap.y, -2.5],
        size=[layout.gap.half_x + 0.4, layout.gap.half_y + 0.4, 0.05],
        rgba=VOID_RGBA,
        friction=friction,
    )

    # Perimeter walls
    ax, ay = layout.arena_half_x, layout.arena_half_y
    t = layout.wall_thickness
    hz = 0.5 * layout.wall_height
    for name, pos, size in (
        ("maze_wall_n", [0.0, ay + t, hz], [ax + t, t, hz]),
        ("maze_wall_s", [0.0, -ay - t, hz], [ax + t, t, hz]),
        ("maze_wall_e", [ax + t, 0.0, hz], [t, ay + t, hz]),
        ("maze_wall_w", [-ax - t, 0.0, hz], [t, ay + t, hz]),
    ):
        _add_box(world, name=name, pos=pos, size=size, rgba=WALL_RGBA, friction=friction)

    for idx, wall in enumerate(layout.walls):
        _add_wall(world, f"maze_seg_{idx}", wall, friction, WALL_RGBA)

    _add_wall(world, "maze_low_wall", layout.low_wall, friction, LOW_WALL_RGBA)

    plat = layout.platform
    _add_box(
        world,
        name="maze_platform",
        pos=[plat.x, plat.y, 0.5 * plat.height],
        size=[plat.half_x, plat.half_y, 0.5 * plat.height],
        rgba=PLATFORM_RGBA,
        friction=friction,
    )

    cage_r = layout.cage_radius

    def add_marker(name: str, xy: np.ndarray, z: float, rgba: list[float], size: float) -> None:
        body = world.add_body(name=name, mocap=True, pos=[float(xy[0]), float(xy[1]), z])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    add_marker("hybrid_spawn", layout.spawn_xy, cage_r + 0.06, [0.15, 0.75, 0.45, 0.95], 0.10)
    add_marker(
        "hybrid_goal",
        layout.goal_xy,
        plat.top_z + cage_r + 0.08,
        [0.95, 0.62, 0.15, 0.95],
        0.12,
    )
    add_marker(
        "hybrid_target_live",
        layout.spawn_xy,
        cage_r + 0.14,
        [0.15, 0.75, 0.95, 0.95],
        0.08,
    )

    _attach_rangefinders(
        spec,
        dense=bool(perception.get("dense_rays", True)),
        dense_ray_count=int(perception.get("dense_ray_count", 24)),
    )

    model = spec.compile()
    model.opt.timestep = float(sim.get("timestep", 0.004))
    return model


def set_live_target_marker(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xy: np.ndarray,
    z: float,
) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hybrid_target_live")
    if bid < 0:
        return
    mid = int(model.body_mocapid[bid])
    if mid >= 0:
        data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), float(z)]
