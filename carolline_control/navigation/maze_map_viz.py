"""Ground-truth and report-quality maze map figures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from carolline_control.navigation.map_viz import MapCropBounds, MapVisualizer
from carolline_control.navigation.maze_layout import MazeLayout


def _rasterize_box(
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    cx: float,
    cy: float,
    half_x: float,
    half_y: float,
) -> None:
    x0, x1 = cx - half_x, cx + half_x
    y0, y1 = cy - half_y, cy + half_y
    gx0 = int(max(0, (x0 - origin[0]) / resolution))
    gx1 = int(min(grid.shape[1], math.ceil((x1 - origin[0]) / resolution)))
    gy0 = int(max(0, (y0 - origin[1]) / resolution))
    gy1 = int(min(grid.shape[0], math.ceil((y1 - origin[1]) / resolution)))
    grid[gy0:gy1, gx0:gx1] = True


def rasterize_maze(
    layout: MazeLayout,
    *,
    origin_xy: np.ndarray,
    width_m: float,
    height_m: float,
    resolution: float,
) -> np.ndarray:
    """Return boolean grid (True = wall) aligned with SLAM map frame."""
    nx = int(math.ceil(width_m / resolution))
    ny = int(math.ceil(height_m / resolution))
    wall = np.zeros((ny, nx), dtype=bool)
    origin = np.asarray(origin_xy[:2], dtype=float)
    a = layout.arena_half
    t = layout.wall_thickness
    gap = layout.entrance_width
    off = layout.entrance_offset

    # Perimeter (north has entrance gap)
    _rasterize_box(wall, origin, resolution, 0.0, -a - t, a + t, t)
    _rasterize_box(wall, origin, resolution, -a - t, 0.0, t, a + t)
    _rasterize_box(wall, origin, resolution, a + t, 0.0, t, a + t)
    gap_center = off
    half_left = (a + gap_center - gap * 0.5) * 0.5
    cx_left = -a + half_left
    half_right = (a - gap_center - gap * 0.5) * 0.5
    cx_right = a - half_right
    if half_left > 0.05:
        _rasterize_box(wall, origin, resolution, cx_left, a + t, half_left, t)
    if half_right > 0.05:
        _rasterize_box(wall, origin, resolution, cx_right, a + t, half_right, t)

    for seg in layout.segments:
        _rasterize_box(wall, origin, resolution, seg.x, seg.y, seg.half_x, seg.half_y)
    return wall


def ground_truth_image(
    layout: MazeLayout,
    *,
    origin_xy: np.ndarray,
    width_m: float,
    height_m: float,
    resolution: float,
) -> np.ndarray:
    """White free, black walls, light-gray exterior (report style)."""
    wall = rasterize_maze(
        layout,
        origin_xy=origin_xy,
        width_m=width_m,
        height_m=height_m,
        resolution=resolution,
    )
    img = np.full(wall.shape, 230, dtype=np.uint8)
    img[~wall] = 254
    img[wall] = 0
    return img


def save_ground_truth_png(
    layout: MazeLayout,
    path: str | Path,
    *,
    origin_xy: np.ndarray,
    width_m: float,
    height_m: float,
    resolution: float,
    goal_xy: np.ndarray | None = None,
    spawn_xy: np.ndarray | None = None,
    crop_bounds: MapCropBounds | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    img_arr = ground_truth_image(
        layout,
        origin_xy=origin_xy,
        width_m=width_m,
        height_m=height_m,
        resolution=resolution,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if crop_bounds is not None:
        gx0 = int(max(0, (crop_bounds.x_min - origin_xy[0]) / resolution))
        gx1 = int(min(img_arr.shape[1], np.ceil((crop_bounds.x_max - origin_xy[0]) / resolution)))
        gy0 = int(max(0, (crop_bounds.y_min - origin_xy[1]) / resolution))
        gy1 = int(min(img_arr.shape[0], np.ceil((crop_bounds.y_max - origin_xy[1]) / resolution)))
        img_arr = img_arr[gy0:gy1, gx0:gx1]
        extent = [crop_bounds.x_min, crop_bounds.x_max, crop_bounds.y_min, crop_bounds.y_max]
    else:
        extent = [origin_xy[0], origin_xy[0] + width_m, origin_xy[1], origin_xy[1] + height_m]

    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
    MapVisualizer._draw_report_axes(
        ax,
        img_arr,
        extent,
        title="Ground-truth maze layout",
        spawn_xy=spawn_xy,
        goal_xy=goal_xy,
        trajectory_xy=None,
    )
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out
