"""Procedural heightmap: flat meadow, rolling hills, and mountainous regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency for PNG export
    raise ImportError("terrain heightmaps require Pillow (pip install pillow)") from exc


@dataclass(frozen=True)
class TerrainLayout:
    """Terrain extent and region boundaries in world coordinates [m]."""

    half_x: float = 14.0
    half_y: float = 14.0
    max_height: float = 1.45
    base_thickness: float = 0.04
    nrow: int = 128
    ncol: int = 128
    flat_x_end: float = -7.0
    hills_x_end: float = 2.0
    seed: int = 7

    @property
    def size_x(self) -> float:
        return 2.0 * self.half_x

    @property
    def size_y(self) -> float:
        return 2.0 * self.half_y


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_heightmap(layout: TerrainLayout) -> tuple[np.ndarray, np.ndarray]:
    """Return (heights[m], grayscale[uint8]) on a regular grid."""
    rng = np.random.default_rng(layout.seed)
    nrow, ncol = layout.nrow, layout.ncol
    xs = np.linspace(-layout.half_x, layout.half_x, ncol)
    ys = np.linspace(-layout.half_y, layout.half_y, nrow)
    xg, yg = np.meshgrid(xs, ys)

    heights = np.zeros((nrow, ncol), dtype=float)

    # Flat meadow (west)
    meadow = xg < layout.flat_x_end
    heights[meadow] = 0.02 * np.sin(0.35 * xg[meadow]) * np.sin(0.25 * yg[meadow])

    # Rolling hills (center)
    hill_mask = (xg >= layout.flat_x_end) & (xg < layout.hills_x_end)
    hx = (xg[hill_mask] - layout.flat_x_end) / max(layout.hills_x_end - layout.flat_x_end, 1e-6)
    hy = yg[hill_mask] / max(layout.half_y, 1e-6)
    hills = (
        0.22 * np.sin(1.8 * np.pi * hx) ** 2
        + 0.14 * np.cos(2.4 * np.pi * hy)
        + 0.10 * np.sin(3.2 * hx + 1.1 * hy)
    )
    hills *= _smoothstep(0.0, 0.25, hx)
    heights[hill_mask] = np.clip(hills, 0.0, 0.42)

    # Mountainous east
    mount_mask = xg >= layout.hills_x_end
    mx = (xg[mount_mask] - layout.hills_x_end) / max(layout.half_x - layout.hills_x_end, 1e-6)
    my = yg[mount_mask] / max(layout.half_y * 0.85, 1e-6)
    peaks = (
        0.95 * np.exp(-((mx - 0.35) ** 2 + (my * 0.75) ** 2) * 7.5)
        + 0.75 * np.exp(-((mx - 0.68) ** 2 + ((my - 0.45) * 1.1) ** 2) * 9.0)
        + 0.55 * np.exp(-((mx - 0.52) ** 2 + ((my + 0.55) * 1.0) ** 2) * 8.0)
    )
    ridge = 0.18 * np.sin(5.5 * mx + 0.8 * my) * _smoothstep(0.15, 0.85, mx)
    mountains = peaks + ridge
    mountains *= _smoothstep(0.0, 0.12, mx)
    heights[mount_mask] = np.clip(mountains, 0.0, layout.max_height)

    # Valleys / paths between features
    path = np.exp(-((xg - 1.5) ** 2 + (yg - 4.0) ** 2) / 6.0)
    path += np.exp(-((xg - 5.5) ** 2 + (yg + 3.5) ** 2) / 5.5)
    heights -= 0.18 * path
    heights = np.clip(heights, 0.0, layout.max_height)

    # Light noise for natural roughness
    noise = rng.normal(0.0, 0.012, size=heights.shape)
    heights = np.clip(heights + noise, 0.0, layout.max_height)

    gray = np.clip(heights / max(layout.max_height, 1e-6), 0.0, 1.0)
    gray_u8 = (gray * 255.0).astype(np.uint8)
    return heights, gray_u8


def write_heightmap_png(path: Path, layout: TerrainLayout) -> tuple[np.ndarray, Path]:
    """Generate and save a grayscale PNG for MuJoCo hfield loading."""
    heights, gray = generate_heightmap(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray, mode="L").save(path)
    return heights, path


def sample_terrain_height(
    x: float,
    y: float,
    heights: np.ndarray,
    layout: TerrainLayout,
) -> float:
    """Bilinear height sample from the discrete height grid [m]."""
    xs = np.linspace(-layout.half_x, layout.half_x, layout.ncol)
    ys = np.linspace(-layout.half_y, layout.half_y, layout.nrow)
    if x <= xs[0]:
        j0 = j1 = 0
        tx = 0.0
    elif x >= xs[-1]:
        j0 = j1 = layout.ncol - 1
        tx = 0.0
    else:
        j1 = int(np.searchsorted(xs, x))
        j0 = max(j1 - 1, 0)
        tx = float((x - xs[j0]) / max(xs[j1] - xs[j0], 1e-9))

    if y <= ys[0]:
        i0 = i1 = 0
        ty = 0.0
    elif y >= ys[-1]:
        i0 = i1 = layout.nrow - 1
        ty = 0.0
    else:
        i1 = int(np.searchsorted(ys, y))
        i0 = max(i1 - 1, 0)
        ty = float((y - ys[i0]) / max(ys[i1] - ys[i0], 1e-9))

    h00 = float(heights[i0, j0])
    h10 = float(heights[i0, j1])
    h01 = float(heights[i1, j0])
    h11 = float(heights[i1, j1])
    h0 = h00 * (1.0 - tx) + h10 * tx
    h1 = h01 * (1.0 - tx) + h11 * tx
    return h0 * (1.0 - ty) + h1 * ty
