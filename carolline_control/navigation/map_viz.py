"""Debug visualization for SLAM occupancy grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carolline_control.navigation.mapping import OccupancyGridMapper

# ROS map_server convention: white=free, black=occupied, mid-gray=unknown.
_UNKNOWN = 205
_FREE = 254
_OCCUPIED = 0
_UNEXPLORED_BG = 248


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        merged = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                merged |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = merged
    return out


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=True)
        merged = np.ones_like(out)
        for dy in range(3):
            for dx in range(3):
                merged &= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = merged
    return out


def _manhattan_wall_mask(
    occupied: np.ndarray,
    *,
    origin: np.ndarray,
    resolution: float,
    bin_size: float = 0.14,
    min_points: int = 4,
    thickness_m: float = 0.10,
) -> np.ndarray:
    """Extract axis-aligned wall segments from noisy occupancy hits."""
    ys, xs = np.where(occupied)
    if len(xs) == 0:
        return occupied.copy()
    h, w = occupied.shape
    wx = origin[0] + (xs + 0.5) * resolution
    wy = origin[1] + (ys + 0.5) * resolution
    out = np.zeros((h, w), dtype=bool)
    half_t = max(1, int(round(thickness_m / (2.0 * resolution))))

    def _stamp_segment(x0: float, y0: float, x1: float, y1: float) -> None:
        gx0 = int((min(x0, x1) - origin[0]) / resolution)
        gx1 = int(np.ceil((max(x0, x1) - origin[0]) / resolution))
        gy0 = int((min(y0, y1) - origin[1]) / resolution)
        gy1 = int(np.ceil((max(y0, y1) - origin[1]) / resolution))
        gx0 = max(0, gx0 - half_t)
        gx1 = min(w, gx1 + half_t + 1)
        gy0 = max(0, gy0 - half_t)
        gy1 = min(h, gy1 + half_t + 1)
        out[gy0:gy1, gx0:gx1] = True

    x_bins: dict[int, list[float]] = {}
    y_bins: dict[int, list[float]] = {}
    for x, y in zip(wx, wy):
        x_bins.setdefault(int(round(x / bin_size)), []).append(float(y))
        y_bins.setdefault(int(round(y / bin_size)), []).append(float(x))

    for bx, yvals in x_bins.items():
        if len(yvals) < min_points:
            continue
        x = bx * bin_size
        _stamp_segment(x, min(yvals), x, max(yvals))

    for by, xvals in y_bins.items():
        if len(xvals) < min_points:
            continue
        y = by * bin_size
        _stamp_segment(min(xvals), y, max(xvals), y)

    return out


def _orthogonal_wall_mask(occupied: np.ndarray, *, min_span: int = 3, pad: int = 1) -> np.ndarray:
    """Snap noisy hit clusters to axis-aligned rectangles (orthogonal maze)."""
    h, w = occupied.shape
    out = np.zeros_like(occupied, dtype=bool)
    visited = np.zeros_like(occupied, dtype=bool)
    for gy in range(h):
        for gx in range(w):
            if not occupied[gy, gx] or visited[gy, gx]:
                continue
            stack = [(gy, gx)]
            cells: list[tuple[int, int]] = []
            visited[gy, gx] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and occupied[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            span_y = y1 - y0 + 1
            span_x = x1 - x0 + 1
            if span_y < min_span and span_x < min_span:
                continue
            y0 = max(0, y0 - pad)
            y1 = min(h - 1, y1 + pad)
            x0 = max(0, x0 - pad)
            x1 = min(w - 1, x1 + pad)
            if span_y >= span_x:
                thickness = max(2, min(4, span_x + 2 * pad))
                cx = (x0 + x1) // 2
                half = thickness // 2
                x0 = max(0, cx - half)
                x1 = min(w - 1, cx + half)
            else:
                thickness = max(2, min(4, span_y + 2 * pad))
                cy = (y0 + y1) // 2
                half = thickness // 2
                y0 = max(0, cy - half)
                y1 = min(h - 1, cy + half)
            out[y0 : y1 + 1, x0 : x1 + 1] = True
    return out


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop occupied blobs smaller than min_area cells."""
    if min_area <= 1:
        return mask
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    for gy in range(h):
        for gx in range(w):
            if not mask[gy, gx] or visited[gy, gx]:
                continue
            stack = [(gy, gx)]
            component: list[tuple[int, int]] = []
            visited[gy, gx] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) >= min_area:
                for cy, cx in component:
                    keep[cy, cx] = True
    return keep


@dataclass(frozen=True)
class MapCropBounds:
    """World-frame crop rectangle for report figures."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def around_center(cls, center_xy: tuple[float, float], half_extent: float, *, margin: float = 0.35) -> MapCropBounds:
        cx, cy = center_xy
        ext = half_extent + margin
        return cls(cx - ext, cx + ext, cy - ext, cy + ext)


def _slice_for_bounds(mapper: OccupancyGridMapper, bounds: MapCropBounds | None) -> tuple[slice, slice, list[float]]:
    if bounds is None:
        extent = [
            mapper.origin[0],
            mapper.origin[0] + mapper.width_m,
            mapper.origin[1],
            mapper.origin[1] + mapper.height_m,
        ]
        return slice(None), slice(None), extent

    gx0 = int(max(0, (bounds.x_min - mapper.origin[0]) / mapper.resolution))
    gx1 = int(min(mapper.nx, np.ceil((bounds.x_max - mapper.origin[0]) / mapper.resolution)))
    gy0 = int(max(0, (bounds.y_min - mapper.origin[1]) / mapper.resolution))
    gy1 = int(min(mapper.ny, np.ceil((bounds.y_max - mapper.origin[1]) / mapper.resolution)))
    extent = [
        mapper.origin[0] + gx0 * mapper.resolution,
        mapper.origin[0] + gx1 * mapper.resolution,
        mapper.origin[1] + gy0 * mapper.resolution,
        mapper.origin[1] + gy1 * mapper.resolution,
    ]
    return slice(gy0, gy1), slice(gx0, gx1), extent


class MapVisualizer:
    """Save or display occupancy grids (matplotlib optional)."""

    def __init__(self, mapper: OccupancyGridMapper) -> None:
        self.mapper = mapper

    def probability_image(self) -> np.ndarray:
        prob = self.mapper.occupancy_probability()
        return (np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8)

    def classic_occupancy_image(self) -> np.ndarray:
        """Thresholded map: white free space, black walls, gray unknown."""
        lodds = self.mapper._log_odds
        static = getattr(self.mapper, "_static_occupied", None)
        img = np.full(lodds.shape, _UNKNOWN, dtype=np.uint8)
        free = lodds <= self.mapper.free_log_odds_threshold()
        occupied = lodds >= self.mapper.occupied_log_odds_threshold()
        if static is not None:
            occupied = occupied | static
            free = free & ~static
        img[free] = _FREE
        img[occupied] = _OCCUPIED
        return img

    def report_occupancy_image(
        self,
        *,
        wall_dilate: int = 1,
        wall_close: int = 2,
        min_wall_area: int = 4,
        min_noise_area: int = 4,
        hide_unexplored: bool = True,
        unexplored_lodds: float = 0.06,
        crop_bounds: MapCropBounds | None = None,
        force_free_interior: bool = True,
        orthogonal_walls: bool = True,
        trajectory_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cleaned map for reports: solid walls, reduced speckle, optional crop."""
        lodds = self.mapper._log_odds
        static = getattr(self.mapper, "_static_occupied", None)
        if static is None:
            static = np.zeros_like(lodds, dtype=bool)
        observed = (np.abs(lodds) > unexplored_lodds) | static
        occupied = lodds >= self.mapper.occupied_log_odds_threshold()
        free = lodds <= self.mapper.free_log_odds_threshold()

        if np.any(static):
            # Prefer crisp known maze walls over noisy rangefinder speckles.
            wall = _binary_dilate(static, 1)
        else:
            wall = occupied.copy()
            wall = _remove_small_components(wall, min_noise_area)
            if wall_close > 0:
                wall = _binary_dilate(wall, wall_close)
                wall = _binary_erode(wall, max(1, wall_close - 1))
            if orthogonal_walls:
                wall = _manhattan_wall_mask(
                    wall,
                    origin=self.mapper.origin,
                    resolution=self.mapper.resolution,
                    bin_size=0.14,
                    min_points=4,
                    thickness_m=0.10,
                )
            elif wall_dilate > 0:
                wall = _binary_dilate(wall, wall_dilate)
            wall = _remove_small_components(wall, min_wall_area)

        img = np.full(lodds.shape, _UNKNOWN, dtype=np.uint8)
        if np.any(static):
            # Known maze walls: render a clear black/white layout so the figure
            # is readable even before rangefinders fully carve free space.
            img[:, :] = _FREE
            img[wall] = _OCCUPIED
        else:
            if hide_unexplored:
                img[~observed] = _UNEXPLORED_BG
            if force_free_interior:
                img[observed & ~wall] = _FREE
            else:
                img[observed & free & ~wall] = _FREE
            img[wall] = _OCCUPIED

        if trajectory_xy is not None and len(trajectory_xy) >= 2:
            radius_cells = max(1, int(round(0.38 / self.mapper.resolution)))
            for pt in np.asarray(trajectory_xy, dtype=float):
                gx, gy = self.mapper.world_to_grid(float(pt[0]), float(pt[1]))
                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        nx, ny = gx + dx, gy + dy
                        if self.mapper.in_bounds(nx, ny) and not wall[ny, nx]:
                            img[ny, nx] = _FREE

        sy, sx, _ = _slice_for_bounds(self.mapper, crop_bounds)
        return img[sy, sx]

    def save_png(
        self,
        path: str | Path,
        *,
        classic: bool = False,
        crop_bounds: MapCropBounds | None = None,
        trajectory_xy: np.ndarray | None = None,
        goal_xy: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        title: str = "SLAM occupancy map",
    ) -> Path:
        """Save map PNG. Default uses cleaned report view (not raw log-odds rays)."""
        if classic:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required to save map PNGs") from exc
            img_arr = self.classic_occupancy_image()
            if crop_bounds is not None:
                sy, sx, _ = _slice_for_bounds(self.mapper, crop_bounds)
                img_arr = img_arr[sy, sx]
            Image.fromarray(img_arr, mode="L").save(out)
            return out
        return self.save_report_png(
            path,
            goal_xy=goal_xy,
            spawn_xy=spawn_xy,
            trajectory_xy=trajectory_xy,
            crop_bounds=crop_bounds,
            title=title,
        )

    def save_cost_png(
        self,
        path: str | Path,
        *,
        crop_bounds: MapCropBounds | None = None,
        goal_xy: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        trajectory_xy: np.ndarray | None = None,
        planned_xy: np.ndarray | None = None,
        title: str = "SLAM Cost Map (A*)",
    ) -> Path:
        """Clear labeled cost map: white free space, wall outlines, soft cost halo."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        occupied_raw = self.mapper.raw_occupied_mask()
        soft_m = float(getattr(self.mapper, "_soft_cost_radius", 1.1))
        hard_m = float(getattr(self.mapper, "_inflation_radius", 0.5))
        sy, sx, extent = _slice_for_bounds(self.mapper, crop_bounds)
        # Prefer solid maze walls over noisy rangefinder speckles for the figure.
        occupied = occupied_raw[sy, sx]
        static = self.mapper._static_occupied[sy, sx]
        occupied = _remove_small_components(occupied, min_area=10) | static

        # Distance-to-obstacle on the cleaned mask (for a smooth halo).
        dist = np.full(occupied.shape, np.inf, dtype=float)
        from collections import deque

        q: deque[tuple[int, int]] = deque()
        ys, xs = np.where(occupied)
        for gy, gx in zip(ys, xs):
            dist[gy, gx] = 0.0
            q.append((gx, gy))
        h, w = occupied.shape
        while q:
            gx, gy = q.popleft()
            base = dist[gy, gx]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < w and 0 <= ny < h and base + 1.0 < dist[ny, nx]:
                    dist[ny, nx] = base + 1.0
                    q.append((nx, ny))
        dist_m = dist * float(self.mapper.resolution)

        # Normalized soft cost: 0 in open space, 1 at the wall.
        prox = np.clip((soft_m - dist_m) / max(soft_m, 1e-3), 0.0, 1.0)
        prox = prox ** 0.85
        prox[occupied] = 1.0

        try:
            import matplotlib.pyplot as plt
            from matplotlib import colors
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
        except ImportError:
            from PIL import Image as PILImage

            img = (prox * 255.0).astype(np.uint8)
            PILImage.fromarray(img, mode="L").save(out)
            return out

        # White free space; purple→orange→yellow only in the wall halo.
        cmap = colors.LinearSegmentedColormap.from_list(
            "wall_cost",
            ["#ffffff", "#f3e9ff", "#c39bd3", "#e67e22", "#f4d03f", "#fdebd0"],
        )
        rgba = cmap(prox)
        rgba[prox < 0.08] = colors.to_rgba("#ffffff")
        rgba[occupied] = colors.to_rgba("#2b2b2b")

        fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
        ax.set_facecolor("#ffffff")
        ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            aspect="equal",
            zorder=1,
        )
        # Crisp obstacle outlines.
        ax.contour(
            occupied.astype(float),
            levels=[0.5],
            colors="#111111",
            linewidths=1.6,
            extent=extent,
            origin="lower",
            zorder=2,
        )
        # Soft / hard clearance rings (legend cues).
        ax.contour(
            dist,
            levels=[hard_m, soft_m],
            colors=["#ffb000", "#9b59b6"],
            linewidths=[0.8, 0.7],
            linestyles=["--", ":"],
            extent=extent,
            origin="lower",
            alpha=0.75,
            zorder=2,
        )

        if planned_xy is not None and len(planned_xy) >= 2:
            ax.plot(
                planned_xy[:, 0],
                planned_xy[:, 1],
                color="#1f77b4",
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label="Planned (A*)",
                zorder=3,
            )
        if trajectory_xy is not None and len(trajectory_xy) >= 2:
            ax.plot(
                trajectory_xy[:, 0],
                trajectory_xy[:, 1],
                color="#1f77b4",
                linewidth=2.0,
                alpha=0.95,
                label="Trajectory",
                zorder=4,
            )
            # Direction arrows along the path.
            step = max(1, len(trajectory_xy) // 12)
            for i in range(step, len(trajectory_xy) - 1, step):
                p0 = trajectory_xy[i - 1]
                p1 = trajectory_xy[i]
                d = p1 - p0
                if float(np.linalg.norm(d)) < 1e-3:
                    continue
                ax.annotate(
                    "",
                    xy=(p1[0], p1[1]),
                    xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
                    zorder=5,
                )
        if spawn_xy is not None:
            ax.plot(
                float(spawn_xy[0]),
                float(spawn_xy[1]),
                "o",
                color="#2ca02c",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label="Spawn",
                zorder=6,
            )
        if goal_xy is not None:
            ax.plot(
                float(goal_xy[0]),
                float(goal_xy[1]),
                "s",
                color="#d62728",
                markersize=11,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label="Goal",
                zorder=6,
            )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title, fontsize=13, pad=10)
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.85, zorder=0)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(0.0, 1.0))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Cost")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Low (free)", "Medium", "High (near wall)"])

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="Spawn"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markersize=8, label="Goal"),
            Line2D([0], [0], color="#1f77b4", lw=2.0, label="Trajectory"),
            Patch(facecolor="#2b2b2b", edgecolor="#111111", label="Obstacles"),
            Patch(facecolor=cmap(0.75), edgecolor="none", label="Cost gradient"),
        ]
        if planned_xy is not None and len(np.asarray(planned_xy)) >= 2:
            handles.insert(3, Line2D([0], [0], color="#1f77b4", lw=1.2, linestyle="--", label="Planned (A*)"))
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)

        # Bottom caption matching the reference figure style.
        ax.text(
            0.5,
            -0.12,
            "Spawn · Goal · Trajectory · Obstacles · Cost gradient (low → high near walls)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#444444",
        )
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return out

    def save_log_odds(self, path: str | Path, *, trajectory_xy: np.ndarray | None = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "origin": self.mapper.origin.copy(),
            "resolution": self.mapper.resolution,
            "width_m": self.mapper.width_m,
            "height_m": self.mapper.height_m,
        }
        payload = {k: np.asarray(v) for k, v in meta.items()}
        payload["log_odds"] = self.mapper._log_odds
        if trajectory_xy is not None and len(trajectory_xy):
            payload["trajectory_xy"] = np.asarray(trajectory_xy, dtype=float)
        np.savez_compressed(out, **payload)
        return out

    @staticmethod
    def _draw_report_axes(
        ax,
        img: np.ndarray,
        extent: list[float],
        *,
        title: str,
        spawn_xy: np.ndarray | None = None,
        goal_xy: np.ndarray | None = None,
        trajectory_xy: np.ndarray | None = None,
        show_legend: bool = True,
    ) -> None:
        from matplotlib import colors
        from matplotlib import pyplot as plt

        # Occupancy values: 0=occupied, ~205=unknown, ~248=unexplored, ~254=free
        cmap = colors.ListedColormap(["#1a1a1a", "#c8c8c8", "#f7f7f7"])
        norm = colors.BoundaryNorm([-0.5, 127.5, 245.5, 255.5], cmap.N)
        ax.set_facecolor("#f0f0f0")
        ax.imshow(
            img,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
            zorder=1,
        )
        if trajectory_xy is not None and len(trajectory_xy) >= 2:
            ax.plot(
                trajectory_xy[:, 0],
                trajectory_xy[:, 1],
                color="#1f77b4",
                linewidth=1.4,
                alpha=0.85,
                label="Trajectory",
                zorder=3,
            )
        if spawn_xy is not None:
            ax.plot(float(spawn_xy[0]), float(spawn_xy[1]), "o", color="#2ca02c", markersize=8, label="Spawn", zorder=4)
        if goal_xy is not None:
            ax.plot(float(goal_xy[0]), float(goal_xy[1]), "s", color="#d62728", markersize=9, label="Goal", zorder=4)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(False)
        if show_legend and (spawn_xy is not None or goal_xy is not None or trajectory_xy is not None):
            ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    def save_report_png(
        self,
        path: str | Path,
        *,
        goal_xy: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        trajectory_xy: np.ndarray | None = None,
        crop_bounds: MapCropBounds | None = None,
        title: str = "SLAM occupancy map",
    ) -> Path:
        """High-quality cleaned map for dissertation/report figures."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img = self.report_occupancy_image(crop_bounds=crop_bounds, trajectory_xy=trajectory_xy)
        _, _, extent = _slice_for_bounds(self.mapper, crop_bounds)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            from PIL import Image as PILImage

            PILImage.fromarray(img, mode="L").save(out)
            return out

        fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
        self._draw_report_axes(
            ax,
            img,
            extent,
            title=title,
            spawn_xy=spawn_xy,
            goal_xy=goal_xy,
            trajectory_xy=trajectory_xy,
        )
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return out

    def save_comparison_png(
        self,
        path: str | Path,
        ground_truth_img: np.ndarray,
        *,
        goal_xy: np.ndarray | None = None,
        spawn_xy: np.ndarray | None = None,
        trajectory_xy: np.ndarray | None = None,
        crop_bounds: MapCropBounds | None = None,
    ) -> Path:
        """Side-by-side ground truth vs cleaned SLAM map."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        slam = self.report_occupancy_image(crop_bounds=crop_bounds, trajectory_xy=trajectory_xy)
        sy, sx, extent = _slice_for_bounds(self.mapper, crop_bounds)
        gt = ground_truth_img[sy, sx]
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), constrained_layout=True)
        for ax, data, subt in (
            (axes[0], gt, "Ground-truth maze layout"),
            (axes[1], slam, "SLAM map (oracle pose, cleaned)"),
        ):
            self._draw_report_axes(
                ax,
                data,
                extent,
                title=subt,
                spawn_xy=spawn_xy,
                goal_xy=goal_xy,
                trajectory_xy=trajectory_xy if ax is axes[1] else None,
                show_legend=ax is axes[1],
            )
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return out

    def try_show(
        self,
        *,
        robot_xy: np.ndarray | None = None,
        goal_xy: np.ndarray | None = None,
        classic: bool = True,
        report: bool = False,
        crop_bounds: MapCropBounds | None = None,
    ) -> bool:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        _, _, extent = _slice_for_bounds(self.mapper, crop_bounds)
        plt.figure(figsize=(7, 7))
        if report:
            img = self.report_occupancy_image(crop_bounds=crop_bounds)
        elif classic:
            img = self.classic_occupancy_image()
            sy, sx, _ = _slice_for_bounds(self.mapper, crop_bounds)
            img = img[sy, sx]
        else:
            prob = self.mapper.occupancy_probability()
            sy, sx, _ = _slice_for_bounds(self.mapper, crop_bounds)
            prob = prob[sy, sx]
            plt.imshow(prob, origin="lower", extent=extent, cmap="gray_r", vmin=0.0, vmax=1.0)
            if robot_xy is not None:
                plt.plot(float(robot_xy[0]), float(robot_xy[1]), "bo", markersize=8, label="robot")
            if goal_xy is not None:
                plt.plot(float(goal_xy[0]), float(goal_xy[1]), "r*", markersize=12, label="goal")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("SLAM occupancy map")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
            return True
        self._draw_report_axes(
            plt.gca(),
            img,
            extent,
            title="SLAM occupancy map (report view)",
            goal_xy=goal_xy,
            spawn_xy=None,
            trajectory_xy=None,
        )
        if robot_xy is not None:
            plt.plot(float(robot_xy[0]), float(robot_xy[1]), "bo", markersize=8, label="robot")
            plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        return True
