"""Log-odds occupancy grid mapping from rangefinder scans."""

from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

from carolline_control.utils.types import RobotState


class OccupancyGridMapper:
    """Maintain a 2D occupancy grid updated from rangefinder endpoints.

    Produces:
    - occupancy / log-odds grid (LIDAR hits)
    - hard inflated lethal mask (robot footprint clearance)
    - soft cost map for A* (higher near walls so paths stay centered)
    """

    def __init__(
        self,
        *,
        origin_xy: np.ndarray,
        width_m: float = 40.0,
        height_m: float = 40.0,
        resolution: float = 0.05,
        log_odds_hit: float = 0.85,
        log_odds_miss: float = -0.4,
        log_odds_min: float = -4.0,
        log_odds_max: float = 4.0,
        inflation_radius: float = 0.40,
        soft_cost_radius: float | None = None,
        cost_weight: float = 2.5,
        unknown_cost: float = 1.5,
        free_log_odds: float = -0.25,
        occupied_log_odds: float = 0.85,
        min_hit_range: float = 0.22,
    ) -> None:
        self.resolution = float(resolution)
        self.origin = np.asarray(origin_xy[:2], dtype=float)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.nx = int(math.ceil(self.width_m / self.resolution))
        self.ny = int(math.ceil(self.height_m / self.resolution))
        self._log_odds = np.zeros((self.ny, self.nx), dtype=float)
        self._log_hit = float(log_odds_hit)
        self._log_miss = float(log_odds_miss)
        self._log_min = float(log_odds_min)
        self._log_max = float(log_odds_max)
        self._inflation_radius = float(inflation_radius)
        self._soft_cost_radius = float(
            soft_cost_radius if soft_cost_radius is not None else inflation_radius * 2.0
        )
        self._cost_weight = float(cost_weight)
        self._unknown_cost = float(unknown_cost)
        self._free_log_odds = float(free_log_odds)
        self._occupied_log_odds = float(occupied_log_odds)
        self._min_hit_range = float(min_hit_range)
        self._inflated: np.ndarray | None = None
        self._cost: np.ndarray | None = None
        self._obstacle_dist_m: np.ndarray | None = None
        self._static_occupied = np.zeros((self.ny, self.nx), dtype=bool)

    def reset(self, origin_xy: np.ndarray | None = None) -> None:
        if origin_xy is not None:
            self.origin = np.asarray(origin_xy[:2], dtype=float)
        self._log_odds.fill(0.0)
        self._static_occupied.fill(False)
        self._inflated = None

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        x = self.origin[0] + (gx + 0.5) * self.resolution
        y = self.origin[1] + (gy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.nx and 0 <= gy < self.ny

    def occupancy_probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._log_odds))

    def occupied_threshold(self) -> float:
        return float(1.0 / (1.0 + math.exp(-self._occupied_log_odds)))

    def free_log_odds_threshold(self) -> float:
        return self._free_log_odds

    def occupied_log_odds_threshold(self) -> float:
        return self._occupied_log_odds

    def is_occupied(self, gx: int, gy: int, *, use_inflated: bool = True) -> bool:
        if not self.in_bounds(gx, gy):
            return True
        if use_inflated:
            grid = self.inflated_grid()
            return bool(grid[gy, gx])
        return float(self.occupancy_probability()[gy, gx]) >= self.occupied_threshold()

    def is_traversable(self, gx: int, gy: int, *, conservative: bool = False) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        if self.is_occupied(gx, gy, use_inflated=True):
            return False
        if conservative and abs(float(self._log_odds[gy, gx])) < 0.05:
            return False
        return True

    def raw_occupied_mask(self) -> np.ndarray:
        """Occupied cells from LIDAR/static seeds (no inflation)."""
        prob = self.occupancy_probability()
        return (prob >= self.occupied_threshold()) | self._static_occupied

    def obstacle_distance_m(self) -> np.ndarray:
        """Distance (m) from each cell to the nearest occupied cell."""
        self._ensure_layers()
        assert self._obstacle_dist_m is not None
        return self._obstacle_dist_m

    def cost_map(self) -> np.ndarray:
        """Soft planning cost grid. Higher near walls; lethal cells are huge."""
        self._ensure_layers()
        assert self._cost is not None
        return self._cost

    def traversal_cost(self, gx: int, gy: int) -> float:
        """A* step cost: wall proximity + poorly observed cells."""
        if not self.in_bounds(gx, gy):
            return 1e6
        return float(self.cost_map()[gy, gx])

    def inflated_grid(self) -> np.ndarray:
        self._ensure_layers()
        assert self._inflated is not None
        return self._inflated

    def invalidate_inflation_cache(self) -> None:
        self._inflated = None
        self._cost = None
        self._obstacle_dist_m = None

    def _ensure_layers(self) -> None:
        if self._inflated is not None and self._cost is not None and self._obstacle_dist_m is not None:
            return
        occupied = self.raw_occupied_mask()
        hard_cells = max(1, int(math.ceil(self._inflation_radius / self.resolution)))
        soft_cells = max(hard_cells + 1, int(math.ceil(self._soft_cost_radius / self.resolution)))

        # Hard lethal inflation (robot footprint clearance).
        ys, xs = np.where(occupied)
        inflated = occupied.copy()
        if len(xs):
            yy, xx = np.ogrid[-hard_cells : hard_cells + 1, -hard_cells : hard_cells + 1]
            disk = xx * xx + yy * yy <= hard_cells * hard_cells
            h, w = inflated.shape
            for gy, gx in zip(ys, xs):
                y0 = max(0, gy - hard_cells)
                y1 = min(h, gy + hard_cells + 1)
                x0 = max(0, gx - hard_cells)
                x1 = min(w, gx + hard_cells + 1)
                dy0 = y0 - (gy - hard_cells)
                dy1 = dy0 + (y1 - y0)
                dx0 = x0 - (gx - hard_cells)
                dx1 = dx0 + (x1 - x0)
                inflated[y0:y1, x0:x1] |= disk[dy0:dy1, dx0:dx1]

        # Multi-source BFS distance-to-obstacle (in cells).
        dist_cells = np.full((self.ny, self.nx), np.inf, dtype=float)
        queue: deque[tuple[int, int]] = deque()
        for gy, gx in zip(*np.where(occupied)):
            dist_cells[gy, gx] = 0.0
            queue.append((gx, gy))
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            gx, gy = queue.popleft()
            base = dist_cells[gy, gx]
            for dx, dy in neighbors:
                nx, ny = gx + dx, gy + dy
                if not self.in_bounds(nx, ny):
                    continue
                nd = base + 1.0
                if nd < dist_cells[ny, nx]:
                    dist_cells[ny, nx] = nd
                    queue.append((nx, ny))

        dist_m = dist_cells * self.resolution
        cost = np.zeros((self.ny, self.nx), dtype=float)
        # Soft penalty between hard and soft radii → A* prefers corridor centers.
        soft_m = self._soft_cost_radius
        hard_m = self._inflation_radius
        span = max(soft_m - hard_m, self.resolution)
        proximity = np.clip((soft_m - dist_m) / span, 0.0, 1.0)
        cost += self._cost_weight * (proximity ** 2)

        # Unknown / poorly observed free space is slightly expensive.
        unknown = np.abs(self._log_odds) < 0.05
        weakly = (np.abs(self._log_odds) >= 0.05) & (np.abs(self._log_odds) < 0.25)
        cost[unknown] += self._unknown_cost
        cost[weakly] += 0.4 * self._unknown_cost
        cost[inflated] = 1e6

        self._inflated = inflated
        self._obstacle_dist_m = dist_m
        self._cost = cost

    def seed_box_obstacle(
        self,
        center_xy: np.ndarray,
        half_xy: np.ndarray,
        *,
        log_odds: float | None = None,
    ) -> None:
        """Mark an axis-aligned rectangle as occupied (for known maze walls)."""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        hx, hy = float(half_xy[0]), float(half_xy[1])
        strength = float(self._log_hit * 2.5 if log_odds is None else log_odds)
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        for gy in range(min(gy0, gy1), max(gy0, gy1) + 1):
            for gx in range(min(gx0, gx1), max(gx0, gx1) + 1):
                if not self.in_bounds(gx, gy):
                    continue
                self._static_occupied[gy, gx] = True
                self._update_cell(gx, gy, strength)
        self.invalidate_inflation_cache()

    def mark_obstacle_ahead(
        self,
        position_xy: np.ndarray,
        *,
        target_xy: np.ndarray | None = None,
        distance: float = 0.35,
        hit_strength: float | None = None,
    ) -> None:
        """Mark a short occupied segment ahead of the robot for planning."""
        pos = np.asarray(position_xy[:2], dtype=float)
        if target_xy is not None:
            direction = np.asarray(target_xy[:2], dtype=float) - pos
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([0.0, -1.0], dtype=float)
            else:
                direction = direction / norm
        else:
            direction = np.array([0.0, -1.0], dtype=float)

        strength = float(self._log_hit if hit_strength is None else hit_strength)
        for step in (0.0, 0.5, 1.0):
            pt = pos + direction * (float(distance) * (0.6 + 0.4 * step))
            gx, gy = self.world_to_grid(float(pt[0]), float(pt[1]))
            self._update_cell(gx, gy, strength)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                self._update_cell(gx + dx, gy + dy, strength * 0.6)
        self.invalidate_inflation_cache()

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def _update_cell(self, gx: int, gy: int, delta: float) -> None:
        if not self.in_bounds(gx, gy):
            return
        # Never clear statically seeded walls with free-space ray misses.
        if delta < 0.0 and self._static_occupied[gy, gx]:
            return
        value = self._log_odds[gy, gx] + delta
        self._log_odds[gy, gx] = float(np.clip(value, self._log_min, self._log_max))

    def update_from_rays(
        self,
        state: RobotState,
        *,
        site_ids: list[int],
        site_is_up: list[bool],
        ranges: list[float],
        model: mujoco.MjModel,
        data: mujoco.MjData,
        max_range: float,
    ) -> None:
        """Mark free space along rays and occupied cells at endpoints."""
        gx0, gy0 = self.world_to_grid(float(state.position[0]), float(state.position[1]))
        self._update_cell(gx0, gy0, self._log_miss * 0.25)

        for site_id, is_up, dist in zip(site_ids, site_is_up, ranges):
            if is_up:
                continue

            site_pos = data.site_xpos[site_id]
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            direction = -site_mat[:, 2]
            ray_dist = float(dist)
            at_max_range = ray_dist >= max_range * 0.98
            if not at_max_range and ray_dist < self._min_hit_range:
                continue

            endpoint = site_pos + direction * ray_dist
            ray_gx0, ray_gy0 = self.world_to_grid(float(site_pos[0]), float(site_pos[1]))
            gx1, gy1 = self.world_to_grid(float(endpoint[0]), float(endpoint[1]))
            for gx, gy in self._bresenham(ray_gx0, ray_gy0, gx1, gy1):
                if (gx, gy) == (gx1, gy1):
                    if not at_max_range:
                        self._update_cell(gx, gy, self._log_hit)
                else:
                    self._update_cell(gx, gy, self._log_miss)

        self.invalidate_inflation_cache()

    def extract_scan_points_world(
        self,
        state: RobotState,
        *,
        site_ids: list[int],
        site_is_up: list[bool],
        ranges: list[float],
        model: mujoco.MjModel,
        data: mujoco.MjData,
        max_range: float,
    ) -> np.ndarray:
        """Return Nx2 world-frame hit points for scan matching."""
        points: list[np.ndarray] = []
        for site_id, is_up, dist in zip(site_ids, site_is_up, ranges):
            if is_up:
                continue
            if dist >= max_range * 0.98:
                continue
            site_pos = data.site_xpos[site_id]
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            direction = -site_mat[:, 2]
            endpoint = site_pos + direction * float(dist)
            points.append(endpoint[:2].copy())
        if not points:
            return np.zeros((0, 2), dtype=float)
        return np.vstack(points)
