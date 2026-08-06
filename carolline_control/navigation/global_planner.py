"""A* global planning on an inflated occupancy grid."""

from __future__ import annotations

import heapq
import math

import numpy as np

from carolline_control.navigation.mapping import OccupancyGridMapper


def _heuristic(ax: int, ay: int, bx: int, by: int) -> float:
    return math.hypot(ax - bx, ay - by)


class GlobalPlanner:
    """Grid-based A* planner producing sparse XY waypoints."""

    def __init__(self, mapper: OccupancyGridMapper) -> None:
        self.mapper = mapper

    def plan(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        *,
        min_waypoint_spacing: float = 0.8,
        conservative: bool = False,
    ) -> list[np.ndarray]:
        sx, sy = self.mapper.world_to_grid(float(start_xy[0]), float(start_xy[1]))
        gx, gy = self.mapper.world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
        if not self.mapper.in_bounds(sx, sy) or not self.mapper.in_bounds(gx, gy):
            return [np.asarray(goal_xy[:2], dtype=float)]

        path_cells = self._astar(sx, sy, gx, gy, conservative=conservative)
        if not path_cells and not conservative:
            path_cells = self._astar(sx, sy, gx, gy, conservative=True)
        if not path_cells:
            return [np.asarray(goal_xy[:2], dtype=float)]

        waypoints: list[np.ndarray] = []
        last: np.ndarray | None = None
        for cx, cy in path_cells:
            wx, wy = self.mapper.grid_to_world(cx, cy)
            pt = np.array([wx, wy], dtype=float)
            if last is not None and float(np.linalg.norm(pt - last)) < min_waypoint_spacing:
                continue
            waypoints.append(pt)
            last = pt

        goal_pt = np.asarray(goal_xy[:2], dtype=float)
        if not waypoints or float(np.linalg.norm(waypoints[-1] - goal_pt)) > 0.25:
            waypoints.append(goal_pt)
        return waypoints

    def _astar(
        self,
        sx: int,
        sy: int,
        gx: int,
        gy: int,
        *,
        conservative: bool,
    ) -> list[tuple[int, int]]:
        if not self.mapper.is_traversable(gx, gy, conservative=conservative):
            nearest = self._nearest_free(gx, gy, conservative=conservative)
            if nearest is None:
                return []
            gx, gy = nearest

        if not self.mapper.is_traversable(sx, sy, conservative=conservative):
            nearest = self._nearest_free(sx, sy, conservative=conservative)
            if nearest is None:
                return []
            sx, sy = nearest

        open_set: list[tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, sx, sy))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        g_score: dict[tuple[int, int], float] = {(sx, sy): 0.0}

        neighbors = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == (gx, gy):
                return self._reconstruct(came_from, (gx, gy))

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not self.mapper.is_traversable(nx, ny, conservative=conservative):
                    continue
                step = math.hypot(dx, dy)
                step += self.mapper.traversal_cost(nx, ny)
                tentative = g_score[(cx, cy)] + step
                key = (nx, ny)
                if tentative >= g_score.get(key, float("inf")):
                    continue
                came_from[key] = (cx, cy)
                g_score[key] = tentative
                f = tentative + _heuristic(nx, ny, gx, gy)
                heapq.heappush(open_set, (f, nx, ny))

        return []

    def _nearest_free(
        self,
        gx: int,
        gy: int,
        *,
        conservative: bool,
    ) -> tuple[int, int] | None:
        for radius in range(1, 12):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if self.mapper.is_traversable(nx, ny, conservative=conservative):
                        return nx, ny
        return None

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
