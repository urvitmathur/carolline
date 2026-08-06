"""Lookahead waypoint follower for SLAM path tracking."""

from __future__ import annotations

import numpy as np

from carolline_control.navigation.perception import ObstacleScan
from carolline_control.utils.types import ControllerConfig, RobotState


class WaypointFollower:
    """Follow an A* polyline with a pure-pursuit style lookahead carrot."""

    def __init__(
        self,
        config: ControllerConfig,
        *,
        speed: float = 2.0,
        arrival_radius: float = 0.45,
        slowdown_radius: float = 1.0,
        lookahead_m: float = 0.95,
    ) -> None:
        self.config = config
        self.speed = float(speed)
        self.arrival_radius = float(arrival_radius)
        self.slowdown_radius = float(slowdown_radius)
        self.lookahead_m = float(lookahead_m)
        self.waypoints: list[np.ndarray] = []
        self.index = 0
        self.enabled = True
        self.respect_blocked_scan = False
        self._carrot_xy = np.zeros(2, dtype=float)

    def set_waypoints(self, waypoints: list[np.ndarray]) -> None:
        self.waypoints = [np.asarray(w[:2], dtype=float).copy() for w in waypoints]
        self.index = 0
        if self.waypoints:
            self._carrot_xy = self.waypoints[0].copy()

    def current_target_xy(self) -> np.ndarray:
        """Lookahead carrot (what the cage steers toward)."""
        if self.waypoints:
            return self._carrot_xy.copy()
        return np.zeros(2, dtype=float)

    def current_waypoint_xy(self) -> np.ndarray:
        if not self.waypoints:
            return np.zeros(2, dtype=float)
        idx = min(self.index, len(self.waypoints) - 1)
        return self.waypoints[idx].copy()

    def at_final_waypoint(self, state: RobotState) -> bool:
        if not self.waypoints:
            return True
        if self.index < len(self.waypoints) - 1:
            return False
        return self.near_current_target(state)

    def near_current_target(self, state: RobotState) -> bool:
        target = self.current_waypoint_xy()
        return float(np.linalg.norm(state.position[:2] - target)) < self.arrival_radius

    def advance(self) -> bool:
        if self.index >= len(self.waypoints) - 1:
            return False
        self.index += 1
        return True

    def snap_to_nearest(
        self,
        state: RobotState,
        *,
        prefer_ahead: bool = True,
        goal_xy: np.ndarray | None = None,
    ) -> None:
        if not self.waypoints:
            return
        pos = state.position[:2]
        goal = None if goal_xy is None else np.asarray(goal_xy[:2], dtype=float)
        best_idx = 0
        best_score = float("inf")
        for idx, wp in enumerate(self.waypoints):
            dist = float(np.linalg.norm(wp - pos))
            score = dist
            if prefer_ahead:
                score -= 0.04 * idx
            if goal is not None:
                score += 0.30 * float(np.linalg.norm(wp - goal))
            if score < best_score:
                best_score = score
                best_idx = idx
        self.index = best_idx
        self._update_carrot(pos)

    def _closest_polyline_index(self, pos: np.ndarray) -> tuple[int, float]:
        """Return segment start index and arc distance along that segment."""
        if len(self.waypoints) == 1:
            return 0, 0.0
        best_i = max(0, self.index - 1)
        best_t = 0.0
        best_d = float("inf")
        start = max(0, self.index - 2)
        end = min(len(self.waypoints) - 1, self.index + 6)
        for i in range(start, end):
            a = self.waypoints[i]
            b = self.waypoints[min(i + 1, len(self.waypoints) - 1)]
            ab = b - a
            length = float(np.linalg.norm(ab))
            if length < 1e-6:
                d = float(np.linalg.norm(pos - a))
                t = 0.0
            else:
                t = float(np.clip(np.dot(pos - a, ab) / (length * length), 0.0, 1.0))
                proj = a + t * ab
                d = float(np.linalg.norm(pos - proj))
            if d < best_d:
                best_d = d
                best_i = i
                best_t = t * length
        return best_i, best_t

    def _update_carrot(self, pos: np.ndarray) -> np.ndarray:
        if not self.waypoints:
            self._carrot_xy = pos.copy()
            return self._carrot_xy
        if len(self.waypoints) == 1:
            self._carrot_xy = self.waypoints[0].copy()
            return self._carrot_xy

        seg_i, along = self._closest_polyline_index(pos)
        self.index = min(seg_i, len(self.waypoints) - 1)
        remaining = float(self.lookahead_m)
        i = seg_i
        # Advance from projection point along the path by lookahead distance.
        a = self.waypoints[i]
        b = self.waypoints[min(i + 1, len(self.waypoints) - 1)]
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        if seg_len > 1e-6:
            cursor = a + ab * (along / seg_len)
            left_on_seg = seg_len - along
        else:
            cursor = a.copy()
            left_on_seg = 0.0

        if remaining <= left_on_seg and seg_len > 1e-6:
            self._carrot_xy = cursor + (ab / seg_len) * remaining
            return self._carrot_xy

        remaining -= left_on_seg
        i += 1
        while i < len(self.waypoints) - 1 and remaining > 0.0:
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len <= 1e-6:
                i += 1
                continue
            if remaining <= seg_len:
                self._carrot_xy = a + (ab / seg_len) * remaining
                self.index = i
                return self._carrot_xy
            remaining -= seg_len
            i += 1
        self.index = len(self.waypoints) - 1
        self._carrot_xy = self.waypoints[-1].copy()
        return self._carrot_xy

    def velocity_command_xy(
        self,
        state: RobotState,
        scan: ObstacleScan | None = None,
        *,
        repulsion_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        _ = scan
        _ = repulsion_xy
        if not self.enabled or not self.waypoints:
            return np.zeros(2, dtype=float)

        pos = state.position[:2]
        carrot = self._update_carrot(pos)
        # Keep discrete waypoint index progressing for mission logic.
        while (
            self.index < len(self.waypoints) - 1
            and float(np.linalg.norm(pos - self.waypoints[self.index])) < self.arrival_radius
        ):
            self.index += 1

        delta = carrot - pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-3:
            return np.zeros(2, dtype=float)

        direction = delta / dist
        goal_dist = float(np.linalg.norm(self.waypoints[-1] - pos))
        if goal_dist >= self.slowdown_radius:
            cruise = self.speed
        else:
            cruise = self.speed * max(goal_dist / self.slowdown_radius, 0.45)

        vel = direction * cruise
        cap = float(self.config.rolling_max_speed)
        norm = float(np.linalg.norm(vel))
        if norm > cap:
            vel *= cap / norm
        return vel
