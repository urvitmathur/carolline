"""Terrain-aware rolling velocity commands toward course checkpoints."""

from __future__ import annotations

import numpy as np

from carolline_control.navigation.course_layout import CourseLayout
from carolline_control.navigation.perception import ObstacleScan
from carolline_control.utils.types import ControllerConfig, RobotState


class TerrainRollingFollower:
    """Patrol-style rolling assist with obstacle stop gate."""

    def __init__(
        self,
        layout: CourseLayout,
        config: ControllerConfig,
        *,
        speed: float | None = None,
        terrain_z_at=None,
    ) -> None:
        self.layout = layout
        self.config = config
        self._terrain_z_at = terrain_z_at
        self.speed = float(speed if speed is not None else layout.patrol_speed)
        self.checkpoint_index = 0
        self.enabled = True
        self.respect_blocked_scan = True
        self._target_xy = layout.checkpoints[0].copy() if layout.checkpoints else layout.goal_xy.copy()

    @property
    def current_leg_index(self) -> int:
        return self.checkpoint_index

    def set_checkpoint_index(self, index: int) -> None:
        self.checkpoint_index = max(0, min(index, len(self.layout.checkpoints)))
        self._refresh_target()

    def advance_checkpoint(self) -> bool:
        if self.checkpoint_index >= len(self.layout.checkpoints):
            self._target_xy = self.layout.goal_xy.copy()
            return False
        self.checkpoint_index += 1
        if self.checkpoint_index >= len(self.layout.checkpoints):
            self._target_xy = self.layout.goal_xy.copy()
        else:
            self._target_xy = self.layout.checkpoints[self.checkpoint_index].copy()
        return self.checkpoint_index < len(self.layout.checkpoints)

    def _refresh_target(self) -> None:
        if self.checkpoint_index >= len(self.layout.checkpoints):
            self._target_xy = self.layout.goal_xy.copy()
        else:
            self._target_xy = self.layout.checkpoints[self.checkpoint_index].copy()

    def current_target_xy(self) -> np.ndarray:
        return self._target_xy.copy()

    def at_final_goal(self, state: RobotState) -> bool:
        if self.checkpoint_index < len(self.layout.checkpoints):
            return False
        return self.layout.near_goal(state.position[:2])

    def near_current_target(self, state: RobotState) -> bool:
        return (
            float(np.linalg.norm(state.position[:2] - self._target_xy))
            < self.layout.patrol_arrival_radius
        )

    def _grade_speed_scale(self, state: RobotState, direction: np.ndarray) -> float:
        if self._terrain_z_at is None:
            return 1.0
        probe = state.position[:2] + direction * 0.75
        z0 = float(self._terrain_z_at(float(state.position[0]), float(state.position[1])))
        z1 = float(self._terrain_z_at(float(probe[0]), float(probe[1])))
        grade = abs(z1 - z0) / 0.75
        if grade > 0.30:
            return 0.72
        if grade > 0.18:
            return 0.86
        return 1.0

    def velocity_command_xy(self, state: RobotState, scan: ObstacleScan | None = None) -> np.ndarray:
        if not self.enabled:
            return np.zeros(2, dtype=float)
        if scan is not None and scan.blocked and self.respect_blocked_scan:
            return np.zeros(2, dtype=float)

        target_xy = self._target_xy
        delta = target_xy - state.position[:2]
        dist = float(np.linalg.norm(delta))
        if dist < 1e-3:
            return np.zeros(2, dtype=float)

        direction = delta / dist
        slowdown = float(self.layout.patrol_slowdown_radius)
        if dist >= slowdown:
            cruise = self.speed
        else:
            cruise = self.speed * max(dist / slowdown, 0.70)

        cruise *= self._grade_speed_scale(state, direction)

        vel = direction * cruise
        cap = float(self.config.rolling_max_speed)
        norm = float(np.linalg.norm(vel))
        if norm > cap:
            vel *= cap / norm
        return vel
