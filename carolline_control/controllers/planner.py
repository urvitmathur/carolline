"""
Trajectory planner for waypoint navigation.

Generates smooth position/velocity/acceleration references using
minimum-jerk segments between waypoints.
"""

from __future__ import annotations

import numpy as np

from carolline_control.utils.types import ControllerConfig, RobotState, TrajectoryTarget


class Planner:
    """Simple waypoint planner with segment timing."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._mission_waypoints = (
            [np.array(w, dtype=float) for w in config.waypoints]
            if config.waypoints
            else [np.array([0.0, 0.0, config.hover_height])]
        )
        self._waypoints = [wp.copy() for wp in self._mission_waypoints]
        self._segment_duration = config.segment_duration
        self._segment_index = 0
        self._segment_time = 0.0
        self._mission_done = False
        self._segment_start: np.ndarray | None = None
        self._flight_yaw = float(config.mission_yaw)
        self._dwelling = False
        self._dwell_timer = 0.0

    def set_flight_yaw(self, yaw: float) -> None:
        self._flight_yaw = float(yaw)

    @property
    def flight_yaw(self) -> float:
        return self._flight_yaw

    @property
    def mission_complete(self) -> bool:
        """True once the final waypoint segment has finished."""
        return self._mission_done

    def reset(self) -> None:
        self._segment_index = 0
        self._segment_time = 0.0
        self._mission_done = False
        self._segment_start = None
        self._flight_yaw = float(self._config.mission_yaw)
        self._waypoints = [wp.copy() for wp in self._mission_waypoints]
        self._dwelling = False
        self._dwell_timer = 0.0

    def begin_flight(self, state: RobotState, yaw: float | None = None) -> None:
        """Snapshot the first segment start when FLIGHT mode begins."""
        self.reset()
        if yaw is not None:
            self._flight_yaw = float(yaw)
        self._segment_start = state.position.copy()
        self._segment_start[2] = self._config.hover_height

        home = np.array(
            [
                self._config.roll_target[0],
                self._config.roll_target[1],
                self._config.hover_height,
            ],
            dtype=float,
        )
        dist_home = float(np.linalg.norm(state.position[:2] - home[:2]))
        if dist_home > self._config.waypoint_reach_tolerance:
            self._waypoints = [home, *self._waypoints]

    def _current_segment_end(self) -> np.ndarray:
        idx = min(self._segment_index, len(self._waypoints) - 1)
        return self._waypoints[idx]

    def _current_segment_start(self) -> np.ndarray:
        if self._segment_index == 0:
            if self._segment_start is None:
                raise RuntimeError("Planner.begin_flight() must be called before the first update.")
            return self._segment_start
        return self._waypoints[self._segment_index - 1]

    @staticmethod
    def _min_jerk_coefficients(tau: float) -> tuple[float, float, float]:
        """Minimum-jerk basis at normalized time tau in [0, 1]."""
        tau = float(np.clip(tau, 0.0, 1.0))
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3)
        return s, s_dot, s_ddot

    def _advance_segment(self) -> None:
        if self._segment_index < len(self._waypoints) - 1:
            self._segment_index += 1
            self._segment_time = 0.0
        else:
            self._mission_done = True

    def _hold_at_waypoint(self, waypoint: np.ndarray) -> TrajectoryTarget:
        """Zero-velocity hold reference while dwelling at a reached waypoint."""
        position = waypoint.copy()
        position[2] = self._config.hover_height
        return TrajectoryTarget(
            position=position,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=self._flight_yaw,
        )

    def update(self, state: RobotState, dt: float) -> TrajectoryTarget:
        """Advance planner and return current target."""
        p1 = self._current_segment_end()
        tol = self._config.waypoint_reach_tolerance
        dist_xy = float(np.linalg.norm(state.position[:2] - p1[:2]))
        speed_xy = float(np.linalg.norm(state.velocity[:2]))
        at_goal = dist_xy < tol and speed_xy < 0.3

        if self._dwelling:
            self._dwell_timer += dt
            hold = self._hold_at_waypoint(p1)
            if self._dwell_timer >= self._config.waypoint_dwell_time:
                self._dwelling = False
                self._dwell_timer = 0.0
                self._advance_segment()
                self._segment_time = 0.0
            return hold

        self._segment_time += dt
        max_dwell = self._segment_duration * 2.5
        if at_goal and self._config.waypoint_dwell_time > 0.0:
            self._dwelling = True
            self._dwell_timer = 0.0
            return self._hold_at_waypoint(p1)
        if self._segment_time >= max_dwell:
            self._advance_segment()
            self._segment_time = 0.0

        p0 = self._current_segment_start()
        p1 = self._current_segment_end()
        tau = min(self._segment_time / self._segment_duration, 1.0)
        s, s_dot, s_ddot = self._min_jerk_coefficients(tau)
        T = self._segment_duration

        position = p0 + (p1 - p0) * s
        velocity = (p1 - p0) * (s_dot / T)
        acceleration = (p1 - p0) * (s_ddot / (T * T))
        position[2] = self._config.hover_height
        velocity[2] = 0.0
        acceleration[2] = 0.0
        return TrajectoryTarget(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            yaw=self._flight_yaw,
        )

    def rolling_velocity(self, state: RobotState) -> np.ndarray:
        """Generate inertial rolling velocity; the inner loop tracks it directly."""
        err = self._config.roll_target - state.position[:2]
        dist = float(np.linalg.norm(err))
        tol = self._config.roll_position_tolerance

        if dist <= tol:
            return np.zeros(2)

        if dist < 3.0 * tol:
            speed_cap = self._config.rolling_max_speed * float(
                np.clip((dist - tol) / max(2.0 * tol, 1e-6), 0.15, 1.0)
            )
        else:
            speed_cap = self._config.rolling_max_speed

        # Eq. (15) already closes the angular-rate loop. Subtracting measured
        # velocity here double-damps the cascade and wastes motor authority.
        braking_distance = max(self._config.rolling_braking_distance, 3.0 * tol)
        taper = float(
            np.clip(
                (dist - tol) / max(braking_distance - tol, 1e-6),
                0.10,
                1.0,
            )
        )
        return (err / dist) * speed_cap * taper

    def rolling_target_position(self, state: RobotState) -> np.ndarray:
        """World-frame roll goal used for logging and targets."""
        return np.array(
            [self._config.roll_target[0], self._config.roll_target[1], state.position[2]],
            dtype=float,
        )
