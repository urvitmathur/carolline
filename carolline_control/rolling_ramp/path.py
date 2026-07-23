"""Ramp path geometry and along-path mission planner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from carolline_control.utils.types import ControllerConfig, RobotState, TrajectoryTarget


class MissionPhase(str, Enum):
    FORWARD = "FORWARD"
    RETURN = "RETURN"
    DONE = "DONE"


@dataclass
class RampGeometry:
    """Piecewise path: flat → ramp up → flat → ramp down → flat."""

    angle_deg: float = 8.0
    flat_start: float = 2.0
    ramp_length: float = 3.5
    flat_top: float = 4.0
    flat_end: float = 2.0
    width: float = 2.5
    y_center: float = 0.0
    cage_radius: float = 0.40

    @property
    def angle_rad(self) -> float:
        return np.radians(self.angle_deg)

    @property
    def rise(self) -> float:
        return self.ramp_length * np.sin(self.angle_rad)

    @property
    def ramp_run(self) -> float:
        return self.ramp_length * np.cos(self.angle_rad)

    @property
    def total_length(self) -> float:
        return self.flat_start + self.ramp_length + self.flat_top + self.ramp_length + self.flat_end

    def segment_kind(self, s: float) -> str:
        s = float(np.clip(s, 0.0, self.total_length))
        if s <= self.flat_start:
            return "flat"
        if s <= self.flat_start + self.ramp_length:
            return "up"
        if s <= self.flat_start + self.ramp_length + self.flat_top:
            return "top"
        if s <= self.flat_start + 2 * self.ramp_length + self.flat_top:
            return "down"
        return "flat"

    def normal_at_s(self, s: float) -> np.ndarray:
        kind = self.segment_kind(s)
        a = self.angle_rad
        if kind == "up":
            return np.array([-np.sin(a), 0.0, np.cos(a)], dtype=float)
        if kind == "down":
            return np.array([np.sin(a), 0.0, np.cos(a)], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def position_at_s(self, s: float) -> np.ndarray:
        """Cage center position in world frame for arclength s along the course."""
        s = float(np.clip(s, 0.0, self.total_length))
        a = self.angle_rad

        if s <= self.flat_start:
            x = s
            z_surface = 0.0
        elif s <= self.flat_start + self.ramp_length:
            ds = s - self.flat_start
            x = self.flat_start + ds * np.cos(a)
            z_surface = ds * np.sin(a)
        elif s <= self.flat_start + self.ramp_length + self.flat_top:
            ds = s - (self.flat_start + self.ramp_length)
            x = self.flat_start + self.ramp_run + ds
            z_surface = self.rise
        elif s <= self.flat_start + 2 * self.ramp_length + self.flat_top:
            ds = s - (self.flat_start + self.ramp_length + self.flat_top)
            x = self.flat_start + self.ramp_run + self.flat_top + ds * np.cos(a)
            z_surface = self.rise - ds * np.sin(a)
        else:
            ds = s - (self.flat_start + 2 * self.ramp_length + self.flat_top)
            x_end = self.flat_start + 2 * self.ramp_run + self.flat_top
            x = x_end + ds
            z_surface = 0.0

        return np.array([x, self.y_center, z_surface + self.cage_radius], dtype=float)

    def tangent_at_s(self, s: float) -> np.ndarray:
        eps = 0.02
        p0 = self.position_at_s(max(0.0, s - eps))
        p1 = self.position_at_s(min(self.total_length, s + eps))
        t = p1 - p0
        n = float(np.linalg.norm(t))
        return t / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def project_s(self, position: np.ndarray) -> float:
        """Project position to arclength using the monotonic X profile (robust on slopes)."""
        x = float(position[0])
        a = self.angle_rad
        x0 = self.flat_start
        x1 = x0 + self.ramp_run
        x2 = x1 + self.flat_top
        x3 = x2 + self.ramp_run
        x4 = x3 + self.flat_end

        if x <= x0:
            s = max(0.0, x)
        elif x <= x1:
            s = x0 + (x - x0) / max(np.cos(a), 1e-6)
        elif x <= x2:
            s = x0 + self.ramp_length + (x - x1)
        elif x <= x3:
            s = x0 + self.ramp_length + self.flat_top + (x - x2) / max(np.cos(a), 1e-6)
        else:
            s = x0 + 2 * self.ramp_length + self.flat_top + max(0.0, x - x3)

        s = float(np.clip(s, 0.0, self.total_length))

        # Refine locally with 3D distance
        local = np.linspace(max(0, s - 0.8), min(self.total_length, s + 0.8), 80)
        pts = np.array([self.position_at_s(v) for v in local])
        idx = int(np.argmin(np.linalg.norm(pts - position, axis=1)))
        return float(local[idx])


class RampPathPlanner:
    """Arc-length path follower with cross-track correction and gated progress."""

    def __init__(
        self,
        geometry: RampGeometry,
        config: ControllerConfig,
        *,
        lookahead: float = 0.35,
        path_kp: float = 0.55,
        path_kd: float = 0.85,
        cross_kp: float = 0.9,
        cross_kd: float = 0.6,
    ) -> None:
        self.geometry = geometry
        self._config = config
        self._lookahead = lookahead
        self._path_kp = path_kp
        self._path_kd = path_kd
        self._cross_kp = cross_kp
        self._cross_kd = cross_kd
        self._s_des = 0.0
        self._s_act = 0.0
        self._phase = MissionPhase.FORWARD
        self._cruise_fwd = config.rolling_max_speed * 0.75
        self._cruise_ret = config.rolling_max_speed * 0.50

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def s_desired(self) -> float:
        return self._s_des

    @property
    def s_actual(self) -> float:
        return self._s_act

    def reset(self) -> None:
        self._s_des = 0.0
        self._s_act = 0.0
        self._phase = MissionPhase.FORWARD

    def _direction(self) -> float:
        return -1.0 if self._phase == MissionPhase.RETURN else 1.0

    def _cross_track(self, position: np.ndarray) -> float:
        return float(position[1] - self.geometry.y_center)

    def update(self, state: RobotState, dt: float) -> None:
        self._s_act = self.geometry.project_s(state.position)
        cross = abs(self._cross_track(state.position))
        speed = float(np.linalg.norm(state.velocity))
        along_err = abs(self._s_des - self._s_act)

        cruise = self._cruise_ret if self._phase == MissionPhase.RETURN else self._cruise_fwd
        track_ok = cross < 0.15 and along_err < 0.55
        advance = cruise * dt if track_ok else cruise * dt * 0.35

        if self._phase == MissionPhase.FORWARD:
            cap = min(self.geometry.total_length, self._s_act + self._lookahead)
            self._s_des = min(cap, self._s_des + advance)
            if self._s_act >= self.geometry.total_length - 0.25 and speed < 0.10 and cross < 0.2:
                self._phase = MissionPhase.RETURN
                self._s_des = self.geometry.total_length
        elif self._phase == MissionPhase.RETURN:
            cap = max(0.0, self._s_act - self._lookahead)
            self._s_des = max(cap, self._s_des - advance)
            if self._s_act <= 0.25 and speed < 0.10 and cross < 0.2:
                self._phase = MissionPhase.DONE

    def target(self, state: RobotState) -> TrajectoryTarget:
        pos_des = self.geometry.position_at_s(self._s_des)
        tangent = self.geometry.tangent_at_s(self._s_des)
        cruise = self._cruise_ret if self._phase == MissionPhase.RETURN else self._cruise_fwd
        vel_des = self._direction() * cruise * tangent
        return TrajectoryTarget(
            position=pos_des,
            velocity=vel_des,
            acceleration=np.zeros(3),
            yaw=float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0])),
        )

    def velocity_command(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        """Return (v_des_world, surface_normal) for slope-aware rolling."""
        s_ref = self._s_act
        tangent = self.geometry.tangent_at_s(s_ref)
        normal = self.geometry.normal_at_s(s_ref)
        direction = self._direction()

        s_err = self._s_des - self._s_act
        v_along = direction * (self._path_kp * s_err) - self._path_kd * float(np.dot(state.velocity, tangent))
        v_along = float(np.clip(v_along, -self._config.rolling_max_speed, self._config.rolling_max_speed))

        cross = self._cross_track(state.position)
        v_lat = -self._cross_kp * cross - self._cross_kd * state.velocity[1]

        v_world = v_along * tangent + np.array([0.0, v_lat, 0.0], dtype=float)
        speed = float(np.linalg.norm(v_world))
        if speed > self._config.rolling_max_speed:
            v_world *= self._config.rolling_max_speed / speed
        if speed < 0.02 and abs(s_err) < self._config.roll_position_tolerance:
            v_world = np.zeros(3)

        return v_world, normal

    def velocity_command_xy(self, state: RobotState) -> np.ndarray:
        """Horizontal velocity for the standard flat rolling controller."""
        v_world, _ = self.velocity_command(state)
        return v_world[:2]
