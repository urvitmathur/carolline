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

        local = np.linspace(max(0, s - 0.8), min(self.total_length, s + 0.8), 80)
        pts = np.array([self.position_at_s(v) for v in local])
        idx = int(np.argmin(np.linalg.norm(pts - position, axis=1)))
        return float(local[idx])


class RampPathPlanner:
    """Arc-length follower with tight carrot tracking and smooth velocity commands."""

    def __init__(
        self,
        geometry: RampGeometry,
        config: ControllerConfig,
        *,
        lookahead: float = 0.35,
        max_carrot: float = 0.14,
        home_s: float = 0.0,
        path_kp: float = 0.55,
        path_kd: float = 0.85,
        cross_kp: float = 0.9,
        cross_kd: float = 0.6,
        velocity_filter_tau: float = 0.15,
        done_settle_time: float = 0.65,
        home_arrival_speed: float = 0.07,
        end_approach_length: float = 0.85,
    ) -> None:
        self.geometry = geometry
        self._config = config
        self._lookahead = lookahead
        self._max_carrot = max_carrot
        self._home_s = home_s
        self._path_kp = path_kp
        self._path_kd = path_kd
        self._cross_kp = cross_kp
        self._cross_kd = cross_kd
        self._velocity_filter_tau = max(0.02, velocity_filter_tau)
        self._done_settle_time = done_settle_time
        self._home_arrival_speed = home_arrival_speed
        self._end_approach_length = end_approach_length
        self._s_des = home_s
        self._s_act = home_s
        self._s_mission = home_s
        self._phase = MissionPhase.FORWARD
        self._done_timer = 0.0
        self._last_dt = 0.004
        self._v_filt = np.zeros(3, dtype=float)
        self._cruise_fwd = config.rolling_max_speed * 0.82
        self._cruise_ret = config.rolling_max_speed * 0.62

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def s_desired(self) -> float:
        return self._s_des

    @property
    def s_actual(self) -> float:
        return self._s_act

    @property
    def hold_requested(self) -> bool:
        return self._phase == MissionPhase.RETURN and self._done_timer >= 0.20 * self._done_settle_time

    def reset(self) -> None:
        self._s_des = self._home_s
        self._s_act = self._home_s
        self._s_mission = self._home_s
        self._phase = MissionPhase.FORWARD
        self._done_timer = 0.0
        self._v_filt[:] = 0.0

    def _cross_track(self, position: np.ndarray) -> float:
        return float(position[1] - self.geometry.y_center)

    def _sync_carrot(self) -> None:
        """Keep the path carrot close to actual pose to avoid large position lag."""
        if self._phase == MissionPhase.FORWARD:
            lead = min(self._max_carrot, max(0.06, self._s_mission - self._s_act))
            self._s_des = min(self._s_mission, self._s_act + lead)
        elif self._phase == MissionPhase.RETURN:
            lead = min(self._max_carrot, max(0.06, self._s_act - self._s_mission))
            self._s_des = max(self._home_s, max(self._s_mission, self._s_act - lead))
        else:
            self._s_des = self._home_s

    def update(self, state: RobotState, dt: float) -> None:
        self._last_dt = max(dt, 1e-6)
        self._s_act = max(self._home_s, self.geometry.project_s(state.position))
        cross = abs(self._cross_track(state.position))
        speed = float(np.linalg.norm(state.velocity))
        cruise = self._cruise_ret if self._phase == MissionPhase.RETURN else self._cruise_fwd
        home_x = float(self.geometry.position_at_s(self._home_s)[0])
        x = float(state.position[0])

        if self._phase == MissionPhase.FORWARD:
            cap = min(self.geometry.total_length, self._s_act + self._lookahead)
            self._s_mission = min(cap, self._s_mission + cruise * dt)
            if self._s_act >= self.geometry.total_length - 0.35 and speed < 0.14 and cross < 0.25:
                self._phase = MissionPhase.RETURN
                self._s_mission = self._s_act
                self._done_timer = 0.0
        elif self._phase == MissionPhase.RETURN:
            remaining = max(0.0, self._s_act - self._home_s)
            approach = 1.0 if remaining >= self._end_approach_length else max(
                0.55, remaining / max(self._end_approach_length, 1e-6)
            )
            cap = max(self._home_s, self._s_act - self._lookahead)
            self._s_mission = max(self._home_s, min(cap, self._s_mission - cruise * approach * dt))

            dx = x - home_x
            at_marker = (
                -0.015 <= dx <= 0.04
                and abs(self._cross_track(state.position)) < 0.10
            )
            near_home = at_marker and speed < self._home_arrival_speed
            if near_home:
                self._done_timer += dt
            else:
                self._done_timer = 0.0
            if self._done_timer >= self._done_settle_time:
                self._phase = MissionPhase.DONE
                self._s_mission = self._home_s
                self._v_filt[:] = 0.0
        else:
            self._s_mission = self._home_s
            self._v_filt[:] = 0.0

        self._sync_carrot()

    def target(self, state: RobotState) -> TrajectoryTarget:
        pos_des = self.geometry.position_at_s(self._s_des)
        tangent = self.geometry.tangent_at_s(self._s_des)
        if self._phase == MissionPhase.DONE:
            vel_des = np.zeros(3)
        else:
            s_err = self._s_des - self._s_act
            sign = float(np.sign(s_err)) if abs(s_err) > 1e-4 else 0.0
            cruise = self._cruise_ret if self._phase == MissionPhase.RETURN else self._cruise_fwd
            vel_des = sign * cruise * tangent
        return TrajectoryTarget(
            position=pos_des,
            velocity=vel_des,
            acceleration=np.zeros(3),
            yaw=float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0])),
        )

    def _filter_velocity(self, raw: np.ndarray) -> np.ndarray:
        alpha = min(1.0, self._last_dt / self._velocity_filter_tau)
        self._v_filt = (1.0 - alpha) * self._v_filt + alpha * raw
        if self._phase == MissionPhase.DONE:
            self._v_filt[:] = 0.0
        return self._v_filt.copy()

    def velocity_command(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        """Return (v_des_world, surface_normal) for slope-aware rolling."""
        if self._phase == MissionPhase.DONE:
            return np.zeros(3), self.geometry.normal_at_s(self._home_s)

        s_ref = self._s_act
        tangent = self.geometry.tangent_at_s(s_ref)
        normal = self.geometry.normal_at_s(s_ref)
        path_pos = self.geometry.position_at_s(s_ref)
        s_err = self._s_des - self._s_act
        cruise = self._cruise_ret if self._phase == MissionPhase.RETURN else self._cruise_fwd
        sign = float(np.sign(s_err)) if abs(s_err) > 1e-4 else 0.0
        v_tangent = float(np.dot(state.velocity, tangent))

        if abs(s_err) < self._config.roll_position_tolerance and self.hold_requested:
            v_along = -self._path_kd * 0.35 * v_tangent
        elif abs(s_err) < self._config.roll_position_tolerance:
            v_along = 0.0
        else:
            taper = min(1.0, abs(s_err) / max(self._max_carrot, 0.08))
            v_ff = sign * cruise * max(0.55, taper)
            v_fb = self._path_kp * s_err - self._path_kd * v_tangent
            v_along = float(v_ff + 0.55 * v_fb)

        if self._phase == MissionPhase.RETURN:
            home_x = float(self.geometry.position_at_s(self._home_s)[0])
            dx = float(state.position[0]) - home_x

            if dx < -0.005:
                v_along = max(v_along, min(0.06, -0.55 * dx))
            elif dx <= 0.03:
                # On the green spot — brake only, no backward command.
                v_along = max(v_along, -self._path_kd * 0.35 * v_tangent)
                if dx <= 0.015:
                    v_along = max(v_along, 0.0)
            elif dx <= 0.20:
                # Final metres: gentle proportional approach, capped crawl.
                v_along = min(v_along, -min(0.07, 0.40 * dx))
            else:
                home_pos = self.geometry.position_at_s(self._home_s)
                to_home = home_pos - state.position
                v_home = float(np.dot(to_home, tangent))
                v_along = min(v_along, v_home * 1.05, -0.08)

        if self._phase == MissionPhase.DONE:
            v_along = 0.0

        if self.geometry.segment_kind(s_ref) == "up" and sign > 0.0:
            v_along = max(v_along, 0.16)
        elif self.geometry.segment_kind(s_ref) == "down" and sign < 0.0:
            v_along = min(v_along, -0.14)

        v_along = float(np.clip(v_along, -self._config.rolling_max_speed, self._config.rolling_max_speed))

        cross = self._cross_track(state.position)
        v_lat = -self._cross_kp * cross - self._cross_kd * state.velocity[1]
        v_world = v_along * tangent + np.array([0.0, v_lat, 0.0], dtype=float)
        speed = float(np.linalg.norm(v_world))
        if speed > self._config.rolling_max_speed:
            v_world *= self._config.rolling_max_speed / speed

        if state.position[2] > path_pos[2] + 0.22 or not state.on_ground:
            v_world *= 0.15

        soft = min(1.0, max(0.0, (state.time - 0.4) / 1.6))
        v_world *= soft

        return self._filter_velocity(v_world), normal

    def velocity_command_xy(self, state: RobotState) -> np.ndarray:
        """Tangent-plane velocity (3D on slopes, XY on flat ground)."""
        v_world, _ = self.velocity_command(state)
        return v_world
