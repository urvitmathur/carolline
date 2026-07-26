"""
Flight controller — position/velocity tracking in SE(3).

Implements Lee et al. (2010), Eq. 17-23. Vertical thrust is computed from
altitude error; horizontal motion commands tilt. Yaw stays fixed (holonomic
cardinal flight) so reversing direction uses tilt, not a 180° flip.
"""

from __future__ import annotations

import numpy as np

from carolline_control.controllers.orientation_generator import OrientationGenerator
from carolline_control.utils.so3 import body_z_world, desired_rotation_from_thrust_direction
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState, TrajectoryTarget


class FlightController:
    """Aerial position tracking controller."""

    E3 = np.array([0.0, 0.0, 1.0])
    MAX_TILT_SINE = 0.38
    MIN_B3_Z = 0.55

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._orientation = OrientationGenerator()

    def compute(
        self,
        state: RobotState,
        target: TrajectoryTarget,
        mode: ControlMode,
    ) -> ControlCommand:
        """Compute flight wrench from trajectory target."""
        ex = state.position - target.position
        ev = state.velocity - target.velocity
        mass = self._config.mass
        g = self._config.gravity
        kx = self._config.kx
        kv = self._config.kv
        kx_z = self._config.kx_z
        kv_z = self._config.kv_z

        bz = float(body_z_world(state.rotation)[2])
        if bz < 0.0:
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            Rd = self._orientation.hover_yaw(yaw)
            weight = mass * g
            thrust = float(np.clip(1.15 * weight, 0.0, 4.0 * self._config.motor_max * 0.98))
            return ControlCommand(
                thrust=thrust,
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=Rd,
                mode=mode,
            )

        acc_ff = np.asarray(target.acceleration, dtype=float)
        f_des = np.array(
            [
                -kx * ex[0] - kv * ev[0] + mass * acc_ff[0],
                -kx * ex[1] - kv * ev[1] + mass * acc_ff[1],
                -kx_z * ex[2] - kv_z * ev[2] + mass * (g + acc_ff[2]),
            ],
            dtype=float,
        )

        if ex[2] > 0.08 and ev[2] > 0.0:
            f_des[2] -= mass * 2.0 * ev[2]
        if ex[2] > 0.12:
            f_des[2] -= mass * 5.0 * (ex[2] - 0.12)

        min_center_z = self._config.min_flight_center_z
        if mode != ControlMode.LANDING and state.position[2] < min_center_z:
            f_des[2] += mass * 18.0 * (min_center_z - state.position[2])

        max_up = 4.0 * self._config.motor_max * 0.98
        max_down = 4.0 * abs(self._config.motor_min) * 0.98
        airborne = (not state.on_ground) or state.position[2] > self._config.ground_height_threshold
        min_thrust = -max_down if self._config.motor_min < 0.0 else 0.0
        if airborne and mode not in (ControlMode.LANDING,) and ex[2] <= 0.0:
            min_thrust = max(min_thrust, self._config.min_airborne_thrust_fraction * mass * g)

        if mode == ControlMode.TAKEOFF:
            weight = mass * g
            bz_takeoff = float(body_z_world(state.rotation)[2])
            alt_err = float(target.position[2] - state.position[2])
            if alt_err > 0.05 and bz_takeoff >= self._config.takeoff_cos_threshold:
                climb_frac = float(
                    np.clip(
                        (state.position[2] - self._config.cage_radius)
                        / max(self._config.hover_height - self._config.cage_radius, 0.1),
                        0.0,
                        1.0,
                    )
                )
                min_vertical = weight * (1.08 + 0.35 * climb_frac)
                f_des[2] = max(f_des[2], min_vertical)
            # Damp horizontal carry-over from rolling before switching to HOVER.
            f_des[0] -= mass * 3.0 * state.velocity[0]
            f_des[1] -= mass * 3.0 * state.velocity[1]
            if state.contact_valid and state.position[2] < self._config.takeoff_height + 0.05:
                f_des[2] = min(f_des[2], weight * 1.45)

        if mode == ControlMode.HOVER:
            if np.linalg.norm(target.velocity[:2]) < 0.05:
                f_des[0] -= 0.6 * kv * state.velocity[0]
                f_des[1] -= 0.6 * kv * state.velocity[1]

        if mode != ControlMode.LANDING:
            alt_margin = state.position[2] - min_center_z
            if alt_margin < 0.35 and np.linalg.norm(target.velocity[:2]) < 0.05:
                tilt_scale = float(np.clip(alt_margin / 0.35, 0.12, 1.0))
                f_des[0] *= tilt_scale
                f_des[1] *= tilt_scale
            if mode == ControlMode.FLIGHT and np.linalg.norm(target.velocity[:2]) < 0.05:
                hover_margin = state.position[2] - self._config.hover_height
                if hover_margin < 0.15:
                    recover = float(np.clip((hover_margin + 0.15) / 0.15, 0.05, 1.0))
                    f_des[0] *= recover
                    f_des[1] *= recover

        a_xy = f_des[:2].copy()
        vertical_force = f_des[2]

        tilt_limit = self.MAX_TILT_SINE * mass * g
        xy_norm = np.linalg.norm(a_xy)
        if xy_norm > tilt_limit:
            a_xy = a_xy * (tilt_limit / xy_norm)

        b3_xy = a_xy / (mass * g)
        b3_z = float(np.sqrt(max(1.0 - np.dot(b3_xy, b3_xy), self.MIN_B3_Z**2)))
        b3 = np.array([b3_xy[0], b3_xy[1], b3_z], dtype=float)
        b3 /= np.linalg.norm(b3)

        b_z = body_z_world(state.rotation)
        thrust = float(np.dot(f_des, b_z))
        if thrust < min_thrust and vertical_force > 0.0:
            thrust = vertical_force / max(float(b_z[2]), self.MIN_B3_Z)
        thrust = float(np.clip(thrust, min_thrust, max_up))

        yaw = target.yaw if mode != ControlMode.LANDING else float(
            np.arctan2(state.rotation[1, 0], state.rotation[0, 0])
        )
        Rd = desired_rotation_from_thrust_direction(b3, yaw)

        return ControlCommand(
            thrust=thrust,
            moment_body=np.zeros(3),
            desired_omega_body=np.zeros(3),
            desired_rotation=Rd,
            mode=mode,
        )

    def compute_velocity(
        self,
        state: RobotState,
        velocity_des: np.ndarray,
        yaw: float,
        mode: ControlMode,
    ) -> ControlCommand:
        """Velocity-only tracking for manual teleop (no position lead or height gating)."""
        v_des = np.asarray(velocity_des, dtype=float)[:3]
        ev = state.velocity - v_des
        mass = self._config.mass
        g = self._config.gravity
        kv = self._config.kv
        kv_z = self._config.kv_z

        bz = float(body_z_world(state.rotation)[2])
        if bz < 0.0:
            yaw_now = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            weight = mass * g
            return ControlCommand(
                thrust=float(np.clip(1.15 * weight, 0.0, 4.0 * self._config.motor_max * 0.98)),
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=self._orientation.hover_yaw(yaw_now),
                mode=mode,
            )

        f_des = np.array(
            [
                -kv * ev[0],
                -kv * ev[1],
                -kv_z * ev[2] + mass * g,
            ],
            dtype=float,
        )

        min_center_z = self._config.min_flight_center_z
        if state.position[2] < min_center_z:
            f_des[2] += mass * 14.0 * (min_center_z - state.position[2])

        max_up = 4.0 * self._config.motor_max * 0.98
        max_down = 4.0 * abs(self._config.motor_min) * 0.98
        min_thrust = -max_down if self._config.motor_min < 0.0 else 0.0
        airborne = (not state.on_ground) or state.position[2] > self._config.ground_height_threshold
        if airborne and mode != ControlMode.LANDING:
            min_thrust = max(min_thrust, self._config.min_airborne_thrust_fraction * mass * g)

        a_xy = f_des[:2].copy()
        vertical_force = f_des[2]
        tilt_limit = self.MAX_TILT_SINE * mass * g
        xy_norm = np.linalg.norm(a_xy)
        if xy_norm > tilt_limit:
            a_xy = a_xy * (tilt_limit / xy_norm)

        b3_xy = a_xy / (mass * g)
        b3_z = float(np.sqrt(max(1.0 - np.dot(b3_xy, b3_xy), self.MIN_B3_Z**2)))
        b3 = np.array([b3_xy[0], b3_xy[1], b3_z], dtype=float)
        b3 /= np.linalg.norm(b3)

        b_z = body_z_world(state.rotation)
        thrust = float(np.dot(f_des, b_z))
        if thrust < min_thrust and vertical_force > 0.0:
            thrust = vertical_force / max(float(b_z[2]), self.MIN_B3_Z)
        thrust = float(np.clip(thrust, min_thrust, max_up))

        Rd = desired_rotation_from_thrust_direction(b3, yaw)
        return ControlCommand(
            thrust=thrust,
            moment_body=np.zeros(3),
            desired_omega_body=np.zeros(3),
            desired_rotation=Rd,
            mode=mode,
        )

    def takeoff_thrust(self, state: RobotState, target_height: float, anchor_xy: np.ndarray, yaw: float) -> ControlCommand:
        """Ramp thrust during takeoff segment."""
        target = TrajectoryTarget(
            position=np.array([anchor_xy[0], anchor_xy[1], target_height]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=yaw,
        )
        return self.compute(state, target, ControlMode.TAKEOFF)

    def hover(
        self,
        state: RobotState,
        height: float,
        anchor_xy: np.ndarray,
        yaw: float,
    ) -> ControlCommand:
        """Regulate to fixed hover pose (locked XY anchor)."""
        target = TrajectoryTarget(
            position=np.array([anchor_xy[0], anchor_xy[1], height]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=yaw,
        )
        return self.compute(state, target, ControlMode.HOVER)
