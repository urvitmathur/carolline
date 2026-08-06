"""
Desired orientation generator for flight and rolling modes.

CAROLLINE paper, Sec. III-D:
- Flight mode: Rd from desired thrust direction and yaw (Lee et al., Eq. 22-23)
- Rolling mode: Rd aligns body so thrust assists rolling without lift-off
"""

from __future__ import annotations

import numpy as np

from carolline_control.utils.so3 import clamp_thrust_direction, desired_rotation_from_thrust_direction
from carolline_control.utils.types import ControlMode, RobotState, TrajectoryTarget


class OrientationGenerator:
    """Construct desired rotation matrices Rd for each operating mode."""

    E3 = np.array([0.0, 0.0, 1.0])

    def flight(
        self,
        state: RobotState,
        target: TrajectoryTarget,
        mass: float,
        gravity: float,
        kx: float,
        kv: float,
    ) -> np.ndarray:
        """Flight-mode Rd (Lee et al., Eq. 22-23).

        b3c = -normalize(kx*ex + kv*ev - m*g*e3 + m*xdd)
        """
        ex = state.position - target.position
        ev = state.velocity - target.velocity
        a_cmd = -kx * ex - kv * ev + mass * gravity * self.E3 + mass * target.acceleration
        norm = np.linalg.norm(a_cmd)
        if norm < 1e-6:
            b3 = self.E3
        else:
            b3 = clamp_thrust_direction(a_cmd / norm)
        return desired_rotation_from_thrust_direction(b3, target.yaw)

    def rolling(
        self,
        state: RobotState,
        desired_velocity_xy: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """Rolling-mode Rd (CAROLLINE paper, Sec. III-C).

        The thrust axis stays near horizontal while the cage rolls.
        b3_des is perpendicular to desired planar velocity and world up.
        """
        v_des = np.asarray(desired_velocity_xy, dtype=float).reshape(-1)
        if v_des.size >= 3:
            v_des = np.array([v_des[0], v_des[1], 0.0], dtype=float)
        else:
            v_des = np.array([v_des[0], v_des[1], 0.0], dtype=float)
        speed = np.linalg.norm(v_des)
        if speed < 1e-3:
            return state.rotation.copy()

        roll_axis = np.cross(self.E3, v_des / speed)
        b3 = np.cross(roll_axis, self.E3)
        b3 = b3 / np.linalg.norm(b3)
        return desired_rotation_from_thrust_direction(b3, yaw)

    def pre_takeoff(self, state: RobotState) -> np.ndarray:
        """Pre-takeoff Rd: fixed upright target preserving current yaw."""
        yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
        return desired_rotation_from_thrust_direction(self.E3, yaw)

    def hover_yaw(self, yaw: float) -> np.ndarray:
        """Fixed hover orientation."""
        return desired_rotation_from_thrust_direction(self.E3, yaw)

    def for_mode(
        self,
        mode: ControlMode,
        state: RobotState,
        target: TrajectoryTarget,
        mass: float,
        gravity: float,
        kx: float,
        kv: float,
        rolling_velocity_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dispatch Rd generation by mode."""
        if mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
            return self.pre_takeoff(state)
        if mode == ControlMode.ROLLING:
            return self.rolling(state, rolling_velocity_xy or np.zeros(2), target.yaw)
        return self.flight(state, target, mass, gravity, kx, kv)
