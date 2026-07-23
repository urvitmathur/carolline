"""CAROLLINE paper rolling controller, Sec. III-A, Eqs. (12)-(18)."""

from __future__ import annotations

import numpy as np

from carolline_control.controllers.ground_dynamics import GroundAllocation, GroundDynamics
from carolline_control.utils.types import (
    ControlCommand,
    ControllerConfig,
    ControlMode,
    RobotState,
)


class RollingController:
    """Attitude-invariant inertial velocity tracking at the cage contact."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self.dynamics = GroundDynamics(config)
        self.last_allocation: GroundAllocation | None = None

    def reset(self) -> None:
        self.last_allocation = None

    def compute(
        self,
        state: RobotState,
        desired_velocity_xy: np.ndarray,
        yaw: float,
        dt: float,
        yaw_rate: float = 0.0,
    ) -> ControlCommand:
        """Map inertial velocity to omega_d and contact torque."""
        _ = yaw, yaw_rate, dt
        velocity = np.asarray(desired_velocity_xy, dtype=float).copy()
        speed = float(np.linalg.norm(velocity))
        if speed > self._config.rolling_max_speed:
            velocity *= self._config.rolling_max_speed / speed

        omega_body = self.dynamics.desired_omega(state, velocity)
        torque_body = self.dynamics.tracking_torque(
            state,
            omega_body,
            omega_kp=self._config.rolling_omega_kp,
        )

        return ControlCommand(
            thrust=0.0,
            moment_body=torque_body,
            desired_omega_body=omega_body,
            desired_rotation=state.rotation.copy(),
            mode=ControlMode.ROLLING,
        )

    def allocate(self, state: RobotState, command: ControlCommand) -> GroundAllocation:
        self.last_allocation = self.dynamics.allocate(state, command.moment_body)
        return self.last_allocation

    @staticmethod
    def planar_velocity_from_state(state: RobotState) -> np.ndarray:
        """Estimate current planar velocity from rigid-body kinematics."""
        return state.velocity[:2]
