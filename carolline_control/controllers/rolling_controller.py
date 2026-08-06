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
        hold: bool = False,
    ) -> ControlCommand:
        """Map inertial velocity to omega_d and contact torque.

        When hold is False and desired velocity is zero, motors are off so the
        cage coasts under gravity and contact friction (pure physics).
        A non-zero yaw_rate still activates the controller for in-place spin
        about the ground contact normal.
        """
        _ = yaw, dt
        velocity = np.asarray(desired_velocity_xy, dtype=float).copy()
        if velocity.size >= 3:
            velocity_world = velocity[:3].astype(float)
        else:
            velocity_world = np.array([velocity[0], velocity[1], 0.0], dtype=float)
        normal = self.dynamics._unit(state.contact_normal_world, self.dynamics.E3)
        velocity_world -= normal * float(np.dot(velocity_world, normal))
        speed = float(np.linalg.norm(velocity_world))
        if speed > self._config.rolling_max_speed:
            velocity_world *= self._config.rolling_max_speed / speed
            speed = float(np.linalg.norm(velocity_world))

        active = hold or speed >= 1e-3 or abs(yaw_rate) >= 1e-3
        if not active:
            return ControlCommand(
                thrust=0.0,
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=state.rotation.copy(),
                mode=ControlMode.ROLLING,
            )

        omega_body = self.dynamics.desired_omega_from_world(state, velocity_world)
        if abs(yaw_rate) >= 1e-3:
            _, _, normal_world = self.dynamics.contact_geometry(state)
            normal_body = state.rotation.T @ normal_world
            omega_body = omega_body + float(yaw_rate) * normal_body
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
