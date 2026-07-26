"""
Pre-takeoff controller (CAROLLINE paper, Sec. III-B).

The robot starts in an arbitrary orientation on the cage. This controller:
1. Detects orientation via R (no Euler angles)
2. Computes SO(3) error to upright pre-takeoff pose
3. Commands angular velocity to roll on the cage using contact
4. Applies low collective thrust to assist righting without liftoff
"""

from __future__ import annotations

import numpy as np

from carolline_control.controllers.ground_dynamics import GroundAllocation, GroundDynamics
from carolline_control.utils.so3 import body_z_world, rot_from_two_vectors, rot_log
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState


class PreTakeoffController:
    """Ground recovery controller before takeoff."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self.dynamics = GroundDynamics(config)
        self.last_allocation: GroundAllocation | None = None
        self._desired_rotation: np.ndarray | None = None

    def reset(self, state: RobotState) -> None:
        """Latch the closest upright attitude without requesting body-z spin."""
        align_up = rot_from_two_vectors(
            body_z_world(state.rotation),
            np.array([0.0, 0.0, 1.0]),
        )
        self._desired_rotation = align_up @ state.rotation
        self.last_allocation = None

    def compute(self, state: RobotState, mode: ControlMode = ControlMode.PRETAKEOFF) -> ControlCommand:
        """Generate paper Eq. (19) angular rate and shared Eq. (15)-(18) torque."""
        if self._desired_rotation is None:
            self.reset(state)
        assert self._desired_rotation is not None
        Rd = self._desired_rotation
        omega_des = self._config.pre_takeoff_omega_gain * rot_log(
            state.rotation.T @ Rd
        )
        omega_des -= 0.85 * np.asarray(state.omega_body, dtype=float)
        omega_norm = float(np.linalg.norm(omega_des))
        if omega_norm > self._config.pre_takeoff_omega_limit:
            omega_des *= self._config.pre_takeoff_omega_limit / omega_norm
        torque = self.dynamics.tracking_torque(
            state,
            omega_des,
            omega_kp=self._config.pre_takeoff_omega_kp,
        )

        return ControlCommand(
            thrust=0.0,
            moment_body=torque,
            desired_omega_body=omega_des,
            desired_rotation=Rd,
            mode=mode,
        )

    def allocate(self, state: RobotState, command: ControlCommand) -> GroundAllocation:
        self.last_allocation = self.dynamics.allocate(state, command.moment_body)
        return self.last_allocation

    @staticmethod
    def is_upright(state: RobotState, cos_threshold: float) -> bool:
        """Check if body z-axis is aligned with world up."""
        return float(body_z_world(state.rotation)[2]) >= cos_threshold
