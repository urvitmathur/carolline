"""Hybrid rolling: flat RollingController with slope thrust scaling only."""

from __future__ import annotations

import numpy as np

from carolline_control.controllers.rolling_controller import RollingController
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlCommand, ControlMode, RobotState


class HybridRampRollingController(RollingController):
    """Proven flat rolling + reduced thrust on inclines to limit hop/fly-off."""

    def compute(
        self,
        state: RobotState,
        desired_velocity_xy: np.ndarray,
        yaw: float,
        dt: float,
        yaw_rate: float = 0.0,
        *,
        slope_normal: np.ndarray | None = None,
    ) -> ControlCommand:
        cmd = super().compute(state, desired_velocity_xy, yaw, dt, yaw_rate)
        b3_z = max(float(body_z_world(state.rotation)[2]), 0.05)
        tilt_deg = float(np.degrees(np.arccos(np.clip(b3_z, -1.0, 1.0))))

        if slope_normal is not None and float(slope_normal[2]) < 0.995:
            cmd.thrust *= 0.62
        if tilt_deg > 115.0:
            cmd.moment_body *= 0.25
            cmd.thrust *= 0.5

        return ControlCommand(
            thrust=cmd.thrust,
            moment_body=cmd.moment_body,
            desired_omega_body=cmd.desired_omega_body,
            desired_rotation=cmd.desired_rotation,
            mode=ControlMode.ROLLING,
        )
