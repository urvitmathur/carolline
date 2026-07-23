"""Utility modules for CAROLLINE control."""

from carolline_control.utils.so3 import (
    attitude_error,
    attitude_error_function,
    body_z_world,
    desired_rotation_from_thrust_direction,
    hat,
    omega_error,
    quat_to_rot,
    rot_from_two_vectors,
    rot_log,
    rot_to_quat,
    vee,
)
from carolline_control.utils.types import (
    ControlCommand,
    ControlMode,
    ControllerConfig,
    MotorCommand,
    RobotState,
    TrajectoryTarget,
)

__all__ = [
    "ControlCommand",
    "ControlMode",
    "ControllerConfig",
    "MotorCommand",
    "RobotState",
    "TrajectoryTarget",
    "attitude_error",
    "attitude_error_function",
    "body_z_world",
    "desired_rotation_from_thrust_direction",
    "hat",
    "omega_error",
    "quat_to_rot",
    "rot_from_two_vectors",
    "rot_log",
    "rot_to_quat",
    "vee",
]
