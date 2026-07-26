"""Shared data types for the CAROLLINE control stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

Vector3 = np.ndarray
Matrix3 = np.ndarray
Quaternion = np.ndarray


class ControlMode(Enum):
    """Finite-state machine modes (CAROLLINE paper, Sec. III)."""

    PRETAKEOFF = auto()
    UPRIGHT = auto()
    TAKEOFF = auto()
    HOVER = auto()
    FLIGHT = auto()
    ROLLING = auto()
    LANDING = auto()
    IDLE = auto()


@dataclass
class RobotState:
    """Estimated rigid-body state in the world frame."""

    position: Vector3
    velocity: Vector3
    quaternion: Quaternion
    rotation: Matrix3
    omega_body: Vector3
    omega_world: Vector3
    accel_body: Vector3
    on_ground: bool
    ground_contact_z: float
    time: float
    contact_point_world: Vector3 = field(default_factory=lambda: np.zeros(3))
    contact_normal_world: Vector3 = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    contact_force: float = 0.0
    contact_confidence: float = 0.0
    contact_valid: bool = False


@dataclass
class ControlCommand:
    """High-level wrench command before motor mixing."""

    thrust: float
    moment_body: Vector3
    desired_omega_body: Vector3
    desired_rotation: Matrix3
    mode: ControlMode


@dataclass
class MotorCommand:
    """Per-rotor thrust commands sent to MuJoCo actuators."""

    thrusts: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class TrajectoryTarget:
    """Planner output for navigation."""

    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    yaw: float
    yaw_rate: float = 0.0


@dataclass
class ControlDiagnostics:
    """Per-step control pipeline metrics for logging and analysis."""

    position_error: Vector3
    velocity_error: Vector3
    euler_rpy: Vector3
    des_euler_rpy: Vector3
    e_R: Vector3
    omega_d_body: Vector3
    moment_body: Vector3
    motor_spread: float
    motor_saturated: bool
    body_z_up: float
    tilt_deg: float
    achievable_contact_torque: Vector3 = field(default_factory=lambda: np.zeros(3))
    allocation_scale: float = 1.0
    contact_normal_world: Vector3 = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    ground_allocation_active: bool = False


@dataclass
class ControllerConfig:
    """Runtime configuration loaded from YAML."""

    model_path: str
    cage_radius: float
    mass: float
    gravity: float
    inertia: Matrix3
    motor_min: float
    motor_max: float
    kx: float
    kv: float
    kx_z: float
    kv_z: float
    kR: float
    kOmega: float
    kR_pre: float
    kOmega_pre: float
    k_roll: float
    k_yaw: float
    upright_cos_threshold: float
    takeoff_cos_threshold: float
    upright_settle_time: float
    upright_omega_tolerance: float
    takeoff_height: float
    hover_height: float
    landing_height: float
    ground_height_threshold: float
    hover_altitude_tolerance: float
    hover_velocity_tolerance: float
    takeoff_horizontal_velocity_tolerance: float
    min_airborne_thrust_fraction: float
    landing_settle_time: float
    pre_takeoff_thrust_fraction: float
    rolling_max_speed: float
    rolling_kp: float
    rolling_kd: float
    motor_slew_rate: float
    roll_target: np.ndarray
    roll_position_tolerance: float
    roll_arrival_speed: float
    spawn_xy: np.ndarray
    leg_distance: float
    segment_duration: float
    yaw_drag_coeff: np.ndarray
    rotor_positions: np.ndarray
    ground_omega_weights: np.ndarray
    ground_omega_kp: np.ndarray
    rolling_omega_kp: np.ndarray
    rolling_braking_distance: float
    ground_pseudoinverse_damping: float
    pre_takeoff_omega_gain: float
    pre_takeoff_omega_limit: float
    pre_takeoff_omega_kp: np.ndarray
    pre_upright_settle_time: float
    contact_min_normal_z: float
    contact_force_threshold: float
    contact_loss_grace: float
    rolling_stall_speed: float
    rolling_stall_time: float
    ground_max_unload_fraction: float
    initial_mode: ControlMode = ControlMode.PRETAKEOFF
    waypoints: Optional[list[list[float]]] = None
    mission_yaw: float = 0.0
    min_flight_center_z: float = 0.72
    waypoint_reach_tolerance: float = 0.15
    waypoint_dwell_time: float = 3.0
    hover_before_flight_time: float = 5.0
    esc_mapping_enabled: bool = True
    esc_thrust_forward_max: float = 13.0
    esc_thrust_reverse_max: float = 10.0
    esc_reverse_efficiency: float = 0.72
