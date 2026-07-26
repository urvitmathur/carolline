"""Equation-level tests for the paper-aligned CAROLLINE ground controller."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.ground_dynamics import GroundDynamics
from carolline_control.controllers.planner import Planner
from carolline_control.controllers.pre_takeoff_controller import PreTakeoffController
from carolline_control.utils.so3 import hat, rot_log
from carolline_control.utils.types import RobotState


def _config():
    config = load_config("carolline_control/config.yaml")
    config.mass = 2.125
    config.inertia = np.diag([0.1037, 0.0795, 0.0684])
    return config


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    skew = hat(axis)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _state(
    rotation: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> RobotState:
    rotation = np.eye(3) if rotation is None else rotation
    normal = np.array([0.0, 0.0, 1.0]) if normal is None else normal
    normal = normal / np.linalg.norm(normal)
    position = np.array([0.2, -0.1, 0.8])
    point = position - 0.4 * normal
    return RobotState(
        position=position,
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        rotation=rotation,
        omega_body=np.zeros(3),
        omega_world=np.zeros(3),
        accel_body=np.zeros(3),
        on_ground=True,
        ground_contact_z=position[2],
        time=0.0,
        contact_point_world=point,
        contact_normal_world=normal,
        contact_force=20.0,
        contact_confidence=1.0,
        contact_valid=True,
    )


def test_eq14_reconstructs_flat_ground_velocity() -> None:
    dynamics = GroundDynamics(_config())
    state = _state(_rotation(np.array([1.0, 2.0, -0.5]), 0.9))
    desired = np.array([0.7, -0.35])

    omega = dynamics.desired_omega(state, desired)
    jacobian, tangent, _ = dynamics.rolling_jacobian(state)

    assert np.allclose(jacobian @ omega, tangent.T @ [*desired, 0.0], atol=2e-5)


def test_eq6_parallel_axis_contact_inertia() -> None:
    config = _config()
    dynamics = GroundDynamics(config)
    state = _state()
    q = np.array([0.0, 0.0, config.cage_radius])
    expected = config.inertia + config.mass * (
        np.dot(q, q) * np.eye(3) - np.outer(q, q)
    )

    assert np.allclose(dynamics.contact_inertia(state), expected)


def test_ramp_hold_torque_cancels_gravity_about_contact() -> None:
    config = _config()
    dynamics = GroundDynamics(config)
    normal = np.array([-0.5, 0.0, np.sqrt(3.0) * 0.5])
    rotation = _rotation(np.array([0.2, 0.8, -0.3]), 0.9)
    state = _state(rotation, normal)

    motor_torque_body = dynamics.tracking_torque(state, np.zeros(3))
    q_world = normal * config.cage_radius
    gravity_world = np.array([0.0, 0.0, -config.mass * config.gravity])

    assert np.allclose(
        rotation @ motor_torque_body + np.cross(q_world, gravity_world),
        np.zeros(3),
        atol=1e-9,
    )


def test_ground_gyroscopic_term_uses_measured_not_desired_rate() -> None:
    config = _config()
    dynamics = GroundDynamics(config)
    state = _state()
    desired = np.array([2.0, -3.0, 0.4])

    torque = dynamics.tracking_torque(state, desired)
    expected = dynamics.contact_inertia(state) @ (
        config.ground_omega_kp * desired
    )

    assert np.allclose(torque, expected)


def test_eq18_qf_matches_reachable_torque() -> None:
    dynamics = GroundDynamics(_config())
    state = _state(_rotation(np.array([0.3, 1.0, 0.2]), 0.7))
    q_matrix = dynamics.thrust_to_contact_torque(state)
    torque = q_matrix @ np.array([3.0, -2.0, 1.5, -1.0])

    allocation = dynamics.allocate(state, torque)

    assert not allocation.saturated
    assert np.allclose(q_matrix @ allocation.motor.thrusts, torque, atol=2e-4)


def test_eq18_uniform_scaling_respects_bidirectional_bounds() -> None:
    config = _config()
    dynamics = GroundDynamics(config)
    state = _state()
    requested = np.array([50.0, -40.0, 5.0])

    allocation = dynamics.allocate(state, requested)

    assert allocation.saturated
    assert allocation.allocation_scale > 1.0
    assert np.max(allocation.motor.thrusts) <= config.motor_max + 1e-9
    assert np.min(allocation.motor.thrusts) >= config.motor_min - 1e-9
    assert np.allclose(
        allocation.achieved_torque,
        requested / allocation.allocation_scale,
        atol=2e-3,
    )


def test_ground_allocation_limits_contact_unloading_without_losing_torque() -> None:
    config = _config()
    dynamics = GroundDynamics(config)
    state = _state(_rotation(np.array([0.3, 1.0, -0.2]), 0.8))
    requested = np.array([2.0, -1.5, 0.03])

    allocation = dynamics.allocate(state, requested)
    normal_component = float(
        np.dot(state.rotation[:, 2], state.contact_normal_world)
        * np.sum(allocation.motor.thrusts)
    )
    max_unload = (
        config.ground_max_unload_fraction * config.mass * config.gravity
    )

    assert normal_component <= max_unload + 1e-8
    assert np.allclose(allocation.achieved_torque, requested, atol=2e-3)


def test_so3_log_at_90_and_180_degrees() -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    for angle in (0.5 * np.pi, np.pi):
        result = rot_log(_rotation(axis, angle))
        assert np.isclose(np.linalg.norm(result), angle, atol=1e-7)
        assert np.isclose(abs(float(np.dot(result / angle, axis))), 1.0, atol=1e-7)


def test_ramp_contact_frame_is_orthonormal_and_reconstructs_tangent_velocity() -> None:
    dynamics = GroundDynamics(_config())
    normal = np.array([-np.sin(np.radians(12.0)), 0.0, np.cos(np.radians(12.0))])
    state = _state(_rotation(np.array([0.2, 0.7, 1.0]), 1.1), normal)
    desired = np.array([0.6, 0.15])

    omega = dynamics.desired_omega(state, desired)
    jacobian, tangent, _ = dynamics.rolling_jacobian(state)
    velocity_world = np.array([*desired, 0.0])
    velocity_world -= normal * np.dot(normal, velocity_world)

    assert np.allclose(tangent.T @ tangent, np.eye(2), atol=1e-9)
    assert np.allclose(tangent.T @ normal, 0.0, atol=1e-9)
    assert np.allclose(jacobian @ omega, tangent.T @ velocity_world, atol=2e-5)


def test_rolling_planner_does_not_double_damp_velocity_command() -> None:
    config = _config()
    config.rolling_max_speed = 2.5
    config.roll_target = np.array([3.0, 0.0])
    state = _state()
    state.position[:2] = 0.0
    state.velocity[:2] = np.array([2.0, 0.0])

    command = Planner(config).rolling_velocity(state)

    assert np.allclose(command, np.array([2.5, 0.0]))


def test_rolling_planner_uses_configured_braking_distance() -> None:
    config = _config()
    config.rolling_max_speed = 3.5
    config.rolling_braking_distance = 2.2
    config.roll_position_tolerance = 0.2
    config.roll_target = np.array([1.2, 0.0])
    state = _state()
    state.position[:2] = 0.0

    command = Planner(config).rolling_velocity(state)

    assert np.isclose(np.linalg.norm(command), 1.75)


def test_pre_takeoff_uses_closest_upright_attitude_without_body_z_spin() -> None:
    config = _config()
    state = _state(_rotation(np.array([0.4, -0.7, 0.2]), 1.3))
    controller = PreTakeoffController(config)

    command = controller.compute(state)

    assert np.allclose(command.desired_rotation[:, 2], np.array([0.0, 0.0, 1.0]))
    assert abs(command.desired_omega_body[2]) < 1e-8


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
