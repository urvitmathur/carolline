"""Unit tests for LQR gain synthesis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.lqr.dare_solver import check_closed_loop_stable, discretize_euler, solve_dare
from carolline_control.controllers.lqr.flight_lqr import FlightLqrController, build_hover_linearization
from carolline_control.controllers.lqr.rolling_lqr import RollingLqrController
from carolline_control.utils.types import RobotState


def _config():
    config = load_config("carolline_control/config.yaml")
    config.mass = 2.125
    config.inertia = np.diag([0.1037, 0.0795, 0.0684])
    return config


def test_dare_produces_stable_flight_gains() -> None:
    config = _config()
    Ac, Bc = build_hover_linearization(config.mass, config.gravity, config.inertia)
    Ad, Bd = discretize_euler(Ac, Bc, 0.004)
    Q = np.eye(12)
    R = np.eye(4) * 0.1
    K, P = solve_dare(Ad, Bd, Q, R)
    assert K.shape == (4, 12)
    assert P.shape == (12, 12)
    assert check_closed_loop_stable(Ad, Bd, K)


def test_flight_lqr_controller_builds() -> None:
    config = _config()
    ctrl = FlightLqrController(config)
    assert ctrl.gain_matrix.shape == (4, 12)


def test_rolling_lqr_builds_on_nominal_state() -> None:
    config = _config()
    ctrl = RollingLqrController(config)
    rotation = np.eye(3)
    state = RobotState(
        position=np.array([0.0, 0.0, config.cage_radius]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        rotation=rotation,
        omega_body=np.zeros(3),
        omega_world=np.zeros(3),
        accel_body=np.zeros(3),
        on_ground=True,
        ground_contact_z=config.cage_radius,
        time=0.0,
        contact_point_world=np.array([0.0, 0.0, 0.0]),
        contact_normal_world=np.array([0.0, 0.0, 1.0]),
        contact_force=20.0,
        contact_confidence=1.0,
        contact_valid=True,
    )
    cmd = ctrl.compute(state, np.array([1.0, 0.0]))
    assert cmd.mode.name == "ROLLING"
    assert ctrl.gain_matrix.shape == (3, 5)


if __name__ == "__main__":
    test_dare_produces_stable_flight_gains()
    print("PASS test_dare_produces_stable_flight_gains")
    test_flight_lqr_controller_builds()
    print("PASS test_flight_lqr_controller_builds")
    test_rolling_lqr_builds_on_nominal_state()
    print("PASS test_rolling_lqr_builds_on_nominal_state")
