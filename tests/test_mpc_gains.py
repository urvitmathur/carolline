"""MPC synthesis sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.mpc.flight_mpc import FlightMpcController
from carolline_control.controllers.mpc.finite_horizon import solve_linear_mpc, discretize_euler
from carolline_control.controllers.lqr.flight_lqr import build_hover_linearization
from carolline_control.controllers.mpc.rolling_mpc import RollingMpcController
from carolline_control.utils.types import RobotState


def _config():
    config = load_config("carolline_control/config.yaml")
    config.mass = 2.125
    config.inertia = np.diag([0.1037, 0.0795, 0.0684])
    return config


def test_mpc_solver_returns_bounded_control() -> None:
    config = _config()
    Ac, Bc = build_hover_linearization(config.mass, config.gravity, config.inertia)
    Ad, Bd = discretize_euler(Ac, Bc, 0.004)
    Q = np.eye(12)
    R = np.eye(4) * 0.1
    Qf = Q * 2
    u0 = solve_linear_mpc(Ad, Bd, Q, R, Qf, 10, np.ones(12) * 0.01, u_min=-np.ones(4) * 5, u_max=np.ones(4) * 5)
    assert u0.shape == (4,)
    assert np.all(u0 >= -5.0) and np.all(u0 <= 5.0)


def test_flight_mpc_builds() -> None:
    ctrl = FlightMpcController(_config())
    assert ctrl._weights.horizon >= 5


def test_rolling_mpc_builds() -> None:
    config = _config()
    ctrl = RollingMpcController(config)
    state = RobotState(
        position=np.array([0.0, 0.0, config.cage_radius]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        rotation=np.eye(3),
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


if __name__ == "__main__":
    test_mpc_solver_returns_bounded_control()
    print("PASS test_mpc_solver_returns_bounded_control")
    test_flight_mpc_builds()
    print("PASS test_flight_mpc_builds")
    test_rolling_mpc_builds()
    print("PASS test_rolling_mpc_builds")
