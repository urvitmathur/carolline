"""Unit tests for MPC gain construction and basic regulation.

Run: python carolline_control/tests/test_mpc_gains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.mpc.config_loader import flight_mpc_weights_from_config, load_mpc_config, rolling_mpc_weights_from_config
from carolline_control.controllers.mpc.finite_horizon import LinearMpcSolver, discretize_euler
from carolline_control.controllers.mpc.flight_mpc import FlightMpcController
from carolline_control.controllers.mpc.rolling_mpc import RollingMpcController
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.types import ControlMode, TrajectoryTarget
from carolline_control.visualization.markers import compile_model_with_markers


def _filled_config():
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    model = compile_model_with_markers(REPO_ROOT / config.model_path, config)
    est = StateEstimator(model, config)
    est.fill_inertial_params(config)
    return config


def test_linear_mpc_solver_returns_bounded_control():
    Ac = np.zeros((2, 2))
    Ac[0, 1] = 1.0
    Bc = np.array([[0.0], [1.0]])
    Ad, Bd = discretize_euler(Ac, Bc, 0.01)
    Q = np.diag([10.0, 1.0])
    R = np.diag([0.1])
    Qf = 2.0 * Q
    solver = LinearMpcSolver(Ad, Bd, Q, R, Qf, horizon=20, u_min=np.array([-1.0]), u_max=np.array([1.0]))
    u = solver.solve(np.array([1.0, 0.0]))
    assert u.shape == (1,)
    assert np.all(np.isfinite(u))
    assert -1.0 <= u[0] <= 1.0


def test_flight_mpc_produces_pitch_moment_for_x_error():
    config = _filled_config()
    mpc = FlightMpcController(config, flight_mpc_weights_from_config(load_mpc_config()))
    data = mujoco.MjData(compile_model_with_markers(REPO_ROOT / config.model_path, config))
    data.qpos[:7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data)
    est = StateEstimator(data.model, config)
    state = est.estimate(data)
    target = TrajectoryTarget(
        position=np.array([0.5, 0.0, 1.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        yaw=0.0,
    )
    cmd = mpc.compute(state, target, ControlMode.HOVER)
    assert np.isfinite(cmd.thrust)
    assert abs(cmd.moment_body[1]) > 0.05


def test_rolling_mpc_builds_and_allocates():
    config = _filled_config()
    mpc = RollingMpcController(config, rolling_mpc_weights_from_config(load_mpc_config()))
    data = mujoco.MjData(compile_model_with_markers(REPO_ROOT / config.model_path, config))
    data.qpos[:7] = [0.0, 0.0, 0.4, 0.924, 0.0, 0.383, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data)
    for _ in range(100):
        mujoco.mj_step(data.model, data)
    est = StateEstimator(data.model, config)
    state = est.estimate(data)
    cmd = mpc.compute(state, np.array([1.5, 0.0]))
    alloc = mpc.allocate(state, cmd)
    assert alloc.motor.thrusts.shape == (4,)
    assert np.all(np.isfinite(alloc.motor.thrusts))


if __name__ == "__main__":
    test_linear_mpc_solver_returns_bounded_control()
    test_flight_mpc_produces_pitch_moment_for_x_error()
    test_rolling_mpc_builds_and_allocates()
    print("All MPC tests passed.")
