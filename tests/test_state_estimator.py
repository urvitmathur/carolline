"""Tests for oracle vs sensor-only state estimation."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.sim_estimator import build_estimator, seed_estimator_from_sim
from carolline_control.visualization.markers import compile_model_with_markers


def test_sensor_only_matches_oracle_at_spawn() -> None:
    config_path = REPO_ROOT / "carolline_control" / "config.yaml"
    config = load_config(config_path)
    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)

    qpos = [1.0, 2.0, 0.40, 0.924, 0.0, 0.383, 0.0]
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    oracle = build_estimator(model, config, sensor_only=False)
    sensor = build_estimator(model, config, sensor_only=True)
    seed_estimator_from_sim(sensor, data)

    s_oracle = oracle.estimate(data)
    s_sensor = sensor.estimate(data)

    assert np.linalg.norm(s_sensor.position - s_oracle.position) < 1e-6
    assert np.linalg.norm(s_sensor.quaternion - s_oracle.quaternion) < 1e-3
    assert sensor.last_oracle_position_error < 1e-6
    assert sensor.last_oracle_attitude_error < 1e-3


def test_sensor_only_integrates_stationary_position() -> None:
    config_path = REPO_ROOT / "carolline_control" / "config.yaml"
    config = load_config(config_path)
    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)

    data.qpos[:7] = [0.0, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    estimator = build_estimator(model, config, sensor_only=True)
    seed_estimator_from_sim(estimator, data)

    state0 = estimator.estimate(data)
    for _ in range(50):
        mujoco.mj_step(model, data)
        state = estimator.estimate(data)

    assert np.linalg.norm(state.position - state0.position) < 1e-4


def test_sensor_only_requires_reset() -> None:
    config_path = REPO_ROOT / "carolline_control" / "config.yaml"
    config = load_config(config_path)
    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    estimator = StateEstimator(model, config, sensor_only=True)
    estimator.fill_inertial_params(config)
    try:
        estimator.estimate(data)
        raise AssertionError("Expected RuntimeError when reset() was not called")
    except RuntimeError as exc:
        assert "reset" in str(exc).lower()


if __name__ == "__main__":
    test_sensor_only_matches_oracle_at_spawn()
    print("PASS test_sensor_only_matches_oracle_at_spawn")
    test_sensor_only_integrates_stationary_position()
    print("PASS test_sensor_only_integrates_stationary_position")
    test_sensor_only_requires_reset()
    print("PASS test_sensor_only_requires_reset")
    print("All sensor-only estimator tests passed.")
