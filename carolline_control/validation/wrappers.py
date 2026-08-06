"""Non-invasive wrappers around estimation and actuation."""

from __future__ import annotations

from collections import deque

import numpy as np

from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.so3 import quat_to_rot
from carolline_control.utils.types import RobotState
from carolline_control.validation.randomization import RunParameters


class NoisyStateEstimator:
    """Wrap StateEstimator; inject sensor noise without modifying controller code."""

    def __init__(self, inner: StateEstimator, params: RunParameters) -> None:
        self._inner = inner
        self._params = params

    @property
    def inner(self) -> StateEstimator:
        return self._inner

    def fill_inertial_params(self, config) -> None:
        self._inner.fill_inertial_params(config)

    @property
    def sensor_only(self) -> bool:
        return self._inner.sensor_only

    @property
    def last_oracle_position_error(self) -> float:
        return self._inner.last_oracle_position_error

    @property
    def last_oracle_attitude_error(self) -> float:
        return self._inner.last_oracle_attitude_error

    def reset(self, position: np.ndarray) -> None:
        self._inner.reset(position)

    def estimate(self, data) -> RobotState:
        state = self._inner.estimate(data)
        p = self._params
        rng = np.random.default_rng(int(data.time * 1e6) % 2**31)

        pos = state.position + rng.normal(0, p.position_noise_std, 3)
        quat = state.quaternion + rng.normal(0, p.position_noise_std * 0.01, 4)
        quat = quat / max(np.linalg.norm(quat), 1e-9)
        omega = state.omega_body + p.gyro_bias + rng.normal(0, p.gyro_noise_std, 3)
        accel = state.accel_body + p.accel_bias + rng.normal(0, p.accel_noise_std, 3)
        vel = state.velocity + rng.normal(0, p.position_noise_std, 3)

        return RobotState(
            position=pos,
            velocity=vel,
            quaternion=quat,
            rotation=quat_to_rot(quat),
            omega_body=omega,
            omega_world=state.rotation @ omega,
            accel_body=accel,
            on_ground=state.on_ground,
            ground_contact_z=state.ground_contact_z,
            time=state.time,
            contact_point_world=state.contact_point_world.copy(),
            contact_normal_world=state.contact_normal_world.copy(),
            contact_force=state.contact_force,
            contact_confidence=state.contact_confidence,
            contact_valid=state.contact_valid,
        )


class ActuatorDelayBuffer:
    """Apply motor command delay in validation layer only."""

    def __init__(self, delay_steps: int) -> None:
        self._delay = max(int(delay_steps), 0)
        self._buf: deque[np.ndarray] = deque(maxlen=self._delay + 1)

    def push(self, thrusts: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(thrusts, dtype=float).copy())
        if self._delay == 0:
            return self._buf[-1]
        if len(self._buf) <= self._delay:
            return np.zeros_like(thrusts)
        return self._buf[0]
