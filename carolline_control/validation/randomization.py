"""Per-run parameter sampling for Monte Carlo validation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from carolline_control.validation.config import RandomizationBounds


@dataclass
class RunParameters:
    run_id: int
    seed: int
    initial_qpos: list[float]
    initial_qvel: list[float]
    mass_scale: float
    inertia_scale: float
    motor_effectiveness: float
    ground_friction_scale: float
    cage_friction_scale: float
    rolling_resistance_scale: float
    wind_vector: np.ndarray
    wind_gust_times: list[float]
    wind_gust_vectors: list[np.ndarray]
    gyro_noise_std: float
    accel_noise_std: float
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    position_noise_std: float
    actuator_delay_steps: int
    control_dt_jitter: float
    timestep_jitter: float
    gain_overrides: dict[str, float]
    campaign: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["wind_vector"] = self.wind_vector.tolist()
        d["wind_gust_vectors"] = [v.tolist() for v in self.wind_gust_vectors]
        d["gyro_bias"] = self.gyro_bias.tolist()
        d["accel_bias"] = self.accel_bias.tolist()
        return d


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def sample_run_parameters(
    run_id: int,
    seed: int,
    bounds: RandomizationBounds,
    spawn_xy: list[float],
    ground_z: float,
    sim_duration: float,
    campaign: str = "nominal",
    overrides: dict[str, Any] | None = None,
    gain_overrides: dict[str, float] | None = None,
) -> RunParameters:
    """Sample one run's environment and initial conditions."""
    rng = np.random.default_rng(seed)
    ov = overrides or {}
    rb = RandomizationBounds(**bounds.__dict__)
    if "randomization" in ov:
        for k, v in ov["randomization"].items():
            if hasattr(rb, k):
                if k.endswith("_steps"):
                    setattr(rb, k, (int(v[0]), int(v[1])))
                elif isinstance(v, list) and len(v) == 2 and k != "wind_direction_random":
                    setattr(rb, k, (float(v[0]), float(v[1])))
                else:
                    setattr(rb, k, v)

    roll = np.radians(_uniform(rng, rb.initial_roll_deg))
    pitch = np.radians(_uniform(rng, rb.initial_pitch_deg))
    yaw = np.radians(_uniform(rng, rb.initial_yaw_deg))

    # Ground rolling start: tilt about horizontal axis (matches main.py convention).
    if campaign != "sensitivity_nominal" and rb.initial_pitch_deg[0] >= 30.0:
        tilt_deg = np.degrees(pitch)
        heading = yaw
        axis = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
        half = np.radians(tilt_deg) * 0.5
        quat = np.array(
            [np.cos(half), *(axis * np.sin(half))],
            dtype=float,
        )
    else:
        quat = _euler_to_quat(roll, pitch, yaw)

    sx = float(spawn_xy[0]) + _uniform(rng, rb.spawn_xy_offset)
    sy = float(spawn_xy[1]) + _uniform(rng, rb.spawn_xy_offset)
    qpos = [sx, sy, ground_z, *quat.tolist()]

    qvel = np.zeros(6)
    qvel[:3] = [_uniform(rng, rb.initial_velocity) for _ in range(3)]
    qvel[3:] = [_uniform(rng, rb.initial_omega_body) for _ in range(3)]

    wind_speed = _uniform(rng, rb.wind_magnitude)
    if rb.wind_direction_random:
        wind_dir = rng.uniform(0, 2 * np.pi)
        wind = wind_speed * np.array([np.cos(wind_dir), np.sin(wind_dir), 0.0])
    else:
        wind = np.array([wind_speed, 0.0, 0.0])

    gust_times: list[float] = []
    gust_vectors: list[np.ndarray] = []
    if rb.wind_gust_probability > 0:
        t = rng.uniform(5.0, max(sim_duration - 10.0, 6.0))
        while t < sim_duration - 5.0:
            if rng.random() < rb.wind_gust_probability:
                gmag = _uniform(rng, rb.wind_gust_magnitude)
                gdir = rng.uniform(0, 2 * np.pi)
                gust_times.append(float(t))
                gust_vectors.append(gmag * np.array([np.cos(gdir), np.sin(gdir), 0.0]))
            t += rng.uniform(8.0, 25.0)

    return RunParameters(
        run_id=run_id,
        seed=seed,
        initial_qpos=qpos,
        initial_qvel=qvel.tolist(),
        mass_scale=float(1.0 + _uniform(rng, rb.mass_fraction)),
        inertia_scale=float(1.0 + _uniform(rng, rb.inertia_fraction)),
        motor_effectiveness=float(1.0 + _uniform(rng, rb.motor_effectiveness)),
        ground_friction_scale=float(_uniform(rng, rb.ground_friction_scale)),
        cage_friction_scale=float(_uniform(rng, rb.cage_friction_scale)),
        rolling_resistance_scale=float(_uniform(rng, rb.rolling_resistance_scale)),
        wind_vector=wind,
        wind_gust_times=gust_times,
        wind_gust_vectors=gust_vectors,
        gyro_noise_std=_uniform(rng, rb.gyro_noise_std),
        accel_noise_std=_uniform(rng, rb.accel_noise_std),
        gyro_bias=np.array([_uniform(rng, rb.gyro_bias) for _ in range(3)]),
        accel_bias=np.array([_uniform(rng, rb.accel_bias) for _ in range(3)]),
        position_noise_std=_uniform(rng, rb.position_noise_std),
        actuator_delay_steps=int(rng.integers(rb.actuator_delay_steps[0], rb.actuator_delay_steps[1] + 1)),
        control_dt_jitter=_uniform(rng, rb.control_dt_jitter),
        timestep_jitter=_uniform(rng, rb.timestep_jitter),
        gain_overrides=dict(gain_overrides or {}),
        campaign=campaign,
        metadata={"roll_deg": np.degrees(roll), "pitch_deg": np.degrees(pitch), "yaw_deg": np.degrees(yaw)},
    )
