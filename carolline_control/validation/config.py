"""Validation configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RandomizationBounds:
    enabled: bool = True
    initial_roll_deg: tuple[float, float] = (-180.0, 180.0)
    initial_pitch_deg: tuple[float, float] = (35.0, 85.0)
    initial_yaw_deg: tuple[float, float] = (-180.0, 180.0)
    initial_omega_body: tuple[float, float] = (-0.5, 0.5)
    initial_velocity: tuple[float, float] = (-0.2, 0.2)
    spawn_xy_offset: tuple[float, float] = (-0.3, 0.3)
    mass_fraction: tuple[float, float] = (-0.05, 0.05)
    inertia_fraction: tuple[float, float] = (-0.05, 0.05)
    motor_effectiveness: tuple[float, float] = (-0.05, 0.05)
    ground_friction_scale: tuple[float, float] = (0.5, 1.5)
    cage_friction_scale: tuple[float, float] = (0.5, 1.5)
    rolling_resistance_scale: tuple[float, float] = (0.5, 1.5)
    wind_magnitude: tuple[float, float] = (0.0, 5.0)
    wind_direction_random: bool = True
    wind_gust_probability: float = 0.15
    wind_gust_magnitude: tuple[float, float] = (1.0, 8.0)
    gyro_noise_std: tuple[float, float] = (0.0, 0.05)
    accel_noise_std: tuple[float, float] = (0.0, 0.3)
    gyro_bias: tuple[float, float] = (-0.02, 0.02)
    accel_bias: tuple[float, float] = (-0.1, 0.1)
    position_noise_std: tuple[float, float] = (0.0, 0.02)
    actuator_delay_steps: tuple[int, int] = (0, 3)
    control_dt_jitter: tuple[float, float] = (0.0, 0.001)
    timestep_jitter: tuple[float, float] = (0.0, 0.0005)


@dataclass
class PhysicsTolerances:
    quat_norm_tol: float = 1e-3
    orthogonality_tol: float = 1e-2
    force_residual_tol: float = 2.0
    torque_residual_tol: float = 0.5
    thrust_sum_tol: float = 0.5


@dataclass
class HealthConfig:
    oscillation_threshold: float = 0.15
    min_samples_fft: int = 128


@dataclass
class ValidationConfig:
    num_runs: int = 200
    base_seed: int = 42
    sim_duration: float = 300.0
    mission_timeout: float = 320.0
    controller_config: str = "carolline_control/config.yaml"
    output_dir: str = "carolline_control/validation/results"
    randomization: RandomizationBounds = field(default_factory=RandomizationBounds)
    physics: PhysicsTolerances = field(default_factory=PhysicsTolerances)
    health: HealthConfig = field(default_factory=HealthConfig)
    sensitivity_gains: list[str] = field(
        default_factory=lambda: [
            "kx", "kv", "kR", "kOmega", "kR_pre", "kOmega_pre", "rolling_kp", "rolling_kd"
        ]
    )
    sensitivity_fractions: list[float] = field(
        default_factory=lambda: [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    )
    campaigns: dict[str, Any] = field(default_factory=dict)


def _pair(raw: list[float] | tuple[float, float], cast=float) -> tuple:
    return (cast(raw[0]), cast(raw[1]))


def load_validation_config(path: str | Path) -> ValidationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    rb = raw.get("randomization", {})
    randomization = RandomizationBounds(
        enabled=bool(rb.get("enabled", True)),
        initial_roll_deg=_pair(rb.get("initial_roll_deg", [-180, 180])),
        initial_pitch_deg=_pair(rb.get("initial_pitch_deg", [35, 85])),
        initial_yaw_deg=_pair(rb.get("initial_yaw_deg", [-180, 180])),
        initial_omega_body=_pair(rb.get("initial_omega_body", [-0.5, 0.5])),
        initial_velocity=_pair(rb.get("initial_velocity", [-0.2, 0.2])),
        spawn_xy_offset=_pair(rb.get("spawn_xy_offset", [-0.3, 0.3])),
        mass_fraction=_pair(rb.get("mass_fraction", [-0.05, 0.05])),
        inertia_fraction=_pair(rb.get("inertia_fraction", [-0.05, 0.05])),
        motor_effectiveness=_pair(rb.get("motor_effectiveness", [-0.05, 0.05])),
        ground_friction_scale=_pair(rb.get("ground_friction_scale", [0.5, 1.5])),
        cage_friction_scale=_pair(rb.get("cage_friction_scale", [0.5, 1.5])),
        rolling_resistance_scale=_pair(rb.get("rolling_resistance_scale", [0.5, 1.5])),
        wind_magnitude=_pair(rb.get("wind_magnitude", [0.0, 5.0])),
        wind_direction_random=bool(rb.get("wind_direction_random", True)),
        wind_gust_probability=float(rb.get("wind_gust_probability", 0.15)),
        wind_gust_magnitude=_pair(rb.get("wind_gust_magnitude", [1.0, 8.0])),
        gyro_noise_std=_pair(rb.get("gyro_noise_std", [0.0, 0.05])),
        accel_noise_std=_pair(rb.get("accel_noise_std", [0.0, 0.3])),
        gyro_bias=_pair(rb.get("gyro_bias", [-0.02, 0.02])),
        accel_bias=_pair(rb.get("accel_bias", [-0.1, 0.1])),
        position_noise_std=_pair(rb.get("position_noise_std", [0.0, 0.02])),
        actuator_delay_steps=(
            int(rb.get("actuator_delay_steps", [0, 3])[0]),
            int(rb.get("actuator_delay_steps", [0, 3])[1]),
        ),
        control_dt_jitter=_pair(rb.get("control_dt_jitter", [0.0, 0.001])),
        timestep_jitter=_pair(rb.get("timestep_jitter", [0.0, 0.0005])),
    )
    ph = raw.get("physics", {})
    physics = PhysicsTolerances(
        quat_norm_tol=float(ph.get("quat_norm_tol", 1e-3)),
        orthogonality_tol=float(ph.get("orthogonality_tol", 1e-2)),
        force_residual_tol=float(ph.get("force_residual_tol", 2.0)),
        torque_residual_tol=float(ph.get("torque_residual_tol", 0.5)),
        thrust_sum_tol=float(ph.get("thrust_sum_tol", 0.5)),
    )
    hc = raw.get("health", {})
    health = HealthConfig(
        oscillation_threshold=float(hc.get("oscillation_threshold", 0.15)),
        min_samples_fft=int(hc.get("min_samples_fft", 128)),
    )
    sens = raw.get("sensitivity", {})
    return ValidationConfig(
        num_runs=int(raw.get("num_runs", 200)),
        base_seed=int(raw.get("base_seed", 42)),
        sim_duration=float(raw.get("sim_duration", 300.0)),
        mission_timeout=float(raw.get("mission_timeout", 320.0)),
        controller_config=str(raw.get("controller_config", "carolline_control/config.yaml")),
        output_dir=str(raw.get("output_dir", "carolline_control/validation/results")),
        randomization=randomization,
        physics=physics,
        health=health,
        sensitivity_gains=list(sens.get("gains", ValidationConfig().sensitivity_gains)),
        sensitivity_fractions=list(sens.get("sweep_fractions", ValidationConfig().sensitivity_fractions)),
        campaigns=dict(raw.get("campaigns", {})),
    )
