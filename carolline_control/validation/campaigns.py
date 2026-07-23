"""Disturbance campaign definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.randomization import RunParameters, sample_run_parameters


@dataclass
class CampaignRunSpec:
    run_params: RunParameters
    campaign_mode: str


def _mission_keys(raw: dict[str, Any]) -> tuple[list[float], float, float]:
    mission = raw.get("mission", {})
    spawn = mission.get("spawn_xy", [0.0, 0.0])
    ground_z = float(raw.get("ground_z", 0.40))
    sim_duration = float(raw.get("sim_duration", 300.0))
    return spawn, ground_z, sim_duration


def iter_campaign_runs(
    campaign_name: str,
    val_cfg: ValidationConfig,
    raw_config: dict[str, Any],
    *,
    num_runs: int | None = None,
) -> Iterator[CampaignRunSpec]:
    """Yield run parameter specs for a named campaign."""
    campaigns = val_cfg.campaigns or {}
    spec = campaigns.get(campaign_name, {"description": campaign_name, "overrides": {}})
    spawn, ground_z, sim_duration = _mission_keys(raw_config)
    duration = val_cfg.sim_duration
    n = num_runs if num_runs is not None else val_cfg.num_runs
    bounds = val_cfg.randomization

    if campaign_name == "constant_wind":
        speeds = spec.get("wind_speeds", [0, 1, 2, 3, 5, 8, 10])
        per = int(spec.get("runs_per_speed", max(1, n // max(len(speeds), 1))))
        run_id = 0
        for speed in speeds:
            for i in range(per):
                seed = val_cfg.base_seed + run_id
                ov = {
                    "randomization": {
                        "wind_magnitude": [float(speed), float(speed)],
                        "wind_direction_random": False,
                    }
                }
                params = sample_run_parameters(
                    run_id=run_id,
                    seed=seed,
                    bounds=bounds,
                    spawn_xy=spawn,
                    ground_z=ground_z,
                    sim_duration=duration,
                    campaign=campaign_name,
                    overrides=ov,
                )
                params.wind_vector = np.array([float(speed), 0.0, 0.0])
                yield CampaignRunSpec(params, "nominal")
                run_id += 1
        return

    mode_map = {
        "nominal": "nominal",
        "random_wind_direction": "nominal",
        "sinusoidal_wind": "sinusoidal_wind",
        "dryden_turbulence": "dryden_turbulence",
        "wind_gust": "nominal",
        "mass_uncertainty": "nominal",
        "sensor_noise": "nominal",
        "actuator_delay": "nominal",
        "combined_disturbances": "nominal",
    }
    campaign_mode = mode_map.get(campaign_name, "nominal")
    overrides: dict[str, Any] = dict(spec.get("overrides", {}))

    if campaign_name == "wind_gust":
        overrides.setdefault("randomization", {})
        overrides["randomization"]["wind_gust_probability"] = 0.8
        overrides["randomization"]["wind_magnitude"] = [0.0, 2.0]

    if campaign_name == "mass_uncertainty":
        overrides.setdefault("randomization", {})
        overrides["randomization"]["mass_fraction"] = [-0.05, 0.05]
        overrides["randomization"]["inertia_fraction"] = [-0.05, 0.05]
        overrides["randomization"]["wind_magnitude"] = [0.0, 0.0]

    if campaign_name == "sensor_noise":
        overrides.setdefault("randomization", {})
        overrides["randomization"]["gyro_noise_std"] = [0.02, 0.08]
        overrides["randomization"]["accel_noise_std"] = [0.1, 0.5]
        overrides["randomization"]["position_noise_std"] = [0.005, 0.03]

    if campaign_name == "actuator_delay":
        overrides.setdefault("randomization", {})
        overrides["randomization"]["actuator_delay_steps"] = [2, 6]

    for run_id in range(n):
        seed = val_cfg.base_seed + run_id
        params = sample_run_parameters(
            run_id=run_id,
            seed=seed,
            bounds=bounds,
            spawn_xy=spawn,
            ground_z=ground_z,
            sim_duration=duration,
            campaign=campaign_name,
            overrides=overrides if overrides else None,
        )
        yield CampaignRunSpec(params, campaign_mode)
