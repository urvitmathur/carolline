"""Load YAML configuration into typed runtime objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carolline_control.utils.types import ControlMode, ControllerConfig


def _mode_from_string(name: str) -> ControlMode:
    return ControlMode[name.strip().upper()]


def build_mission_waypoints(raw: dict[str, Any], hover_height: float) -> list[list[float]]:
    """Build aerial legs: four axis directions out-and-back, then return to spawn."""
    mission = raw.get("mission", {})
    roll_target = mission.get("roll_target", [3.0, 0.0])
    spawn = mission.get("spawn_xy", [0.0, 0.0])
    leg = float(mission.get("leg_distance", 3.0))
    home = np.array([float(roll_target[0]), float(roll_target[1]), hover_height], dtype=float)
    spawn_xy = np.array(spawn[:2], dtype=float)
    waypoints: list[list[float]] = []
    for delta in (
        np.array([leg, 0.0, 0.0]),
        np.array([-leg, 0.0, 0.0]),
        np.array([0.0, leg, 0.0]),
        np.array([0.0, -leg, 0.0]),
    ):
        waypoints.append((home + delta).tolist())
        waypoints.append(home.tolist())
    waypoints.append([float(spawn_xy[0]), float(spawn_xy[1]), hover_height])
    return waypoints


def load_config(path: str | Path) -> ControllerConfig:
    """Load controller configuration from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    repo_root = config_path.parent.parent
    model_path = Path(raw["model_path"])
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    inertia = raw.get("inertia")
    inertia_mat = np.diag(inertia) if inertia is not None else np.eye(3)

    mission = raw.get("mission", {})
    hover_height = float(raw["hover_height"])
    waypoints = raw.get("waypoints")
    if waypoints is None and mission:
        waypoints = build_mission_waypoints(raw, hover_height)

    roll_target = np.asarray(mission.get("roll_target", [3.0, 0.0]), dtype=float)
    spawn_xy = np.asarray(mission.get("spawn_xy", [0.0, 0.0]), dtype=float)

    return ControllerConfig(
        model_path=str(model_path),
        cage_radius=float(raw["cage_radius"]),
        mass=float(raw["mass"]) if raw.get("mass") is not None else 0.0,
        gravity=float(raw["gravity"]),
        inertia=inertia_mat,
        motor_min=float(raw["motor_min"]),
        motor_max=float(raw["motor_max"]),
        kx=float(raw["kx"]),
        kv=float(raw["kv"]),
        kx_z=float(raw.get("kx_z", raw["kx"] * 2.5)),
        kv_z=float(raw.get("kv_z", raw["kv"] * 1.2)),
        kR=float(raw["kR"]),
        kOmega=float(raw["kOmega"]),
        kR_pre=float(raw["kR_pre"]),
        kOmega_pre=float(raw["kOmega_pre"]),
        k_roll=float(raw["k_roll"]),
        k_yaw=float(raw["k_yaw"]),
        upright_cos_threshold=float(raw["upright_cos_threshold"]),
        takeoff_cos_threshold=float(raw.get("takeoff_cos_threshold", 0.98)),
        upright_settle_time=float(raw.get("upright_settle_time", 1.5)),
        upright_omega_tolerance=float(raw.get("upright_omega_tolerance", 0.12)),
        takeoff_height=float(raw["takeoff_height"]),
        hover_height=hover_height,
        landing_height=float(raw["landing_height"]),
        ground_height_threshold=float(raw["ground_height_threshold"]),
        hover_altitude_tolerance=float(raw.get("hover_altitude_tolerance", 0.1)),
        hover_velocity_tolerance=float(raw.get("hover_velocity_tolerance", 0.2)),
        min_airborne_thrust_fraction=float(raw.get("min_airborne_thrust_fraction", 0.15)),
        landing_settle_time=float(raw.get("landing_settle_time", 1.0)),
        pre_takeoff_thrust_fraction=float(raw["pre_takeoff_thrust_fraction"]),
        rolling_max_speed=float(raw["rolling_max_speed"]),
        rolling_kp=float(raw.get("rolling_kp", 0.55)),
        rolling_kd=float(raw.get("rolling_kd", 0.85)),
        motor_slew_rate=float(raw.get("motor_slew_rate", 1200.0)),
        roll_target=roll_target,
        roll_position_tolerance=float(mission.get("roll_position_tolerance", 0.2)),
        spawn_xy=spawn_xy,
        leg_distance=float(mission.get("leg_distance", 3.0)),
        segment_duration=float(mission.get("segment_duration", 7.0)),
        mission_yaw=float(mission.get("mission_yaw", 0.0)),
        min_flight_center_z=float(raw.get("min_flight_center_z", 0.72)),
        waypoint_reach_tolerance=float(mission.get("waypoint_reach_tolerance", 0.15)),
        yaw_drag_coeff=np.asarray(raw["yaw_drag_coeff"], dtype=float),
        rotor_positions=np.asarray(raw["rotor_positions"], dtype=float),
        ground_omega_weights=np.asarray(
            raw.get("ground_omega_weights", [1.0, 1.0, 30.0]), dtype=float
        ),
        ground_omega_kp=np.asarray(
            raw.get("ground_omega_kp", [6.0, 6.0, 3.0]), dtype=float
        ),
        rolling_omega_kp=np.asarray(
            raw.get("rolling_omega_kp", [6.0, 6.0, 2.0]), dtype=float
        ),
        rolling_braking_distance=float(raw.get("rolling_braking_distance", 1.8)),
        ground_pseudoinverse_damping=float(
            raw.get("ground_pseudoinverse_damping", 1.0e-6)
        ),
        pre_takeoff_omega_gain=float(raw.get("pre_takeoff_omega_gain", 1.5)),
        pre_takeoff_omega_limit=float(raw.get("pre_takeoff_omega_limit", 6.0)),
        contact_min_normal_z=float(raw.get("contact_min_normal_z", 0.15)),
        contact_force_threshold=float(raw.get("contact_force_threshold", 0.0)),
        contact_loss_grace=float(raw.get("contact_loss_grace", 0.30)),
        rolling_stall_speed=float(raw.get("rolling_stall_speed", 0.04)),
        rolling_stall_time=float(raw.get("rolling_stall_time", 1.5)),
        ground_max_unload_fraction=float(
            raw.get("ground_max_unload_fraction", 0.35)
        ),
        initial_mode=_mode_from_string(raw.get("initial_mode", "PRETAKEOFF")),
        waypoints=waypoints,
    )


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Load raw YAML dictionary (includes sim-only keys)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
