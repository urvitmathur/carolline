"""Load MPC tuning from mpc_config.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from carolline_control.controllers.mpc.flight_mpc import FlightMpcWeights
from carolline_control.controllers.mpc.rolling_mpc import RollingMpcWeights

CAROLLINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPC_CONFIG = CAROLLINE_ROOT / "mpc_config.yaml"


def load_mpc_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_MPC_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def flight_mpc_weights_from_config(raw: dict) -> FlightMpcWeights:
    f = raw.get("flight", {})
    return FlightMpcWeights(
        q_pos=np.asarray(f.get("q_pos", [14.0, 14.0, 20.0]), dtype=float),
        q_vel=np.asarray(f.get("q_vel", [9.0, 9.0, 11.0]), dtype=float),
        q_att=np.asarray(f.get("q_att", [16.0, 16.0, 7.0]), dtype=float),
        q_rate=np.asarray(f.get("q_rate", [3.5, 3.5, 2.5]), dtype=float),
        qf_scale=float(f.get("qf_scale", 2.0)),
        r_thrust=float(f.get("r_thrust", 0.10)),
        r_moment=np.asarray(f.get("r_moment", [0.14, 0.14, 0.07]), dtype=float),
        horizon=int(f.get("horizon", 15)),
        dt=float(f.get("dt", 0.004)),
    )


def rolling_mpc_weights_from_config(raw: dict) -> RollingMpcWeights:
    r = raw.get("rolling", {})
    return RollingMpcWeights(
        q_vel=np.asarray(r.get("q_vel", [12.0, 12.0]), dtype=float),
        q_rate=np.asarray(r.get("q_rate", [5.0, 5.0, 2.5]), dtype=float),
        qf_scale=float(r.get("qf_scale", 2.0)),
        r_torque=np.asarray(r.get("r_torque", [0.08, 0.08, 0.12]), dtype=float),
        horizon=int(r.get("horizon", 12)),
        dt=float(r.get("dt", 0.004)),
        torque_limit=float(r.get("torque_limit", 8.0)),
    )
