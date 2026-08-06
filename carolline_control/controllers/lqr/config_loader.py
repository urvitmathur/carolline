"""Load LQR tuning weights from lqr_config.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from carolline_control.controllers.lqr.flight_lqr import FlightLqrWeights
from carolline_control.controllers.lqr.rolling_lqr import RollingLqrWeights

CAROLLINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LQR_CONFIG = CAROLLINE_ROOT / "lqr_config.yaml"


def load_lqr_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_LQR_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def flight_lqr_weights_from_config(raw: dict) -> FlightLqrWeights:
    f = raw.get("flight", {})
    return FlightLqrWeights(
        q_pos=np.asarray(f.get("q_pos", [12.0, 12.0, 18.0]), dtype=float),
        q_vel=np.asarray(f.get("q_vel", [8.0, 8.0, 10.0]), dtype=float),
        q_att=np.asarray(f.get("q_att", [14.0, 14.0, 6.0]), dtype=float),
        q_rate=np.asarray(f.get("q_rate", [3.0, 3.0, 2.0]), dtype=float),
        r_thrust=float(f.get("r_thrust", 0.08)),
        r_moment=np.asarray(f.get("r_moment", [0.12, 0.12, 0.06]), dtype=float),
        dt=float(f.get("dt", 0.004)),
    )


def rolling_lqr_weights_from_config(raw: dict) -> RollingLqrWeights:
    r = raw.get("rolling", {})
    return RollingLqrWeights(
        q_vel=np.asarray(r.get("q_vel", [10.0, 10.0]), dtype=float),
        q_rate=np.asarray(r.get("q_rate", [4.0, 4.0, 2.0]), dtype=float),
        r_torque=np.asarray(r.get("r_torque", [0.06, 0.06, 0.10]), dtype=float),
        dt=float(r.get("dt", 0.004)),
    )
