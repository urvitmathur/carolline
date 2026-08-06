"""Extract validation metrics from logs into report_data.yaml."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "carolline_control" / "logs"
OUT = REPO / "carolline_control" / "docs" / "dissertation" / "report_data.yaml"


def _read_csv(name: str) -> list[dict]:
    path = LOGS / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flight_log_metrics() -> dict:
    rows = _read_csv("flight_log.csv")
    if not rows:
        return {}
    t = np.array([float(r["time"]) for r in rows])
    pz = np.array([float(r["pz"]) for r in rows])
    des = np.array([float(r["des_pz"]) for r in rows])
    modes = [r["mode"] for r in rows]

    def first_mode(name: str) -> float | None:
        for r in rows:
            if r["mode"] == name:
                return float(r["time"])
        return None

    takeoff_rows = [r for r in rows if r["mode"] == "TAKEOFF"]
    z_peak_takeoff = float(max((float(r["pz"]) for r in takeoff_rows), default=0.0))
    if takeoff_rows:
        hover_h = float(takeoff_rows[0]["des_pz"])
    else:
        hover_h = 1.0

    return {
        "duration_s": float(t[-1]),
        "mode_times_s": {
            "pretakeoff": first_mode("PRETAKEOFF"),
            "upright": first_mode("UPRIGHT"),
            "takeoff": first_mode("TAKEOFF"),
            "hover": first_mode("HOVER"),
            "flight": first_mode("FLIGHT"),
            "landing": first_mode("LANDING"),
            "idle": first_mode("IDLE"),
        },
        "takeoff_z_peak_m": z_peak_takeoff,
        "takeoff_z_overshoot_m": max(0.0, z_peak_takeoff - hover_h),
        "hover_height_m": hover_h,
        "final_mode": modes[-1],
        "max_z_tracking_error_m": float(np.max(np.abs(pz - des))),
    }


def _recovery_metrics() -> dict:
    rows = _read_csv("recovery_validation.csv")
    if not rows:
        return {}
    success = sum(1 for r in rows if str(r.get("success", "")).lower() in ("true", "1", "yes"))
    tilts = [float(r["initial_tilt_deg"]) for r in rows if r.get("initial_tilt_deg")]
    times = [
        float(r["recovery_time_s"])
        for r in rows
        if r.get("recovery_time_s") and float(r["recovery_time_s"]) > 0
    ]
    return {
        "trials": len(rows),
        "success_count": success,
        "success_rate_pct": round(100.0 * success / max(len(rows), 1), 1),
        "tilt_deg_min": round(min(tilts), 1) if tilts else None,
        "tilt_deg_max": round(max(tilts), 1) if tilts else None,
        "mean_time_to_upright_s": round(float(np.mean(times)), 2) if times else None,
    }


def _takeoff_metrics() -> dict:
    rows = _read_csv("takeoff_validation.csv")
    if not rows:
        return {}
    hover_ok = sum(1 for r in rows if str(r.get("hover_achieved", "")).lower().startswith("y"))
    overs = [float(r["max_vertical_overshoot_m"]) for r in rows if r.get("max_vertical_overshoot_m")]
    t_hover = [float(r["time_to_hover_s"]) for r in rows if r.get("time_to_hover_s") and float(r["time_to_hover_s"]) > 0]
    return {
        "trials": len(rows),
        "hover_achieved_count": hover_ok,
        "hover_achieved_rate_pct": round(100.0 * hover_ok / max(len(rows), 1), 1),
        "mean_time_to_hover_s": round(float(np.mean(t_hover)), 2) if t_hover else None,
        "mean_vertical_overshoot_m": round(float(np.mean(overs)), 3) if overs else None,
        "max_vertical_overshoot_m": round(float(np.max(overs)), 3) if overs else None,
    }


def _monte_carlo_metrics() -> dict:
    rows = _read_csv("monte_carlo_validation.csv")
    if not rows:
        rows = _read_csv("monte_carlo_paper.csv")
    if not rows:
        return {}
    success = sum(1 for r in rows if str(r.get("success", "")).lower() in ("true", "1", "yes"))
    return {
        "trials": len(rows),
        "success_count": success,
        "success_rate_pct": round(100.0 * success / max(len(rows), 1), 1),
    }


def main() -> None:
    data = {
        "generated_from": "carolline_control/logs/*.csv",
        "seed_reference": 42,
        "vehicle": {
            "mass_kg": 2.125,
            "hover_height_m": 1.0,
            "motor_thrust_range_N": [-13.0, 13.0],
            "cage_radius_m": 0.4,
        },
        "hybrid_mission": _flight_log_metrics(),
        "recovery": _recovery_metrics(),
        "takeoff": _takeoff_metrics(),
        "monte_carlo": _monte_carlo_metrics(),
        "maze_slam": {
            "arena_size_m": 9.0,
            "oracle_goal_reached_s": 105.6,
            "oracle_dist_goal_m": 0.45,
            "map_resolution_m": 0.03,
        },
        "takeoff_overshoot_before_fix_m": 0.286,
        "takeoff_overshoot_after_fix_m": _flight_log_metrics().get("takeoff_z_overshoot_m", 0.0),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUT}")
    print(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
