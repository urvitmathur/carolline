"""
Headless multi-scenario study: geometric (PD/SO3) vs LQR.

Run from repo root:
    python carolline_control/scripts/run_lqr_study.py
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.attitude_controller import AttitudeController
from carolline_control.controllers.flight_controller import FlightController
from carolline_control.controllers.lqr.comparison_metrics import MetricsAccumulator
from carolline_control.controllers.lqr.config_loader import flight_lqr_weights_from_config, load_lqr_config, rolling_lqr_weights_from_config
from carolline_control.controllers.lqr.flight_lqr import FlightLqrController
from carolline_control.controllers.lqr.rolling_lqr import RollingLqrController
from carolline_control.controllers.motor_mixer import MotorMixer
from carolline_control.controllers.rolling_controller import RollingController
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.controllers.torque_controller import TorqueController
from carolline_control.scripts.compare_flight_controllers import (
    _geometric_step,
    _lqr_step,
    _target,
    _tilt_deg,
    _upright_hover_qpos,
)
from carolline_control.utils.types import ControlMode, TrajectoryTarget
from carolline_control.visualization.markers import compile_model_with_markers


@dataclass
class TrialRow:
    domain: str
    scenario: str
    controller: str
    rmse_pos: float
    rmse_vel: float
    max_pos_err: float
    max_vel_err: float
    settling_s: float
    effort: float
    saturation_pct: float
    stable: bool


def _flight_trial(
    config,
    model: mujoco.MjModel,
    *,
    use_lqr: bool,
    lqr: FlightLqrController | None,
    duration: float,
    step_x: float,
    height: float,
    label: str,
    scenario: str,
) -> TrialRow:
    data = mujoco.MjData(model)
    anchor = np.zeros(2)
    data.qpos[:7] = _upright_hover_qpos(anchor, height)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    estimator = StateEstimator(model, config)
    flight = FlightController(config)
    attitude = AttitudeController()
    torque = TorqueController()
    mixer = MotorMixer(config)
    acc = MetricsAccumulator(motor_min=config.motor_min, motor_max=config.motor_max)

    while data.time < duration:
        state = estimator.estimate(data)
        target = _target(data.time, anchor, height, step_x, 1.0)
        if use_lqr:
            thrusts = _lqr_step(state, target, lqr, mixer)
        else:
            thrusts = _geometric_step(state, target, config, flight, attitude, torque, mixer)
        data.ctrl[:] = thrusts
        acc.add(
            time=data.time,
            pos=state.position,
            vel=state.velocity,
            pos_ref=target.position,
            vel_ref=target.velocity,
            motors=thrusts,
            tilt_deg=_tilt_deg(state),
        )
        mujoco.mj_step(model, data)

    m = acc.finalize(controller=label, domain="flight", settle_axis=0, step_target=step_x)
    return TrialRow("flight", scenario, label, m.rmse_position, m.rmse_velocity, m.max_position_error,
                    m.max_velocity_error, m.settling_time_s, m.control_effort, m.motor_saturation_pct, m.stable)


def _rolling_trial(
    config,
    model: mujoco.MjModel,
    *,
    use_lqr: bool,
    lqr: RollingLqrController | None,
    duration: float,
    speed: float,
    tilt_deg: float,
    label: str,
    scenario: str,
    ground_z: float,
) -> TrialRow:
    data = mujoco.MjData(model)
    half = math.radians(tilt_deg) * 0.5
    data.qpos[:7] = [0.0, 0.0, ground_z, math.cos(half), 0.0, math.sin(half), 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)

    estimator = StateEstimator(model, config)
    paper = RollingController(config)
    acc = MetricsAccumulator(motor_min=config.motor_min, motor_max=config.motor_max)
    v_des = np.array([speed, 0.0])

    while data.time < duration:
        state = estimator.estimate(data)
        if use_lqr:
            cmd = lqr.compute(state, v_des)
            motor = lqr.allocate(state, cmd).motor
        else:
            cmd = paper.compute(state, v_des, yaw=0.0, dt=float(model.opt.timestep))
            motor = paper.allocate(state, cmd).motor
        data.ctrl[:] = motor.thrusts
        acc.add(
            time=data.time,
            pos=state.position,
            vel=state.velocity,
            pos_ref=state.position,
            vel_ref=np.array([speed, 0.0, 0.0]),
            motors=motor.thrusts,
            tilt_deg=_tilt_deg(state),
        )
        mujoco.mj_step(model, data)

    m = acc.finalize(controller=label, domain="rolling")
    return TrialRow("rolling", scenario, label, m.rmse_position, m.rmse_velocity, m.max_position_error,
                    m.max_velocity_error, m.settling_time_s, m.control_effort, m.motor_saturation_pct, m.stable)


def _aggregate(rows: list[TrialRow], domain: str, controller: str) -> dict:
    subset = [r for r in rows if r.domain == domain and r.controller == controller]
    if not subset:
        return {}
    return {
        "n": len(subset),
        "rmse_pos_mean": float(np.mean([r.rmse_pos for r in subset])),
        "rmse_vel_mean": float(np.mean([r.rmse_vel for r in subset])),
        "settle_mean": float(np.mean([r.settling_s for r in subset if math.isfinite(r.settling_s)])),
        "effort_mean": float(np.mean([r.effort for r in subset])),
        "sat_mean": float(np.mean([r.saturation_pct for r in subset])),
        "stable_rate": float(np.mean([1.0 if r.stable else 0.0 for r in subset])),
    }


def main() -> None:
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    raw = load_raw_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    lqr_raw = load_lqr_config()

    model_path = REPO_ROOT / config.model_path
    model = compile_model_with_markers(model_path, config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    lqr_flight = FlightLqrController(config, flight_lqr_weights_from_config(lqr_raw))
    lqr_roll = RollingLqrController(config, rolling_lqr_weights_from_config(lqr_raw))

    rows: list[TrialRow] = []
    flight_steps = [0.5, 0.75, 1.0]
    roll_speeds = [1.0, 1.5, 2.0]
    roll_tilts = [45.0, 55.0, 65.0]
    ground_z = float(raw.get("ground_z", 0.40))

    for step in flight_steps:
        sc = f"step_{step:.2f}m"
        rows.append(_flight_trial(config, model, use_lqr=False, lqr=None, duration=8.0,
                                step_x=step, height=1.0, label="geometric", scenario=sc))
        rows.append(_flight_trial(config, model, use_lqr=True, lqr=lqr_flight, duration=8.0,
                                step_x=step, height=1.0, label="lqr", scenario=sc))

    for speed in roll_speeds:
        for tilt in roll_tilts:
            sc = f"speed_{speed:.1f}_tilt_{tilt:.0f}"
            rows.append(_rolling_trial(config, model, use_lqr=False, lqr=None, duration=6.0,
                                     speed=speed, tilt_deg=tilt, label="geometric", scenario=sc,
                                     ground_z=ground_z))
            lqr_roll = RollingLqrController(config, rolling_lqr_weights_from_config(lqr_raw))
            rows.append(_rolling_trial(config, model, use_lqr=True, lqr=lqr_roll, duration=6.0,
                                     speed=speed, tilt_deg=tilt, label="lqr", scenario=sc,
                                     ground_z=ground_z))

    out_dir = REPO_ROOT / "carolline_control" / "plots" / "lqr_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "study_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    geo_f = _aggregate(rows, "flight", "geometric")
    lqr_f = _aggregate(rows, "flight", "lqr")
    geo_r = _aggregate(rows, "rolling", "geometric")
    lqr_r = _aggregate(rows, "rolling", "lqr")

    def wins(metric: str, geo: dict, lqr: dict, *, lower_better: bool = True) -> str:
        g = geo.get(metric, float("inf"))
        l = lqr.get(metric, float("inf"))
        if abs(g - l) < 1e-6:
            return "tie"
        if lower_better:
            return "lqr" if l < g else "geometric"
        return "lqr" if l > g else "geometric"

    print("=" * 60)
    print("HEADLESS STUDY: Geometric (PD/SO3) vs LQR")
    print("=" * 60)
    print(f"\nFlight scenarios: {len(flight_steps)} step sizes x 2 controllers")
    print(f"Rolling scenarios: {len(roll_speeds)*len(roll_tilts)} pose/speed combos x 2 controllers")
    print(f"Full results: {csv_path}\n")

    print("FLIGHT (hover step response, lower is better)")
    print(f"  {'metric':<22} {'geometric':>12} {'lqr':>12} {'winner':>10}")
    for key, label, lb in [
        ("rmse_pos_mean", "RMSE position", True),
        ("rmse_vel_mean", "RMSE velocity", True),
        ("settle_mean", "Settling time", True),
        ("effort_mean", "Motor effort", True),
        ("sat_mean", "Saturation %", True),
    ]:
        print(f"  {label:<22} {geo_f[key]:>12.4f} {lqr_f[key]:>12.4f} {wins(key, geo_f, lqr_f, lower_better=lb):>10}")

    print("\nROLLING (velocity tracking, lower RMSE/effort is better)")
    print(f"  {'metric':<22} {'geometric':>12} {'lqr':>12} {'winner':>10}")
    for key, label, lb in [
        ("rmse_vel_mean", "RMSE velocity", True),
        ("effort_mean", "Motor effort", True),
        ("sat_mean", "Saturation %", True),
        ("stable_rate", "Stability rate", False),
    ]:
        print(f"  {label:<22} {geo_r[key]:>12.4f} {lqr_r[key]:>12.4f} {wins(key, geo_r, lqr_r, lower_better=lb):>10}")

    flight_score = sum([
        wins("rmse_pos_mean", geo_f, lqr_f) == "lqr",
        wins("rmse_vel_mean", geo_f, lqr_f) == "lqr",
        wins("settle_mean", geo_f, lqr_f) == "lqr",
    ])
    roll_score = sum([
        wins("rmse_vel_mean", geo_r, lqr_r) == "lqr",
        wins("effort_mean", geo_r, lqr_r) == "lqr",
    ])

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if flight_score >= 2 and lqr_f["rmse_pos_mean"] < geo_f["rmse_pos_mean"]:
        print("Aerial: LQR wins on tracking accuracy for step responses.")
    elif geo_f["rmse_pos_mean"] <= lqr_f["rmse_pos_mean"]:
        print("Aerial: Geometric (Lee SO3/PD) matches or beats LQR on this linearized benchmark.")
    else:
        print("Aerial: Mixed — check per-scenario CSV.")

    if lqr_r["rmse_vel_mean"] < geo_r["rmse_vel_mean"] and geo_r["effort_mean"] <= lqr_r["effort_mean"]:
        print("Ground: LQR tracks velocity better; geometric uses less motor effort.")
    elif geo_r["rmse_vel_mean"] <= lqr_r["rmse_vel_mean"]:
        print("Ground: Geometric paper controller is competitive or better for rolling.")
    else:
        print("Ground: LQR improves velocity RMSE at higher motor effort.")

    print("\nFor CAROLLINE comparative study:")
    print("  - Use geometric for full nonlinear mission (large tilt, mode switches).")
    print("  - LQR is useful near hover / moderate rolling speeds with tuned Q/R.")
    print("  - Report both RMSE and control effort — lower error may cost more thrust.")


if __name__ == "__main__":
    main()
