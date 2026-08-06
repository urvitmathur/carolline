"""
Headless study: Geometric (PD/SO3) vs LQR vs MPC.

Run from repo root:
    python carolline_control/scripts/run_controller_study.py
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
from carolline_control.controllers.mpc.config_loader import flight_mpc_weights_from_config, load_mpc_config, rolling_mpc_weights_from_config
from carolline_control.controllers.mpc.flight_mpc import FlightMpcController
from carolline_control.controllers.mpc.rolling_mpc import RollingMpcController
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
from carolline_control.utils.types import ControlMode
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


def _mpc_flight_step(state, target, mpc: FlightMpcController, mixer: MotorMixer) -> np.ndarray:
    cmd = mpc.compute(state, target, ControlMode.HOVER)
    return mixer.mix(cmd.thrust, cmd.moment_body).thrusts


def _flight_trial(
    config,
    model: mujoco.MjModel,
    *,
    controller: str,
    flight_ctrl,
    duration: float,
    step_x: float,
    height: float,
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
        if controller == "geometric":
            thrusts = _geometric_step(state, target, config, flight, attitude, torque, mixer)
        elif controller == "lqr":
            thrusts = _lqr_step(state, target, flight_ctrl, mixer)
        else:
            thrusts = _mpc_flight_step(state, target, flight_ctrl, mixer)
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

    m = acc.finalize(controller=controller, domain="flight", settle_axis=0, step_target=step_x)
    return TrialRow("flight", scenario, controller, m.rmse_position, m.rmse_velocity, m.max_position_error,
                    m.max_velocity_error, m.settling_time_s, m.control_effort, m.motor_saturation_pct, m.stable)


def _rolling_trial(
    config,
    model: mujoco.MjModel,
    *,
    controller: str,
    roll_ctrl,
    duration: float,
    speed: float,
    tilt_deg: float,
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
        if controller == "geometric":
            cmd = paper.compute(state, v_des, yaw=0.0, dt=float(model.opt.timestep))
            motor = paper.allocate(state, cmd).motor
        elif controller == "lqr":
            cmd = roll_ctrl.compute(state, v_des)
            motor = roll_ctrl.allocate(state, cmd).motor
        else:
            cmd = roll_ctrl.compute(state, v_des)
            motor = roll_ctrl.allocate(state, cmd).motor
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

    m = acc.finalize(controller=controller, domain="rolling")
    return TrialRow("rolling", scenario, controller, m.rmse_position, m.rmse_velocity, m.max_position_error,
                    m.max_velocity_error, m.settling_time_s, m.control_effort, m.motor_saturation_pct, m.stable)


def _aggregate(rows: list[TrialRow], domain: str, controller: str) -> dict:
    subset = [r for r in rows if r.domain == domain and r.controller == controller]
    settle = [r.settling_s for r in subset if math.isfinite(r.settling_s)]
    return {
        "rmse_pos_mean": float(np.mean([r.rmse_pos for r in subset])),
        "rmse_vel_mean": float(np.mean([r.rmse_vel for r in subset])),
        "settle_mean": float(np.mean(settle)) if settle else float("nan"),
        "effort_mean": float(np.mean([r.effort for r in subset])),
        "sat_mean": float(np.mean([r.saturation_pct for r in subset])),
    }


def _winner(metric: str, a: dict, b: dict, c: dict, *, lower_better: bool = True) -> str:
    vals = {"geometric": a[metric], "lqr": b[metric], "mpc": c[metric]}
    if not lower_better:
        vals = {k: -v for k, v in vals.items()}
    best = min(vals, key=vals.get)
    return best


def main() -> None:
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    raw = load_raw_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    lqr_raw = load_lqr_config()
    mpc_raw = load_mpc_config()

    model_path = REPO_ROOT / config.model_path
    model = compile_model_with_markers(model_path, config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    lqr_f = FlightLqrController(config, flight_lqr_weights_from_config(lqr_raw))
    mpc_f = FlightMpcController(config, flight_mpc_weights_from_config(mpc_raw))

    rows: list[TrialRow] = []
    flight_steps = [0.5, 0.75, 1.0]
    roll_speeds = [1.0, 1.5, 2.0]
    roll_tilts = [45.0, 55.0, 65.0]
    ground_z = float(raw.get("ground_z", 0.40))

    for step in flight_steps:
        sc = f"step_{step:.2f}m"
        rows.append(_flight_trial(config, model, controller="geometric", flight_ctrl=None,
                                duration=8.0, step_x=step, height=1.0, scenario=sc))
        rows.append(_flight_trial(config, model, controller="lqr", flight_ctrl=lqr_f,
                                duration=8.0, step_x=step, height=1.0, scenario=sc))
        rows.append(_flight_trial(config, model, controller="mpc", flight_ctrl=mpc_f,
                                duration=8.0, step_x=step, height=1.0, scenario=sc))

    for speed in roll_speeds:
        for tilt in roll_tilts:
            sc = f"speed_{speed:.1f}_tilt_{tilt:.0f}"
            lqr_r = RollingLqrController(config, rolling_lqr_weights_from_config(lqr_raw))
            mpc_r = RollingMpcController(config, rolling_mpc_weights_from_config(mpc_raw))
            rows.append(_rolling_trial(config, model, controller="geometric", roll_ctrl=None,
                                       duration=6.0, speed=speed, tilt_deg=tilt, scenario=sc, ground_z=ground_z))
            rows.append(_rolling_trial(config, model, controller="lqr", roll_ctrl=lqr_r,
                                       duration=6.0, speed=speed, tilt_deg=tilt, scenario=sc, ground_z=ground_z))
            rows.append(_rolling_trial(config, model, controller="mpc", roll_ctrl=mpc_r,
                                       duration=6.0, speed=speed, tilt_deg=tilt, scenario=sc, ground_z=ground_z))

    out_dir = REPO_ROOT / "carolline_control" / "plots" / "controller_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "study_geometric_lqr_mpc.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    geo_f = _aggregate(rows, "flight", "geometric")
    lqr_f_m = _aggregate(rows, "flight", "lqr")
    mpc_f_m = _aggregate(rows, "flight", "mpc")
    geo_r = _aggregate(rows, "rolling", "geometric")
    lqr_r_m = _aggregate(rows, "rolling", "lqr")
    mpc_r_m = _aggregate(rows, "rolling", "mpc")

    print("=" * 70)
    print("HEADLESS STUDY: Geometric (PD) vs LQR vs MPC")
    print("=" * 70)
    print(f"Results: {csv_path}\n")

    print("FLIGHT (lower is better except where noted)")
    hdr = f"  {'metric':<20} {'geometric':>11} {'lqr':>11} {'mpc':>11} {'best':>8}"
    print(hdr)
    for key, label in [
        ("rmse_pos_mean", "RMSE position"),
        ("rmse_vel_mean", "RMSE velocity"),
        ("settle_mean", "Settling time"),
        ("effort_mean", "Motor effort"),
        ("sat_mean", "Saturation %"),
    ]:
        print(
            f"  {label:<20} {geo_f[key]:11.4f} {lqr_f_m[key]:11.4f} {mpc_f_m[key]:11.4f} "
            f"{_winner(key, geo_f, lqr_f_m, mpc_f_m):>8}"
        )

    print("\nROLLING (lower is better)")
    print(hdr)
    for key, label in [
        ("rmse_vel_mean", "RMSE velocity"),
        ("effort_mean", "Motor effort"),
        ("sat_mean", "Saturation %"),
    ]:
        print(
            f"  {label:<20} {geo_r[key]:11.4f} {lqr_r_m[key]:11.4f} {mpc_r_m[key]:11.4f} "
            f"{_winner(key, geo_r, lqr_r_m, mpc_r_m):>8}"
        )

    # Per-scenario win counts
    from collections import defaultdict
    pairs: dict[tuple[str, str], dict[str, TrialRow]] = defaultdict(dict)
    for r in rows:
        pairs[(r.domain, r.scenario)][r.controller] = r

    wins = {"geometric": 0, "lqr": 0, "mpc": 0}
    for (dom, _), d in pairs.items():
        if dom == "flight":
            metric = "rmse_pos"
        else:
            metric = "rmse_vel"
        best = min(d, key=lambda k: float(d[k].__dict__[metric]))
        wins[best] += 1

    print("\n" + "=" * 70)
    print("CONCLUSION (accuracy-focused scenarios)")
    print("=" * 70)
    print(f"  Scenario wins on primary metric: geometric={wins['geometric']}, lqr={wins['lqr']}, mpc={wins['mpc']}")

    if mpc_f_m["rmse_pos_mean"] <= min(geo_f["rmse_pos_mean"], lqr_f_m["rmse_pos_mean"]):
        print("  Aerial accuracy: MPC/LQR class beats geometric PD on position RMSE.")
    else:
        print("  Aerial accuracy: Geometric PD competitive on position RMSE.")

    if mpc_r_m["rmse_vel_mean"] <= min(geo_r["rmse_vel_mean"], lqr_r_m["rmse_vel_mean"]):
        print("  Rolling accuracy: MPC leads velocity RMSE.")
    elif lqr_r_m["rmse_vel_mean"] <= min(geo_r["rmse_vel_mean"], mpc_r_m["rmse_vel_mean"]):
        print("  Rolling accuracy: LQR leads velocity RMSE.")
    else:
        print("  Rolling accuracy: Geometric paper controller leads.")

    if geo_r["effort_mean"] <= min(lqr_r_m["effort_mean"], mpc_r_m["effort_mean"]):
        print("  Efficiency: Geometric uses least motor effort on rolling.")
    print("\n  Note: MPC uses finite horizon + input bounds; best for local linear regimes.")
    print("  Full CAROLLINE mission (large tilt, mode switches): prefer geometric.")


if __name__ == "__main__":
    main()
