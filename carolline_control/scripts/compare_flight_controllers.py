"""
Compare geometric (Lee SO(3)) vs LQR flight controllers on a hover step response.

Run from repo root:
    python carolline_control/scripts/compare_flight_controllers.py
    python carolline_control/scripts/compare_flight_controllers.py --no-viewer
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.attitude_controller import AttitudeController
from carolline_control.controllers.flight_controller import FlightController
from carolline_control.controllers.lqr.comparison_metrics import MetricsAccumulator
from carolline_control.controllers.lqr.config_loader import flight_lqr_weights_from_config, load_lqr_config
from carolline_control.controllers.lqr.flight_lqr import FlightLqrController
from carolline_control.controllers.motor_mixer import MotorMixer
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.controllers.torque_controller import TorqueController
from carolline_control.utils.so3 import body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode, TrajectoryTarget
from carolline_control.visualization.markers import compile_model_with_markers


def _upright_hover_qpos(xy: np.ndarray, height: float) -> list[float]:
    return [float(xy[0]), float(xy[1]), float(height), 1.0, 0.0, 0.0, 0.0]


def _tilt_deg(state) -> float:
    bz = float(body_z_world(state.rotation)[2])
    return float(np.degrees(np.arccos(np.clip(bz, -1.0, 1.0))))


def _target(time: float, anchor: np.ndarray, height: float, step_x: float, step_time: float) -> TrajectoryTarget:
    pos = anchor.copy()
    if time >= step_time:
        pos[0] += step_x
    return TrajectoryTarget(
        position=np.array([pos[0], pos[1], height]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        yaw=0.0,
    )


def _geometric_step(
    state,
    target: TrajectoryTarget,
    config,
    flight: FlightController,
    attitude: AttitudeController,
    torque: TorqueController,
    mixer: MotorMixer,
) -> np.ndarray:
    cmd = flight.compute(state, target, ControlMode.HOVER)
    omega_d = attitude.compute_desired_omega(state.rotation, cmd.desired_rotation, config.kR)
    moment = torque.compute(
        state.rotation,
        cmd.desired_rotation,
        state.omega_body,
        omega_d,
        config.inertia,
        config.kR,
        config.kOmega,
    )
    motor = mixer.mix(cmd.thrust, moment)
    return motor.thrusts


def _lqr_step(state, target: TrajectoryTarget, lqr: FlightLqrController, mixer: MotorMixer) -> np.ndarray:
    cmd = lqr.compute(state, target, ControlMode.HOVER)
    motor = mixer.mix(cmd.thrust, cmd.moment_body)
    return motor.thrusts


def run_trial(
    *,
    label: str,
    use_lqr: bool,
    config,
    model: mujoco.MjModel,
    duration: float,
    anchor: np.ndarray,
    height: float,
    step_x: float,
    step_time: float,
    lqr: FlightLqrController | None,
) -> tuple[MetricsAccumulator, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:7] = _upright_hover_qpos(anchor, height)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    flight = FlightController(config)
    attitude = AttitudeController()
    torque = TorqueController()
    mixer = MotorMixer(config)

    acc = MetricsAccumulator(motor_min=config.motor_min, motor_max=config.motor_max)
    times: list[float] = []
    x_hist: list[float] = []

    dt = float(model.opt.timestep)
    while data.time < duration:
        state = estimator.estimate(data)
        target = _target(data.time, anchor, height, step_x, step_time)
        if use_lqr:
            assert lqr is not None
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
        times.append(data.time)
        x_hist.append(float(state.position[0]))
        mujoco.mj_step(model, data)

    metrics = acc.finalize(
        controller=label,
        domain="flight",
        settle_axis=0,
        step_target=step_x,
    )
    return metrics, np.asarray(times), np.asarray(x_hist)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare geometric vs LQR flight control")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--lqr-config", default=str(REPO_ROOT / "carolline_control" / "lqr_config.yaml"))
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_config(args.config)
    lqr_raw = load_lqr_config(Path(args.lqr_config))
    cmp_cfg = lqr_raw.get("comparison", {})

    duration = float(cmp_cfg.get("flight_duration", 8.0))
    step_x = float(cmp_cfg.get("flight_step_x", 0.75))
    height = float(cmp_cfg.get("hover_height", config.hover_height))
    out_dir = REPO_ROOT / cmp_cfg.get("output_dir", "carolline_control/plots/lqr_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model = compile_model_with_markers(model_path, config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    anchor = np.array([0.0, 0.0])
    lqr = FlightLqrController(config, flight_lqr_weights_from_config(lqr_raw))

    geo_metrics, geo_t, geo_x = run_trial(
        label="geometric",
        use_lqr=False,
        config=config,
        model=model,
        duration=duration,
        anchor=anchor,
        height=height,
        step_x=step_x,
        step_time=1.0,
        lqr=None,
    )
    lqr_metrics, lqr_t, lqr_x = run_trial(
        label="lqr",
        use_lqr=True,
        config=config,
        model=model,
        duration=duration,
        anchor=anchor,
        height=height,
        step_x=step_x,
        step_time=1.0,
        lqr=lqr,
    )

    csv_path = out_dir / "flight_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geo_metrics.to_row().keys()))
        writer.writeheader()
        writer.writerow(geo_metrics.to_row())
        writer.writerow(lqr_metrics.to_row())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(geo_t, geo_x, label="Geometric (Lee SO3)")
    ax.plot(lqr_t, lqr_x, label="LQR")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6, label="Step @ 1s")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("X position [m]")
    ax.set_title("Flight step response: geometric vs LQR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "flight_step_response.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print("Flight controller comparison")
    print(f"  Step: +{step_x:.2f} m in X at t=1s, hover z={height:.2f} m")
    for m in (geo_metrics, lqr_metrics):
        print(
            f"  [{m.controller}] RMSE pos={m.rmse_position:.4f} m  "
            f"RMSE vel={m.rmse_velocity:.4f} m/s  settle={m.settling_time_s:.2f} s  "
            f"sat={m.motor_saturation_pct:.1f}%  stable={m.stable}"
        )
    print(f"  CSV: {csv_path}")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
