"""
Compare paper rolling controller vs LQR rolling on flat ground velocity tracking.

Run from repo root:
    python carolline_control/scripts/compare_rolling_controllers.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.lqr.comparison_metrics import MetricsAccumulator
from carolline_control.controllers.lqr.config_loader import load_lqr_config, rolling_lqr_weights_from_config
from carolline_control.controllers.lqr.rolling_lqr import RollingLqrController
from carolline_control.controllers.rolling_controller import RollingController
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


def _tilted_qpos(spawn_xy: list[float], ground_z: float, tilt_deg: float = 55.0) -> list[float]:
    half = math.radians(tilt_deg) * 0.5
    return [
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        float(ground_z),
        float(math.cos(half)),
        0.0,
        float(math.sin(half)),
        0.0,
    ]


def _tilt_deg(state) -> float:
    bz = float(body_z_world(state.rotation)[2])
    return float(np.degrees(np.arccos(np.clip(bz, -1.0, 1.0))))


def run_trial(
    *,
    label: str,
    use_lqr: bool,
    config,
    model: mujoco.MjModel,
    duration: float,
    speed: float,
    spawn_xy: list[float],
    ground_z: float,
    lqr: RollingLqrController | None,
) -> tuple[MetricsAccumulator, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:7] = _tilted_qpos(spawn_xy, ground_z)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    paper = RollingController(config)

    acc = MetricsAccumulator(motor_min=config.motor_min, motor_max=config.motor_max)
    times: list[float] = []
    speed_hist: list[float] = []
    v_des = np.array([speed, 0.0])

    while data.time < duration:
        state = estimator.estimate(data)
        if use_lqr:
            assert lqr is not None
            cmd = lqr.compute(state, v_des)
            motor = lqr.allocate(state, cmd).motor
        else:
            cmd = paper.compute(state, v_des, yaw=0.0, dt=float(model.opt.timestep))
            motor = paper.allocate(state, cmd).motor

        data.ctrl[:] = motor.thrusts
        speed_xy = float(np.linalg.norm(state.velocity[:2]))
        acc.add(
            time=data.time,
            pos=state.position,
            vel=state.velocity,
            pos_ref=np.array([state.position[0] + speed * duration, state.position[1], state.position[2]]),
            vel_ref=np.array([speed, 0.0, 0.0]),
            motors=motor.thrusts,
            tilt_deg=_tilt_deg(state),
        )
        times.append(data.time)
        speed_hist.append(speed_xy)
        mujoco.mj_step(model, data)

    metrics = acc.finalize(controller=label, domain="rolling")
    return metrics, np.asarray(times), np.asarray(speed_hist)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paper rolling vs LQR rolling")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--lqr-config", default=str(REPO_ROOT / "carolline_control" / "lqr_config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_config(args.config)
    lqr_raw = load_lqr_config(Path(args.lqr_config))
    cmp_cfg = lqr_raw.get("comparison", {})

    duration = float(cmp_cfg.get("rolling_duration", 6.0))
    speed = float(cmp_cfg.get("rolling_speed", 1.5))
    ground_z = float(raw.get("ground_z", 0.40))
    spawn = list(config.spawn_xy)
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

    lqr = RollingLqrController(config, rolling_lqr_weights_from_config(lqr_raw))

    paper_metrics, paper_t, paper_v = run_trial(
        label="paper_rolling",
        use_lqr=False,
        config=config,
        model=model,
        duration=duration,
        speed=speed,
        spawn_xy=spawn,
        ground_z=ground_z,
        lqr=None,
    )
    lqr_metrics, lqr_t, lqr_v = run_trial(
        label="lqr_rolling",
        use_lqr=True,
        config=config,
        model=model,
        duration=duration,
        speed=speed,
        spawn_xy=spawn,
        ground_z=ground_z,
        lqr=lqr,
    )

    csv_path = out_dir / "rolling_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paper_metrics.to_row().keys()))
        writer.writeheader()
        writer.writerow(paper_metrics.to_row())
        writer.writerow(lqr_metrics.to_row())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(paper_t, paper_v, label="Paper rolling (PD + alloc)")
    ax.plot(lqr_t, lqr_v, label="LQR rolling")
    ax.axhline(speed, color="gray", linestyle="--", alpha=0.6, label="Target speed")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|v_xy| [m/s]")
    ax.set_title("Rolling velocity tracking: paper vs LQR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "rolling_speed_tracking.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    print("Rolling controller comparison")
    print(f"  Target speed: {speed:.2f} m/s for {duration:.1f} s")
    for m in (paper_metrics, lqr_metrics):
        print(
            f"  [{m.controller}] RMSE vel={m.rmse_velocity:.4f} m/s  "
            f"max vel err={m.max_velocity_error:.4f} m/s  "
            f"effort={m.control_effort:.2f}  sat={m.motor_saturation_pct:.1f}%  stable={m.stable}"
        )
    print(f"  CSV: {csv_path}")
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
