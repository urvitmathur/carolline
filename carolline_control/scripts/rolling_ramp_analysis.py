"""
Rolling ramp analysis — simulate and visualize cage motion on an inclined course.

Course layout (along +X):
    flat approach → ramp up → flat plateau → ramp down → flat exit → return to start

Does not modify the main CAROLLINE mission pipeline or state machine.

Usage (from repo root):
    python carolline_control/scripts/rolling_ramp_analysis.py
    python carolline_control/scripts/rolling_ramp_analysis.py --no-viewer --seed 7
    python carolline_control/scripts/rolling_ramp_analysis.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.logging.logger import Logger
from carolline_control.main import _random_ground_qpos
from carolline_control.rolling_ramp.path import MissionPhase, RampGeometry, RampPathPlanner
from carolline_control.rolling_ramp.plots import plot_rolling_ramp_analysis
from carolline_control.rolling_ramp.scene import compile_ramp_scene
from carolline_control.utils.types import ControlMode


def load_ramp_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_rolling_overrides(config, raw: dict) -> None:
    for key in ("rolling_max_speed", "rolling_kp", "rolling_kd", "roll_position_tolerance"):
        if key in raw:
            setattr(config, key, float(raw[key]))


def run_simulation(
    *,
    raw: dict,
    no_viewer: bool,
    seed: int | None,
) -> Path:
    ctrl_path = REPO_ROOT / "carolline_control" / "config.yaml"
    config = load_config(ctrl_path)
    apply_rolling_overrides(config, raw)

    ramp_raw = raw.get("ramp", {})
    geometry = RampGeometry(
        angle_deg=float(ramp_raw.get("angle_deg", 8.0)),
        flat_start=float(ramp_raw.get("flat_start", 2.0)),
        ramp_length=float(ramp_raw.get("ramp_length", 3.5)),
        flat_top=float(ramp_raw.get("flat_top", 4.0)),
        flat_end=float(ramp_raw.get("flat_end", 2.0)),
        width=float(ramp_raw.get("width", 2.5)),
        y_center=float(ramp_raw.get("y_center", 0.0)),
        cage_radius=float(config.cage_radius),
    )

    base_scene = REPO_ROOT / raw.get("model_path", "mujoco_menagerie-main/skydio_x2/scene.xml")
    model = compile_ramp_scene(base_scene, geometry)
    data = mujoco.MjData(model)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    ramp_planner = RampPathPlanner(
        geometry,
        config,
        lookahead=float(raw.get("lookahead", 0.35)),
        path_kp=float(raw.get("path_kp", 0.55)),
        path_kd=float(raw.get("path_kd", 0.9)),
        cross_kp=float(raw.get("cross_kp", 1.0)),
        cross_kd=float(raw.get("cross_kd", 0.7)),
    )
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.ROLLING
    controller.planner.rolling_velocity = ramp_planner.velocity_command_xy
    controller.planner.rolling_target_position = lambda s: ramp_planner.target(s).position

    def _frozen_mode_update(state, dt, *, mission_complete=False):
        ramp_planner.update(state, dt)
        if ramp_planner.phase == MissionPhase.DONE:
            return ControlMode.IDLE
        return ControlMode.ROLLING

    controller.mode_manager.update = _frozen_mode_update

    log_path = REPO_ROOT / raw.get("log_path", "carolline_control/logs/rolling_ramp_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)

    qpos = _random_ground_qpos(
        [float(raw.get("spawn_xy", [0.5, 0.0])[0]), float(raw.get("spawn_xy", [0.5, 0.0])[1])],
        float(geometry.position_at_s(0.05)[2]),
    )
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    duration = float(raw.get("sim_duration", 240.0))
    dt = float(model.opt.timestep)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = log_path.open("w", newline="", encoding="utf-8")
    header = Logger.HEADER + ["phase", "s_desired"]
    writer = csv.writer(csv_file)
    writer.writerow(header)

    def log_step(state, target, cmd, motor, diagnostics, phase, s_des):
        rpy_deg = np.degrees(diagnostics.euler_rpy)
        des_rpy_deg = np.degrees(diagnostics.des_euler_rpy)
        writer.writerow(
            [
                state.time,
                ControlMode.ROLLING.name,
                *state.position.tolist(),
                *state.velocity.tolist(),
                *target.position.tolist(),
                *target.velocity.tolist(),
                *diagnostics.position_error.tolist(),
                *diagnostics.velocity_error.tolist(),
                *rpy_deg.tolist(),
                *des_rpy_deg.tolist(),
                diagnostics.body_z_up,
                diagnostics.tilt_deg,
                cmd.thrust,
                *diagnostics.moment_body.tolist(),
                *diagnostics.e_R.tolist(),
                *diagnostics.omega_d_body.tolist(),
                *motor.thrusts.tolist(),
                diagnostics.motor_spread,
                int(diagnostics.motor_saturated),
                int(state.on_ground),
                *state.quaternion.tolist(),
                phase,
                s_des,
            ]
        )

    def control_step() -> bool:
        state = estimator.estimate(data)
        if state.position[2] < -1.5 or state.position[2] > 2.5:
            return False
        motor, cmd, mode, _ = controller.compute(state, dt)
        target = ramp_planner.target(state)
        diagnostics = controller.last_diagnostics
        diagnostics.position_error = state.position - target.position
        diagnostics.velocity_error = state.velocity - target.velocity
        log_step(state, target, cmd, motor, diagnostics, ramp_planner.phase.value, ramp_planner.s_desired)
        data.ctrl[:] = motor.thrusts
        return ramp_planner.phase != MissionPhase.DONE

    def run_loop(step_fn) -> None:
        while data.time < duration:
            if not control_step():
                break
            step_fn()

    print(f"Ramp course length: {geometry.total_length:.2f} m  rise: {geometry.rise:.2f} m")
    print(f"Log: {log_path}")

    if no_viewer:
        run_loop(lambda: mujoco.mj_step(model, data))
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                if not control_step():
                    break
                mujoco.mj_step(model, data)
                viewer.sync()

    csv_file.close()
    print(f"Simulation finished at t={data.time:.1f}s  phase={ramp_planner.phase.value}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling ramp analysis and visualization")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "rolling_ramp_config.yaml"),
        help="Rolling ramp YAML config",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Headless simulation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot-only", action="store_true", help="Generate plots from existing log")
    args = parser.parse_args()

    raw = load_ramp_config(Path(args.config))
    log_path = REPO_ROOT / raw.get("log_path", "carolline_control/logs/rolling_ramp_log.csv")
    plot_dir = REPO_ROOT / raw.get("plot_dir", "carolline_control/plots/rolling_ramp")

    if not args.plot_only:
        seed = args.seed if args.seed is not None else raw.get("seed")
        log_path = run_simulation(raw=raw, no_viewer=args.no_viewer, seed=seed)

    out = plot_rolling_ramp_analysis(log_path, plot_dir)
    print(f"Charts saved to: {out}")
    for png in sorted(out.glob("*.png")):
        print(f"  {png.name}")


if __name__ == "__main__":
    main()
