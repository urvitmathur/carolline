"""CAROLLINE MuJoCo simulation entry point."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.logging.logger import Logger
from carolline_control.logging.pipeline_tracer import PipelineAbort, PipelineTracer
from carolline_control.plots.trajectory_tracking import plot_actual_vs_desired
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


def _set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, qpos: list[float]) -> None:
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _random_ground_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    """Random tilt on the ground (CAROLLINE cage rolling start)."""
    tilt_deg = random.uniform(35.0, 85.0)
    heading = random.uniform(0.0, 2.0 * np.pi)
    axis = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
    half = np.radians(tilt_deg) * 0.5
    qw = float(np.cos(half))
    qvec = axis * float(np.sin(half))
    return [float(spawn_xy[0]), float(spawn_xy[1]), ground_z, qw, qvec[0], qvec[1], qvec[2]]


def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE MuJoCo simulation")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Headless simulation")
    parser.add_argument(
        "--debug-pipeline",
        action="store_true",
        help="Trace control pipeline (mode change, saturation, >10%% value change)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip actual vs desired plot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initial orientation")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    raw = load_raw_config(args.config)
    config = load_config(args.config)

    model = compile_model_with_markers(config.model_path, config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])
    data = mujoco.MjData(model)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    pipeline_tracer = (
        PipelineTracer(motor_min=config.motor_min, motor_max=config.motor_max)
        if args.debug_pipeline
        else None
    )
    controller = CarollineController(config, pipeline_tracer=pipeline_tracer)

    repo_root = Path(__file__).parent.parent
    log_path = repo_root / raw.get("log_path", "carolline_control/logs/flight_log.csv")
    plot_path = repo_root / raw.get("plot_path", "carolline_control/plots/actual_vs_desired.png")
    logger = Logger(log_path)

    mission = raw.get("mission", {})
    spawn_xy = mission.get("spawn_xy", [0.0, 0.0])
    ground_z = float(raw.get("ground_z", 0.40))
    if raw.get("random_initial_orientation", False):
        initial_qpos = _random_ground_qpos(spawn_xy, ground_z)
    else:
        initial_qpos = raw.get("initial_qpos")
    if initial_qpos:
        _set_initial_pose(model, data, initial_qpos)
        print(f"Initial pose: pos=({initial_qpos[0]:.2f}, {initial_qpos[1]:.2f}, {initial_qpos[2]:.2f})")

    duration = float(raw.get("sim_duration", 120.0))
    idle_exit_delay = float(raw.get("idle_exit_delay", 3.0))
    viewer_realtime = bool(raw.get("viewer_realtime", True))
    viewer_speed = max(float(raw.get("viewer_speed", 1.0)), 0.05)
    viewer_hold_after_idle = float(raw.get("viewer_hold_after_idle", 25.0))
    dt = float(model.opt.timestep)

    print(f"Loaded model: {config.model_path}")
    print(f"Mass: {config.mass:.3f} kg, hover thrust ~ {config.mass * config.gravity:.2f} N")
    print(f"Motors: [{config.motor_min:.1f}, {config.motor_max:.1f}] N (bidirectional)")
    print(f"Mission: roll to {config.roll_target}, fly {config.leg_distance} m/cardinal, land at spawn")
    print(
        f"Flight pacing: segment={config.segment_duration:.1f}s, "
        f"waypoint dwell={config.waypoint_dwell_time:.1f}s, "
        f"hover before flight={config.hover_before_flight_time:.1f}s"
    )
    if not args.no_viewer and viewer_realtime:
        print(f"Viewer: realtime at {viewer_speed:.2f}x (hold {viewer_hold_after_idle:.0f}s after mission)")
    print("Viewer markers: green=spawn, red=roll target, blue=flight waypoints")
    print(f"Initial mode: {controller.mode_manager.mode.name} (transitions logged to {log_path})")
    if args.debug_pipeline:
        print("Pipeline debug enabled: events on mode change, motor saturation, or >10% control delta.")

    last_logged_mode = controller.mode_manager.mode.name
    idle_since: float | None = None

    def control_step() -> bool:
        nonlocal last_logged_mode, idle_since
        state = estimator.estimate(data)
        try:
            motor, cmd, mode, target = controller.compute(state, dt)
        except PipelineAbort:
            return False
        data.ctrl[:] = motor.thrusts
        logger.log(state, mode, cmd, motor, target, diagnostics=controller.last_diagnostics)
        if not args.debug_pipeline and mode.name != last_logged_mode:
            print(f"t={state.time:.2f}s  mode: {last_logged_mode} -> {mode.name}  pos=({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})")
            last_logged_mode = mode.name
        if mode == ControlMode.IDLE:
            if idle_since is None:
                idle_since = state.time
            elif state.time - idle_since > idle_exit_delay:
                return False
        else:
            idle_since = None
        return True

    def run_loop(step_fn) -> None:
        while data.time < duration:
            if not control_step():
                break
            step_fn()

    if args.no_viewer:
        run_loop(lambda: mujoco.mj_step(model, data))
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_anchor = time.perf_counter()
            sim_anchor = data.time
            mission_finished_at: float | None = None

            while viewer.is_running():
                sim_active = data.time < duration and mission_finished_at is None
                if sim_active:
                    if not control_step():
                        mission_finished_at = time.perf_counter()
                    else:
                        mujoco.mj_step(model, data)
                        if viewer_realtime:
                            sim_elapsed = data.time - sim_anchor
                            wall_elapsed = time.perf_counter() - wall_anchor
                            delay = (sim_elapsed / viewer_speed) - wall_elapsed
                            if delay > 0.0:
                                time.sleep(delay)
                else:
                    if mission_finished_at is None:
                        mission_finished_at = time.perf_counter()
                    if time.perf_counter() - mission_finished_at >= viewer_hold_after_idle:
                        break
                    time.sleep(0.05)
                viewer.sync()

    logger.close()
    print(f"Final mode: {controller.mode_manager.mode.name}")
    print(f"Simulation complete. Log: {log_path}")

    if not args.no_plot:
        try:
            saved = plot_actual_vs_desired(log_path, plot_path)
            print(f"Trajectory plot: {saved}")
        except Exception as exc:
            print(f"Could not generate plot: {exc}")


if __name__ == "__main__":
    main()
