"""
Rolling ramp analysis — simulate and visualize cage motion on an inclined course.

Course layout (along +X):
    flat approach → ramp up → flat plateau → ramp down → flat exit → return to start

Does not modify the main CAROLLINE mission pipeline or state machine.

Usage (from repo root):
    python carolline_control/scripts/rolling_ramp_analysis.py
    python carolline_control/scripts/rolling_ramp_analysis.py --no-viewer --seed 7
    python carolline_control/scripts/rolling_ramp_analysis.py --record
    python carolline_control/scripts/rolling_ramp_analysis.py --record carolline_control/plots/rolling_ramp/run.mp4
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
from carolline_control.logging.paths import resolve_repo_path
from carolline_control.navigation.sim_recorder import MujocoVideoRecorder
from carolline_control.rolling_ramp.path import MissionPhase, RampGeometry, RampPathPlanner
from carolline_control.rolling_ramp.plots import plot_rolling_ramp_analysis
from carolline_control.rolling_ramp.scene import compile_ramp_scene
from carolline_control.scripts.manual_teleop import update_tracking_camera
from carolline_control.sim.viewer_loop import run_passive_viewer_loop, tune_viewer_for_speed
from carolline_control.utils.types import ControlMode


def load_ramp_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_rolling_overrides(config, raw: dict) -> None:
    """Apply rolling-ramp mobility overrides from YAML."""
    mob = raw.get("mobility", raw)
    for key in (
        "rolling_max_speed",
        "rolling_kp",
        "rolling_kd",
        "roll_position_tolerance",
        "motor_slew_rate",
        "ground_max_unload_fraction",
    ):
        if key in mob:
            setattr(config, key, float(mob[key]))
    for key in ("rolling_omega_kp", "ground_omega_kp", "ground_omega_weights"):
        if key in mob:
            setattr(config, key, np.asarray(mob[key], dtype=float))

def _spawn_qpos(
    geometry: RampGeometry,
    spawn_s: float,
    *,
    random_tilt: bool,
    tilt_deg: float = 58.0,
) -> list[float]:
    """Place the cage on the path with a controlled rolling tilt."""
    pos = geometry.position_at_s(spawn_s)
    if random_tilt:
        import random

        tilt = random.uniform(35.0, 85.0)
        heading = random.uniform(0.0, 2.0 * np.pi)
    else:
        tilt = tilt_deg
        heading = 0.0
    axis = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
    half = np.radians(tilt) * 0.5
    qw = float(np.cos(half))
    qvec = axis * float(np.sin(half))
    return [float(pos[0]), float(pos[1]), float(pos[2]), qw, qvec[0], qvec[1], qvec[2]]


def _snap_to_path(model: mujoco.MjModel, data: mujoco.MjData, geometry: RampGeometry, spawn_s: float) -> None:
    """Align pose to the path surface after passive settle."""
    pos = geometry.position_at_s(spawn_s)
    data.qpos[0:3] = pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    """Passive settle with motors off so contacts stabilize before recording."""
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)

def run_simulation(
    *,
    raw: dict,
    no_viewer: bool,
    seed: int | None,
    substeps: int | None = None,
    target_fps: float | None = None,
    realtime: bool = False,
    record_path: Path | None = None,
    record_fps: int | None = None,
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
    planner_raw = raw.get("planner", {})
    ramp_planner = RampPathPlanner(
        geometry,
        config,
        lookahead=float(planner_raw.get("lookahead", raw.get("lookahead", 0.35))),
        max_carrot=float(planner_raw.get("max_carrot", 0.14)),
        home_s=float(planner_raw.get("home_s", raw.get("spawn_s", 0.05))),
        path_kp=float(planner_raw.get("path_kp", raw.get("path_kp", 0.55))),
        path_kd=float(planner_raw.get("path_kd", raw.get("path_kd", 0.9))),
        cross_kp=float(planner_raw.get("cross_kp", raw.get("cross_kp", 1.0))),
        cross_kd=float(planner_raw.get("cross_kd", raw.get("cross_kd", 0.7))),
        velocity_filter_tau=float(planner_raw.get("velocity_filter_tau", 0.16)),
        done_settle_time=float(planner_raw.get("done_settle_time", 0.65)),
        home_arrival_speed=float(planner_raw.get("home_arrival_speed", 0.07)),
        end_approach_length=float(planner_raw.get("end_approach_length", 0.85)),
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

    spawn_s = float(raw.get("spawn_s", 0.05))
    qpos = _spawn_qpos(
        geometry,
        spawn_s,
        random_tilt=bool(raw.get("random_initial_orientation", False)),
        tilt_deg=float(raw.get("spawn_tilt_deg", 58.0)),
    )
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _snap_to_path(model, data, geometry, spawn_s)

    settle_steps = int(raw.get("settle_steps", 120))
    record_warmup = float(raw.get("record_warmup", 1.0))
    if settle_steps > 0:
        _settle(model, data, settle_steps)
        mujoco.mj_forward(model, data)

    ramp_planner.reset()

    duration = float(raw.get("sim_duration", 240.0))
    dt = float(model.opt.timestep)
    viewer_substeps = int(substeps if substeps is not None else raw.get("viewer_substeps", 8))
    viewer_control_every = int(raw.get("viewer_control_every", 5))
    viewer_target_fps = float(target_fps if target_fps is not None else raw.get("viewer_target_fps", 30.0))
    sync_state_only = bool(raw.get("viewer_sync_state_only", True))
    viewer_keep_textures = bool(raw.get("viewer_keep_textures", True))
    log_decimation = 1 if no_viewer else int(raw.get("viewer_log_decimation", 4))
    log_counter = 0
    track_distance = float(raw.get("viewer_track_distance", 9.0))
    cam_azimuth = float(raw.get("viewer_cam_azimuth", 125.0))
    cam_elevation = float(raw.get("viewer_cam_elevation", -22.0))
    look_height = float(raw.get("viewer_track_look_height", 0.35))

    recorder: MujocoVideoRecorder | None = None
    record_interval = 0.0
    last_record_time = -1.0
    if record_path is not None:
        fps = int(record_fps if record_fps is not None else raw.get("record_fps", 30))
        recorder = MujocoVideoRecorder(
            model,
            record_path,
            width=int(raw.get("record_width", 1280)),
            height=int(raw.get("record_height", 720)),
            fps=fps,
        )
        record_interval = 1.0 / max(1, fps)
        last_record_time = -record_interval

    def maybe_capture_frame(*, viewer_cam: mujoco.MjvCamera | None = None) -> None:
        nonlocal last_record_time
        if recorder is None:
            return
        if data.time < record_warmup:
            return
        if data.time - last_record_time < record_interval:
            return
        last_record_time = data.time
        if viewer_cam is not None:
            recorder.sync_camera_from_viewer(viewer_cam)
        else:
            recorder.update_tracking_camera(
                data.qpos[:3],
                look_height=look_height,
                distance=track_distance,
                azimuth=cam_azimuth,
                elevation=cam_elevation,
            )
        recorder.capture(data)

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

    def control_step(*, force_log: bool = False, step_dt: float | None = None) -> bool:
        nonlocal log_counter
        use_dt = dt if step_dt is None else step_dt
        state = estimator.estimate(data)
        if state.position[2] < -1.5 or state.position[2] > 2.5:
            return False
        motor, cmd, mode, _ = controller.compute(state, use_dt)
        if ramp_planner.hold_requested:
            controller.mode_manager.request_roll_hold(True)
        else:
            controller.mode_manager.request_roll_hold(False)
        target = ramp_planner.target(state)
        diagnostics = controller.last_diagnostics
        diagnostics.position_error = state.position - target.position
        diagnostics.velocity_error = state.velocity - target.velocity
        log_counter += 1
        if force_log or log_counter >= log_decimation:
            log_step(state, target, cmd, motor, diagnostics, ramp_planner.phase.value, ramp_planner.s_desired)
            log_counter = 0
        data.ctrl[:] = motor.thrusts
        return ramp_planner.phase != MissionPhase.DONE

    def run_loop(step_fn) -> None:
        while data.time < duration:
            if not control_step():
                break
            step_fn()
            maybe_capture_frame()

    print(f"Ramp course length: {geometry.total_length:.2f} m  rise: {geometry.rise:.2f} m")
    print(f"Log: {log_path}")
    if recorder is not None:
        print(f"Recording: {record_path} @ {recorder.fps} fps")
    if not no_viewer:
        control_hz = viewer_target_fps * viewer_substeps / max(1, viewer_control_every)
        print(
            f"Viewer: {viewer_substeps} substeps/frame, control ~{control_hz:.0f} Hz, "
            f"target {viewer_target_fps:.0f} FPS"
            + (" (realtime)" if realtime else "")
        )

    try:
        if no_viewer:
            run_loop(lambda: mujoco.mj_step(model, data))
        else:
            control_stride = max(1, viewer_control_every)
            control_dt = dt * control_stride
            with mujoco.viewer.launch_passive(model, data) as viewer:
                tune_viewer_for_speed(viewer, keep_textures=viewer_keep_textures)
                viewer.cam.azimuth = cam_azimuth
                viewer.cam.elevation = cam_elevation
                update_tracking_camera(
                    viewer,
                    data.qpos[:3],
                    look_height=look_height,
                    distance=track_distance,
                )

                def on_frame(v, sim_data: mujoco.MjData) -> None:
                    update_tracking_camera(
                        v,
                        sim_data.qpos[:3],
                        look_height=look_height,
                        distance=track_distance,
                    )
                    if recorder is not None:
                        maybe_capture_frame(viewer_cam=v.cam)

                run_passive_viewer_loop(
                    viewer,
                    model,
                    data,
                    lambda: control_step(step_dt=control_dt),
                    substeps=viewer_substeps,
                    target_fps=viewer_target_fps,
                    realtime=realtime,
                    sync_state_only=sync_state_only,
                    control_once_per_frame=False,
                    control_every=control_stride,
                    on_frame=on_frame,
                    should_continue=lambda: data.time < duration,
                )
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Recording saved: {record_path}  ({recorder.frame_count} frames)")

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
    parser.add_argument("--realtime", action="store_true", help="Cap sim to wall clock")
    parser.add_argument("--substeps", type=int, default=None, help="Physics steps per viewer frame")
    parser.add_argument("--target-fps", type=float, default=None, help="Target viewer refresh rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot-only", action="store_true", help="Generate plots from existing log")
    parser.add_argument(
        "--record",
        nargs="?",
        const="carolline_control/plots/rolling_ramp/rolling_ramp.mp4",
        default=None,
        metavar="MP4",
        help="Record MP4 while sim runs (default: carolline_control/plots/rolling_ramp/rolling_ramp.mp4)",
    )
    parser.add_argument("--record-fps", type=int, default=None, help="Video frame rate (default: 30)")
    args = parser.parse_args()

    raw = load_ramp_config(Path(args.config))
    log_path = REPO_ROOT / raw.get("log_path", "carolline_control/logs/rolling_ramp_log.csv")
    plot_dir = REPO_ROOT / raw.get("plot_dir", "carolline_control/plots/rolling_ramp")
    record_path = (
        resolve_repo_path(REPO_ROOT, args.record)
        if args.record
        else None
    )

    if not args.plot_only:
        seed = args.seed if args.seed is not None else raw.get("seed")
        log_path = run_simulation(
            raw=raw,
            no_viewer=args.no_viewer,
            seed=seed,
            substeps=args.substeps,
            target_fps=args.target_fps,
            realtime=args.realtime,
            record_path=record_path,
            record_fps=args.record_fps,
        )

    out = plot_rolling_ramp_analysis(log_path, plot_dir)
    print(f"Charts saved to: {out}")
    for png in sorted(out.glob("*.png")):
        print(f"  {png.name}")


if __name__ == "__main__":
    main()
