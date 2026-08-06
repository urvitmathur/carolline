"""Autonomous roll-fly-roll navigation on a terrain course with walls.

Run from repo root:
    python carolline_control/scripts/simulate_autonomous_course.py
    python carolline_control/scripts/simulate_autonomous_course.py --no-viewer
    python carolline_control/scripts/simulate_autonomous_course.py --sensor-only
    python carolline_control/scripts/simulate_autonomous_course.py --record
    python carolline_control/scripts/simulate_autonomous_course.py --record logs/run.mp4 --no-plot
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config
from carolline_control.logging.paths import resolve_repo_path
from carolline_control.navigation.course_analysis import plot_course_tracking
from carolline_control.navigation.course_layout import load_course_layout
from carolline_control.navigation.course_logger import CourseFlightLogger
from carolline_control.navigation.course_scene import compile_course_scene, set_live_target_marker
from carolline_control.navigation.director import CoursePhase, HybridCourseDirector
from carolline_control.navigation.perception import RangePerception
from carolline_control.navigation.rolling_follower import TerrainRollingFollower
from carolline_control.navigation.sensor_viz import draw_rangefinder_rays
from carolline_control.navigation.sim_recorder import MujocoVideoRecorder
from carolline_control.scripts.manual_teleop import update_tracking_camera
from carolline_control.sim.viewer_loop import run_passive_viewer_loop, tune_viewer_for_speed
from carolline_control.sim_estimator import (
    add_sensor_only_argument,
    build_estimator,
    print_estimator_mode,
    seed_estimator_from_sim,
)
from carolline_control.terrain.heightmap import sample_terrain_height
from carolline_control.terrain.scene import apply_terrain_mobility_tuning
from carolline_control.utils.types import ControlMode


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _spawn_on_course(
    data: mujoco.MjData,
    *,
    spawn_xy: np.ndarray,
    cage_radius: float,
    heights: np.ndarray,
    layout,
    random_orientation: bool,
) -> None:
    from carolline_control.scripts.manual_teleop import _random_ground_qpos

    x, y = float(spawn_xy[0]), float(spawn_xy[1])
    surface_z = sample_terrain_height(x, y, heights, layout)
    center_z = surface_z + cage_radius
    if random_orientation:
        qpos = _random_ground_qpos([x, y], center_z)
    else:
        half = math.radians(55.0) * 0.5
        qpos = [x, y, center_z, math.cos(half), 0.0, math.sin(half), 0.0]
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous CAROLLINE course navigation")
    parser.add_argument(
        "--course-config",
        default=str(REPO_ROOT / "carolline_control" / "navigation" / "course_config.yaml"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--substeps",
        type=int,
        default=20,
        help="Physics steps per viewer frame",
    )
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target viewer refresh rate")
    parser.add_argument(
        "--no-sensor-viz",
        action="store_true",
        help="Disable rangefinder ray overlay in the viewer",
    )
    parser.add_argument(
        "--log",
        default="carolline_control/logs/course_flight_log.csv",
        help="CSV flight log with actual/desired tracking data",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip actual vs desired charts")
    parser.add_argument(
        "--plot-dir",
        default="carolline_control/plots/course",
        help="Directory for generated tracking charts",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="carolline_control/logs/course_recording.mp4",
        default=None,
        metavar="MP4",
        help="Record simulation video (default: carolline_control/logs/course_recording.mp4)",
    )
    parser.add_argument("--record-fps", type=int, default=10, help="Video frame rate (default 10)")
    add_sensor_only_argument(parser)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    course_layout, course_raw = load_course_layout(Path(args.course_config))
    config = load_config(args.config)
    apply_terrain_mobility_tuning(config, course_raw)

    model_path = _resolve_model_path(course_raw.get("model_path", config.model_path))
    model, terrain_layout, heights, _range_names = compile_course_scene(
        model_path,
        course_layout,
        course_raw,
    )
    data = mujoco.MjData(model)

    estimator = build_estimator(model, config, sensor_only=args.sensor_only)
    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    spawn_cfg = course_raw.get("spawn", {})
    random_orientation = bool(spawn_cfg.get("random_orientation", False))
    settle_steps = int(spawn_cfg.get("settle_steps", 400))
    config.spawn_xy = course_layout.spawn_xy.copy()
    config.roll_target = course_layout.checkpoints[0].copy() if course_layout.checkpoints else course_layout.goal_xy.copy()

    _spawn_on_course(
        data,
        spawn_xy=course_layout.spawn_xy,
        cage_radius=config.cage_radius,
        heights=heights,
        layout=terrain_layout,
        random_orientation=random_orientation,
    )
    seed_estimator_from_sim(estimator, data)
    _settle(model, data, settle_steps)

    def terrain_z_at(x: float, y: float) -> float:
        return float(sample_terrain_height(x, y, heights, terrain_layout))

    follower = TerrainRollingFollower(course_layout, config, terrain_z_at=terrain_z_at)
    director = HybridCourseDirector(controller, course_layout, follower, terrain_z_at)
    perception_cfg = course_raw.get("perception", {})
    perception = RangePerception(
        model,
        stop_distance=course_layout.stop_distance,
        max_roll_height=course_layout.max_roll_height,
        fly_clearance=course_layout.fly_clearance,
        max_range=float(perception_cfg.get("max_range", 8.0)),
        forward_cone_deg=float(perception_cfg.get("forward_cone_deg", 60.0)),
    )

    duration = float(course_raw.get("simulation", {}).get("duration", 300.0))
    dt = float(model.opt.timestep)
    director.begin(data.time)

    log_path = resolve_repo_path(REPO_ROOT, args.log)
    logger = CourseFlightLogger(log_path)
    plot_dir = resolve_repo_path(REPO_ROOT, args.plot_dir)
    record_path = resolve_repo_path(REPO_ROOT, args.record) if args.record else None
    recorder: MujocoVideoRecorder | None = None
    if record_path is not None:
        recorder = MujocoVideoRecorder(
            model,
            record_path,
            fps=int(args.record_fps),
        )

    print_estimator_mode(sensor_only=args.sensor_only)
    print(f"Course: {len(course_layout.checkpoints)} checkpoints, {len(course_layout.walls)} walls")
    print(f"Rangefinder sensors: {len(perception._sensor_ids)}")
    print(f"Rolling cruise: {follower.speed:.2f} m/s  max: {config.rolling_max_speed:.2f} m/s")
    print(f"Flight log: {logger.path}")
    if recorder is not None:
        print(f"Recording: {record_path} @ {args.record_fps} fps")
    if not args.no_plot:
        print(f"Charts: {plot_dir}")

    last_scan = perception.scan(data, estimator.estimate(data))
    controller.planner.rolling_velocity = lambda state: follower.velocity_command_xy(
        state,
        director.mission_scan(last_scan, state),
    )
    substeps = max(1, int(args.substeps))
    track_distance = 8.0
    record_interval = 1.0 / max(1, int(args.record_fps))
    last_record_time = -record_interval

    def control_step(*, step_dt: float | None = None) -> bool:
        nonlocal last_scan
        use_dt = dt if step_dt is None else step_dt
        state = estimator.estimate(data)
        last_scan = perception.scan(data, state)
        mission_scan = director.mission_scan(last_scan, state)
        motor, cmd, mode, target = controller.compute(state, use_dt)
        running = director.update(state, mode, state.time, use_dt, last_scan)
        data.ctrl[:] = motor.thrusts

        target_xy = director.current_target_xy()
        marker_z = terrain_z_at(float(target_xy[0]), float(target_xy[1])) + config.cage_radius + 0.12
        set_live_target_marker(model, data, target_xy, marker_z)

        cmd_xy = follower.velocity_command_xy(state, mission_scan)
        cmd_speed = float(np.linalg.norm(cmd_xy))
        logger.log_course(
            state,
            mode,
            cmd,
            motor,
            target,
            controller.last_diagnostics,
            phase=director.phase.value,
            checkpoint=follower.checkpoint_index,
            cmd_speed=cmd_speed,
            forward_min_m=float(last_scan.forward_min_m),
            mission_blocked=bool(mission_scan.blocked),
        )
        return running and data.time < duration

    def capture_frame(state_position: np.ndarray, viewer_cam: mujoco.MjvCamera | None = None) -> None:
        nonlocal last_record_time
        if recorder is None:
            return
        if data.time - last_record_time < record_interval:
            return
        last_record_time = data.time
        if viewer_cam is not None:
            recorder.sync_camera_from_viewer(viewer_cam)
        else:
            recorder.update_tracking_camera(
                state_position,
                look_height=0.35,
                distance=track_distance,
            )
        recorder.capture(data)

    success = False
    try:
        if args.no_viewer:
            while control_step():
                mujoco.mj_step(model, data)
                state = estimator.estimate(data)
                capture_frame(state.position)
            success = director.phase == CoursePhase.DONE
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                tune_viewer_for_speed(viewer)
                viewer.cam.distance = track_distance
                viewer.cam.azimuth = 130.0
                viewer.cam.elevation = -22.0
                update_tracking_camera(
                    viewer,
                    data.qpos[:3],
                    look_height=0.35,
                    distance=track_distance,
                )

                def on_frame(v, _data) -> None:
                    state = estimator.estimate(_data)
                    mission_scan = director.mission_scan(last_scan, state)
                    update_tracking_camera(v, state.position, look_height=0.35)
                    if not args.no_sensor_viz:
                        draw_rangefinder_rays(v, perception, _data, last_scan)
                    cmd_xy = follower.velocity_command_xy(state, mission_scan)
                    cmd_speed = float(np.linalg.norm(cmd_xy))
                    act_speed = float(np.linalg.norm(state.velocity[:2]))
                    ray_hint = "red=blocked  yellow=mid  green=clear"
                    v.set_texts(
                        (
                            int(mujoco.mjtFontScale.mjFONTSCALE_150),
                            int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                            f"Phase: {director.phase.value}  Mode: {controller.mode_manager.mode.name}\n"
                            f"Checkpoint: {follower.checkpoint_index}/{len(course_layout.checkpoints)}  "
                            f"range: {last_scan.forward_min_m:.2f}m  "
                            f"raw_blocked={last_scan.blocked} mission_blocked={mission_scan.blocked}\n"
                            f"cmd={cmd_speed:.2f} m/s  speed={act_speed:.2f} m/s  "
                            f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})\n"
                            f"Sensors: {ray_hint}",
                            "",
                        )
                    )
                    capture_frame(state.position, v.cam)

                frame_dt = dt * substeps
                run_passive_viewer_loop(
                    viewer,
                    model,
                    data,
                    lambda: control_step(step_dt=frame_dt),
                    substeps=substeps,
                    target_fps=args.target_fps,
                    realtime=args.realtime,
                    sync_state_only=args.no_sensor_viz,
                    control_once_per_frame=True,
                    on_frame=on_frame,
                    hud_every=2,
                )
                success = director.phase == CoursePhase.DONE
    finally:
        logger.close()
        if recorder is not None:
            recorder.close()

    print(f"Finished t={data.time:.1f}s  phase={director.phase.value}  success={success}")
    print(f"Log saved: {logger.path}")
    if recorder is not None:
        print(f"Recording saved: {record_path}  ({recorder.frame_count} frames)")

    if not args.no_plot:
        try:
            saved = plot_course_tracking(logger.path, plot_dir)
            for path in saved:
                print(f"Chart: {path}")
        except Exception as exc:
            print(f"Could not generate charts: {exc}")

    if not success and director.phase != CoursePhase.DONE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
