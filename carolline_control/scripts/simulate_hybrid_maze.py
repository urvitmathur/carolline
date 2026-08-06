"""Hybrid roll/fly maze — obstacle-driven roll/fly transitions.

Mission:
  Roll to dead-end → roll toward platform goal →
  perception-triggered wall hop, gap fly, shaft climb → roll to target

Run from repo root:
    python carolline_control/scripts/simulate_hybrid_maze.py
    python carolline_control/scripts/simulate_hybrid_maze.py --no-viewer
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config
from carolline_control.navigation.hybrid_maze_director import HybridMazeDirector, HybridMazePhase
from carolline_control.navigation.hybrid_maze_layout import load_hybrid_maze_layout
from carolline_control.navigation.hybrid_maze_report import HybridMazeRecorder, generate_hybrid_maze_report
from carolline_control.navigation.hybrid_maze_scene import compile_hybrid_maze_scene, set_live_target_marker
from carolline_control.navigation.perception import RangePerception
from carolline_control.navigation.sensor_viz import draw_rangefinder_rays
from carolline_control.scripts.manual_teleop import update_tracking_camera
from carolline_control.sim.viewer_loop import run_passive_viewer_loop, tune_viewer_for_speed
from carolline_control.sim_estimator import build_estimator, print_estimator_mode, seed_estimator_from_sim
from carolline_control.utils.types import ControlMode


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _apply_mobility(config, raw: dict) -> None:
    mob = raw.get("mobility", {})
    for key in ("rolling_max_speed", "rolling_kp", "rolling_kd", "motor_slew_rate"):
        if key in mob:
            setattr(config, key, float(mob[key]))


def _spawn_flat(data: mujoco.MjData, spawn_xy: np.ndarray, cage_radius: float) -> None:
    x, y = float(spawn_xy[0]), float(spawn_xy[1])
    half = math.radians(55.0) * 0.5
    data.qpos[:7] = [x, y, cage_radius, math.cos(half), 0.0, math.sin(half), 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid roll/fly maze course")
    parser.add_argument(
        "--maze-config",
        default=str(REPO_ROOT / "carolline_control" / "navigation" / "hybrid_maze_config.yaml"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--no-sensor-viz", action="store_true")
    parser.add_argument("--top-down", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--plot-dir",
        default=str(REPO_ROOT / "carolline_control" / "logs" / "hybrid_maze"),
        help="Directory for timeseries CSV and report PNGs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving timeseries CSV and report charts",
    )
    args = parser.parse_args()

    layout, maze_raw = load_hybrid_maze_layout(Path(args.maze_config))
    config = load_config(args.config)
    _apply_mobility(config, maze_raw)

    layout.cage_radius = float(config.cage_radius)
    layout.hover_height_nominal = float(config.hover_height)

    config.initial_mode = ControlMode.ROLLING
    config.spawn_xy = layout.spawn_xy.copy()
    config.roll_target = layout.checkpoints["deadend"].copy()

    model_path = _resolve_model_path(maze_raw.get("model_path", config.model_path))
    model = compile_hybrid_maze_scene(model_path, layout, maze_raw)
    data = mujoco.MjData(model)

    estimator = build_estimator(model, config, slam_odom=False)
    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    roll_cfg = maze_raw.get("rolling", {})
    sim = maze_raw.get("simulation", {})
    perception_cfg = maze_raw.get("perception", {})
    settle_steps = int(sim.get("settle_steps", 400))
    duration = float(sim.get("duration", 240.0))

    _spawn_flat(data, layout.spawn_xy, config.cage_radius)
    seed_estimator_from_sim(estimator, data)
    _settle(model, data, settle_steps)

    perception = RangePerception(
        model,
        stop_distance=float(perception_cfg.get("stop_distance", 0.40)),
        max_range=float(perception_cfg.get("max_range", 6.0)),
        forward_cone_deg=float(perception_cfg.get("forward_cone_deg", 80.0)),
        fly_clearance=float(perception_cfg.get("fly_clearance", 0.90)),
        climb_clearance=float(perception_cfg.get("climb_clearance", 1.80)),
        gap_down_threshold=float(perception_cfg.get("gap_down_threshold", 1.50)),
        max_roll_height=float(perception_cfg.get("max_roll_height", 0.35)),
    )

    director = HybridMazeDirector(
        controller,
        layout,
        blocked_confirm_time=float(perception_cfg.get("blocked_confirm_time", 0.35)),
    )
    director.begin(data.time)
    recorder = HybridMazeRecorder(layout)

    dt = float(model.opt.timestep)
    substeps = max(1, int(args.substeps if args.substeps is not None else sim.get("viewer_substeps", 12)))
    control_every = int(sim.get("viewer_control_every", 4))
    sync_state_only = bool(sim.get("viewer_sync_state_only", True))
    keep_textures = bool(sim.get("viewer_keep_textures", True))
    keep_lights = bool(sim.get("viewer_keep_lights", True))
    track_distance = float(sim.get("viewer_track_distance", 11.0))

    print_estimator_mode(slam_odom=False)
    print(
        f"Hybrid maze: arena {layout.arena_half_x * 2:.0f}x{layout.arena_half_y * 2:.0f} m  "
        f"platform_z={layout.platform.top_z:.2f} m"
    )
    print(
        f"Spawn: ({layout.spawn_xy[0]:.2f}, {layout.spawn_xy[1]:.2f})  "
        f"Goal: ({layout.goal_xy[0]:.2f}, {layout.goal_xy[1]:.2f}) on platform"
    )
    print("Mission: dead-end map -> perception-driven wall/gap/climb -> goal")
    print(f"Rangefinders: {len(perception._sensor_ids)}")

    last_scan = perception.scan(data, estimator.estimate(data))
    last_mode = controller.mode_manager.mode.name
    last_status_t = -1.0

    def control_step(*, step_dt: float | None = None) -> bool:
        nonlocal last_scan, last_mode, last_status_t
        use_dt = dt if step_dt is None else step_dt
        state = estimator.estimate(data)
        last_scan = perception.scan(data, state)
        motor, cmd, mode, target = controller.compute(state, use_dt)
        running = director.update(state, mode, float(state.time), use_dt, last_scan)
        data.ctrl[:] = motor.thrusts

        if mode.name != last_mode:
            print(
                f"t={state.time:6.2f}s  MODE  {last_mode} -> {mode.name}  "
                f"phase={director.phase.value}  "
                f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})"
            )
            last_mode = mode.name

        if state.time - last_status_t >= 4.0:
            tgt = director.current_target_xy()
            dist = float(np.linalg.norm(state.position[:2] - tgt))
            print(
                f"t={state.time:6.1f}s  phase={director.phase.value}  mode={mode.name}  "
                f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})  "
                f"tgt=({tgt[0]:.2f},{tgt[1]:.2f})  dist={dist:.2f}  "
                f"range={last_scan.forward_min_m:.2f}  "
                f"wall={last_scan.wall_block} gap={last_scan.floor_gap} climb={last_scan.climb_face}  "
                f"down={last_scan.down_min_m:.2f}"
            )
            last_status_t = state.time

        marker_z = float(state.position[2]) + 0.15
        if mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.TAKEOFF, ControlMode.LANDING):
            marker_z = float(target.position[2])
        set_live_target_marker(model, data, director.current_target_xy(), marker_z)
        recorder.record(
            time=float(state.time),
            position=state.position,
            velocity=state.velocity,
            mode=mode,
            phase=director.phase.value,
            desired_position=target.position,
            scan=last_scan,
            takeoff_reason=director._last_takeoff_reason,
        )
        return running and data.time < duration

    success = False
    try:
        if args.no_viewer:
            while control_step():
                mujoco.mj_step(model, data)
            success = director.phase == HybridMazePhase.DONE
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                tune_viewer_for_speed(viewer, keep_textures=keep_textures, keep_lights=keep_lights)
                if args.top_down:
                    viewer.cam.azimuth = 90.0
                    viewer.cam.elevation = -75.0
                    viewer.cam.distance = max(layout.arena_half_x, layout.arena_half_y) * 2.6
                else:
                    viewer.cam.azimuth = float(sim.get("viewer_cam_azimuth", 125.0))
                    viewer.cam.elevation = float(sim.get("viewer_cam_elevation", -28.0))
                    viewer.cam.distance = track_distance
                update_tracking_camera(
                    viewer,
                    data.qpos[:3],
                    look_height=0.45,
                    distance=track_distance,
                )

                def on_frame(v, sim_data) -> None:
                    state = estimator.estimate(sim_data)
                    update_tracking_camera(
                        v,
                        state.position,
                        look_height=0.45,
                        distance=track_distance,
                    )
                    if not args.no_sensor_viz:
                        draw_rangefinder_rays(v, perception, sim_data, last_scan)
                    tgt = director.current_target_xy()
                    v.set_texts(
                        (
                            int(mujoco.mjtFontScale.mjFONTSCALE_150),
                            int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                            f"Phase: {director.phase.value}\n"
                            f"Mode: {controller.mode_manager.mode.name}  "
                            f"z={state.position[2]:.2f} m  range={last_scan.forward_min_m:.2f}\n"
                            f"scan wall={last_scan.wall_block} gap={last_scan.floor_gap} "
                            f"climb={last_scan.climb_face} down={last_scan.down_min_m:.2f}\n"
                            f"last_takeoff={director._last_takeoff_reason or '-'}  "
                            f"pos=({state.position[0]:.2f},{state.position[1]:.2f})  "
                            f"tgt=({tgt[0]:.2f},{tgt[1]:.2f})  HYBRID ROLL/FLY",
                            "",
                        )
                    )

                control_stride = max(1, control_every)
                control_dt = dt * control_stride

                run_passive_viewer_loop(
                    viewer,
                    model,
                    data,
                    lambda: control_step(step_dt=control_dt),
                    substeps=substeps,
                    target_fps=args.target_fps,
                    realtime=args.realtime,
                    sync_state_only=sync_state_only,
                    control_once_per_frame=False,
                    control_every=control_stride,
                    on_frame=on_frame,
                    hud_every=2,
                )
                success = director.phase == HybridMazePhase.DONE
    finally:
        pass

    final = estimator.estimate(data)
    dist_goal = float(np.linalg.norm(final.position[:2] - layout.goal_xy))
    print(
        f"Finished t={data.time:.1f}s  phase={director.phase.value}  "
        f"success={success}  dist_goal={dist_goal:.2f} m  "
        f"z={final.position[2]:.2f} m"
    )

    if not args.no_plot and recorder.rows:
        plot_dir = Path(args.plot_dir)
        if not plot_dir.is_absolute():
            plot_dir = REPO_ROOT / plot_dir
        artifacts = generate_hybrid_maze_report(recorder, layout, plot_dir, success=success)
        print(f"Report artifacts ({len(artifacts)} files) in {plot_dir.resolve()}:")
        for path in artifacts:
            print(f"  {path.resolve()}")

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
