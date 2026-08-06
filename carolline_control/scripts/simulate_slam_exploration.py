"""SLAM exploration: map an unknown obstacle field and navigate to a goal.

Run from repo root:
    python carolline_control/scripts/simulate_slam_exploration.py
    python carolline_control/scripts/simulate_slam_exploration.py --no-viewer
    python carolline_control/scripts/simulate_slam_exploration.py --oracle
    python carolline_control/scripts/simulate_slam_exploration.py --save-map logs/slam_map.png
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
from carolline_control.navigation.exploration_director import ExplorationDirector, ExplorationPhase
from carolline_control.navigation.exploration_scene import compile_exploration_scene
from carolline_control.navigation.map_viz import MapVisualizer
from carolline_control.navigation.perception import RangePerception
from carolline_control.navigation.sensor_viz import draw_rangefinder_rays
from carolline_control.navigation.slam_stack import SlamNavigator, load_slam_config
from carolline_control.navigation.waypoint_follower import WaypointFollower
from carolline_control.scripts.manual_teleop import update_tracking_camera
from carolline_control.sim.viewer_loop import run_passive_viewer_loop, tune_viewer_for_speed
from carolline_control.sim_estimator import (
    add_slam_nav_argument,
    build_estimator,
    print_estimator_mode,
    seed_estimator_from_sim,
)
from carolline_control.utils.types import ControlMode


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _spawn_flat(data: mujoco.MjData, spawn_xy: np.ndarray, cage_radius: float, random_orientation: bool) -> None:
    from carolline_control.scripts.manual_teleop import _random_ground_qpos

    x, y = float(spawn_xy[0]), float(spawn_xy[1])
    center_z = cage_radius
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
    parser = argparse.ArgumentParser(description="SLAM exploration navigation")
    parser.add_argument(
        "--slam-config",
        default=str(REPO_ROOT / "carolline_control" / "navigation" / "slam_config.yaml"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--no-sensor-viz", action="store_true")
    parser.add_argument("--save-map", default=None, help="Save final occupancy map PNG")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Use oracle pose for control (SLAM still runs for mapping/planning eval)",
    )
    add_slam_nav_argument(parser)
    args = parser.parse_args()

    slam_config, slam_raw = load_slam_config(Path(args.slam_config))
    if args.seed is not None:
        np.random.seed(args.seed)
        if "exploration" in slam_raw:
            slam_raw["exploration"]["seed"] = args.seed

    config = load_config(args.config)
    model_path = _resolve_model_path(slam_raw.get("model_path", config.model_path))
    model, obstacles, spawn_xy, goal_xy = compile_exploration_scene(model_path, slam_raw)
    data = mujoco.MjData(model)

    use_slam_pose = args.slam_nav and not args.oracle
    estimator = build_estimator(model, config, slam_odom=use_slam_pose)
    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    exp = slam_raw.get("exploration", {})
    sim = slam_raw.get("simulation", {})
    perception_cfg = slam_raw.get("perception", {})
    settle_steps = int(sim.get("settle_steps", 350))
    random_orientation = bool(sim.get("random_orientation", False))

    _spawn_flat(data, spawn_xy, config.cage_radius, random_orientation)
    seed_estimator_from_sim(estimator, data)
    _settle(model, data, settle_steps)

    perception = RangePerception(
        model,
        stop_distance=float(exp.get("stop_distance", 0.65)),
        max_roll_height=float(exp.get("max_roll_height", 0.35)),
        fly_clearance=float(exp.get("fly_clearance", 0.90)),
        max_range=float(perception_cfg.get("max_range", 8.0)),
        forward_cone_deg=float(perception_cfg.get("forward_cone_deg", 75.0)),
    )
    navigator = SlamNavigator(model, perception, slam_config, map_origin_xy=spawn_xy - 18.0)
    navigator.reset(spawn_xy, float(data.qpos[2]), data.qpos[3:7].copy())
    navigator.set_goal(goal_xy)

    def terrain_z_at(x: float, y: float) -> float:
        return 0.0

    follower = WaypointFollower(
        config,
        speed=float(exp.get("patrol_speed", 2.2)),
        arrival_radius=0.55,
    )
    director = ExplorationDirector(
        controller,
        follower,
        navigator,
        goal_xy=goal_xy,
        terrain_z_at=terrain_z_at,
        fly_clearance=float(exp.get("fly_clearance", 0.55)),
        blocked_confirm_time=float(exp.get("blocked_confirm_time", 0.35)),
        hover_height=float(exp.get("hover_height", 1.05)),
        cage_radius=config.cage_radius,
        max_obstacle_height=float(exp.get("max_obstacle_height", 1.0)),
        max_flight_height=float(exp.get("max_flight_height", 1.85)),
    )
    director._arena_half_x = float(exp.get("arena_half_x", 14.0))
    director._arena_half_y = float(exp.get("arena_half_y", 10.0))
    # Shorter hover gate for exploration hops (default config uses 5 s).
    config.hover_before_flight_time = float(exp.get("hover_before_flight_time", 0.5))

    duration = float(sim.get("duration", 240.0))
    dt = float(model.opt.timestep)
    director.begin(data.time)

    print_estimator_mode(slam_odom=use_slam_pose)
    print(f"Exploration scene: {len(obstacles)} random obstacles")
    print(f"Spawn: ({spawn_xy[0]:.1f}, {spawn_xy[1]:.1f})  Goal: ({goal_xy[0]:.1f}, {goal_xy[1]:.1f})")
    print(f"Rangefinder sensors: {len(perception._sensor_ids)}")
    print(f"SLAM map: {slam_config.map_width_m:.0f}x{slam_config.map_height_m:.0f} m @ {slam_config.map_resolution} m/cell")

    last_scan = perception.scan(data, estimator.estimate(data))
    controller.planner.rolling_velocity = lambda state: follower.velocity_command_xy(
        state,
        director.mission_scan(last_scan, state),
    )

    substeps = max(1, int(args.substeps))
    track_distance = 10.0
    map_viz = MapVisualizer(navigator.mapper)

    def control_step(*, step_dt: float | None = None) -> bool:
        nonlocal last_scan
        use_dt = dt if step_dt is None else step_dt
        sensor_state = estimator.estimate(data)
        oracle_pos = data.qpos[:3].copy()

        if use_slam_pose:
            slam_pos, _slam_quat = navigator.update(
                data,
                sensor_state,
                oracle_position=oracle_pos,
            )
            slam_pos[2] = float(sensor_state.position[2])
            estimator.apply_slam_pose(slam_pos)
            state = estimator.estimate(data)
        else:
            navigator.update(data, sensor_state, oracle_position=oracle_pos)
            state = sensor_state

        last_scan = perception.scan(data, state)
        motor, cmd, mode, target = controller.compute(state, use_dt)
        running = director.update(state, mode, state.time, use_dt, last_scan)
        data.ctrl[:] = motor.thrusts
        return running and data.time < duration

    success = False
    try:
        if args.no_viewer:
            while control_step():
                mujoco.mj_step(model, data)
            success = director.phase == ExplorationPhase.GOAL_REACHED
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                tune_viewer_for_speed(viewer)
                viewer.cam.distance = track_distance
                viewer.cam.azimuth = 130.0
                viewer.cam.elevation = -22.0
                update_tracking_camera(viewer, data.qpos[:3], look_height=0.35, distance=track_distance)

                def on_frame(v, _data) -> None:
                    state = estimator.estimate(_data)
                    look_z = max(0.35, float(state.position[2]) * 0.45)
                    cam_dist = max(8.0, min(14.0, 8.0 + float(state.position[2]) * 0.8))
                    update_tracking_camera(v, state.position, look_height=look_z, distance=cam_dist)
                    if not args.no_sensor_viz:
                        draw_rangefinder_rays(v, perception, _data, last_scan)
                    cmd_xy = follower.velocity_command_xy(state, last_scan)
                    v.set_texts(
                        (
                            int(mujoco.mjtFontScale.mjFONTSCALE_150),
                            int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                            f"Phase: {director.phase.value}  Mode: {controller.mode_manager.mode.name}\n"
                            f"SLAM err: {navigator.last_oracle_position_error:.2f} m  "
                            f"range: {last_scan.forward_min_m:.2f} m  blocked={last_scan.blocked}\n"
                            f"cmd={float(np.linalg.norm(cmd_xy)):.2f} m/s  "
                            f"speed={float(np.linalg.norm(state.velocity[:2])):.2f} m/s  "
                            f"wp={follower.index}/{max(len(follower.waypoints), 1)}\n"
                            f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})  "
                            f"goal=({goal_xy[0]:.1f},{goal_xy[1]:.1f})  "
                            f"hover_z={controller.config.hover_height:.2f}",
                            "",
                        )
                    )

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
                success = director.phase == ExplorationPhase.GOAL_REACHED
    finally:
        if args.save_map:
            out = resolve_repo_path(REPO_ROOT, args.save_map)
            saved = map_viz.save_png(out)
            print(f"Map saved: {saved}")

    print(
        f"Finished t={data.time:.1f}s  phase={director.phase.value}  "
        f"success={success}  slam_err={navigator.last_oracle_position_error:.2f} m  "
        f"dist_goal={float(np.linalg.norm(estimator.estimate(data).position[:2] - goal_xy)):.2f} m"
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
