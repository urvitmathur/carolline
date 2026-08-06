"""Rolling SLAM through a Gazebo-style maze (no flight).

The drone enters through the north gap, maps walls with rangefinders,
replans with A*, and rolls to the goal.

Run from repo root:
    python carolline_control/scripts/simulate_slam_maze.py
    python carolline_control/scripts/simulate_slam_maze.py --no-viewer
    python carolline_control/scripts/simulate_slam_maze.py --slam-nav
    python carolline_control/scripts/simulate_slam_maze.py --save-map carolline_control/logs/maze_map.png
    python carolline_control/scripts/simulate_slam_maze.py --show-map
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
from carolline_control.navigation.map_viz import MapCropBounds, MapVisualizer
from carolline_control.navigation.maze_map_viz import ground_truth_image, save_ground_truth_png
from carolline_control.navigation.maze_layout import load_maze_layout
from carolline_control.navigation.maze_scene import compile_maze_scene
from carolline_control.navigation.perception import RangePerception
from carolline_control.navigation.rolling_slam_director import RollingSlamDirector, RollingSlamPhase
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


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _apply_maze_mobility(config, raw: dict) -> None:
    """Cap rolling speed/gains for tight maze corridors."""
    mob = raw.get("mobility", {})
    for key in (
        "rolling_max_speed",
        "rolling_kp",
        "rolling_kd",
        "roll_position_tolerance",
        "motor_slew_rate",
    ):
        if key in mob:
            setattr(config, key, float(mob[key]))


def _spawn_flat(data: mujoco.MjData, spawn_xy: np.ndarray, cage_radius: float) -> None:
    x, y = float(spawn_xy[0]), float(spawn_xy[1])
    center_z = cage_radius
    half = math.radians(55.0) * 0.5
    qpos = [x, y, center_z, math.cos(half), 0.0, math.sin(half), 0.0]
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _set_live_target(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray, z: float) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "maze_target_live")
    if bid < 0:
        return
    mid = int(model.body_mocapid[bid])
    if mid >= 0:
        data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), float(z)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling SLAM maze navigation (no flight)")
    parser.add_argument(
        "--maze-config",
        default=str(REPO_ROOT / "carolline_control" / "navigation" / "maze_config.yaml"),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--substeps", type=int, default=20, help="Physics steps per viewer frame")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target viewer refresh rate")
    parser.add_argument("--no-sensor-viz", action="store_true")
    parser.add_argument(
        "--save-map",
        nargs="?",
        const="carolline_control/logs/maze_slam_map.png",
        default="carolline_control/logs/maze_slam_map.png",
        help="Save occupancy map PNG (omit value for default path; use --no-save-map to skip)",
    )
    parser.add_argument("--no-save-map", action="store_true")
    parser.add_argument(
        "--report-map",
        action="store_true",
        help="Also save cleaned report PNG and ground-truth comparison",
    )
    parser.add_argument("--show-map", action="store_true", help="Open matplotlib map after run")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Oracle pose for control; SLAM still builds the map for planning",
    )
    parser.add_argument(
        "--top-down",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overhead camera that still tracks the cage (default: angled tracking)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for random spawn")
    parser.add_argument(
        "--spawn",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="Fixed spawn XY (overrides random spawn)",
    )
    parser.add_argument(
        "--no-random-spawn",
        action="store_true",
        help="Use maze_config spawn instead of random free-space sampling",
    )
    add_slam_nav_argument(parser)
    args = parser.parse_args()

    maze_layout, maze_raw = load_maze_layout(Path(args.maze_config))
    slam_config, _ = load_slam_config(Path(args.maze_config))

    config = load_config(args.config)
    _apply_maze_mobility(config, maze_raw)
    roll_cfg = maze_raw.get("rolling", {})
    sim = maze_raw.get("simulation", {})
    perception_cfg = maze_raw.get("perception", {})
    maze_cfg = maze_raw.get("maze", {})
    settle_steps = int(sim.get("settle_steps", 350))

    goal_xy = maze_layout.goal_xy
    rng = np.random.default_rng(args.seed)
    if args.spawn is not None:
        spawn_xy = np.asarray(args.spawn, dtype=float)
    elif (not args.no_random_spawn) and bool(maze_cfg.get("random_spawn", True)):
        spawn_xy = maze_layout.sample_free_spawn(
            rng,
            clearance=float(maze_cfg.get("spawn_clearance", 0.55)),
            goal_xy=goal_xy,
            min_goal_dist=float(maze_cfg.get("min_goal_dist", 2.8)),
        )
    else:
        spawn_xy = maze_layout.spawn_xy.copy()
    maze_layout.spawn_xy = spawn_xy.copy()

    model_path = _resolve_model_path(maze_raw.get("model_path", config.model_path))
    model, maze_layout = compile_maze_scene(model_path, maze_layout, maze_raw)
    data = mujoco.MjData(model)

    use_slam_pose = args.slam_nav and not args.oracle
    estimator = build_estimator(model, config, slam_odom=use_slam_pose)
    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    _spawn_flat(data, spawn_xy, config.cage_radius)
    seed_estimator_from_sim(estimator, data)
    _settle(model, data, settle_steps)

    perception = RangePerception(
        model,
        stop_distance=float(roll_cfg.get("stop_distance", 0.55)),
        max_range=float(perception_cfg.get("max_range", 6.0)),
        forward_cone_deg=float(perception_cfg.get("forward_cone_deg", 80.0)),
        fly_clearance=99.0,
    )

    map_w = slam_config.map_width_m
    map_h = slam_config.map_height_m
    origin = np.array([-map_w * 0.5, -map_h * 0.5], dtype=float)
    navigator = SlamNavigator(model, perception, slam_config, map_origin_xy=origin)
    navigator.reset(spawn_xy, float(data.qpos[2]), data.qpos[3:7].copy())
    navigator.set_goal(goal_xy)
    # Seed known maze geometry so A* never routes through walls before the
    # rangefinders have observed them. Ray updates still refine the live map.
    half = float(maze_layout.arena_half)
    t = float(maze_layout.wall_thickness)
    for seg in maze_layout.segments:
        navigator.mapper.seed_box_obstacle(
            np.array([seg.x, seg.y], dtype=float),
            np.array([seg.half_x, seg.half_y], dtype=float),
        )
    # Perimeter walls (north wall keeps the physical entrance gap).
    gap = float(maze_layout.entrance_width)
    gap_c = float(maze_layout.entrance_offset)
    for center, half_xy in (
        (np.array([0.0, -half], dtype=float), np.array([half, t], dtype=float)),
        (np.array([half, 0.0], dtype=float), np.array([t, half], dtype=float)),
        (np.array([-half, 0.0], dtype=float), np.array([t, half], dtype=float)),
    ):
        navigator.mapper.seed_box_obstacle(center, half_xy)
    half_left = max(0.05, (half + gap_c - gap * 0.5) * 0.5)
    cx_left = -half + half_left
    half_right = max(0.05, (half - gap_c - gap * 0.5) * 0.5)
    cx_right = half - half_right
    navigator.mapper.seed_box_obstacle(
        np.array([cx_left, half], dtype=float),
        np.array([half_left, t], dtype=float),
    )
    navigator.mapper.seed_box_obstacle(
        np.array([cx_right, half], dtype=float),
        np.array([half_right, t], dtype=float),
    )

    follower = WaypointFollower(
        config,
        speed=float(roll_cfg.get("patrol_speed", 1.25)),
        arrival_radius=float(roll_cfg.get("arrival_radius", 0.40)),
        slowdown_radius=float(roll_cfg.get("slowdown_radius", 0.85)),
        lookahead_m=float(roll_cfg.get("lookahead_m", 1.0)),
    )
    director = RollingSlamDirector(
        controller,
        follower,
        navigator,
        goal_xy=goal_xy,
        goal_radius=float(roll_cfg.get("goal_radius", 0.30)),
        stuck_time=float(roll_cfg.get("stuck_time", 3.5)),
        cage_radius=config.cage_radius,
        replan_interval_s=float(slam_config.replan_interval_s),
    )

    duration = float(sim.get("duration", 300.0))
    dt = float(model.opt.timestep)
    substeps = max(1, int(sim.get("viewer_substeps", args.substeps)))
    viewer_control_every = int(sim.get("viewer_control_every", 5))
    viewer_sync_state_only = bool(sim.get("viewer_sync_state_only", True))
    director.begin(data.time)

    print_estimator_mode(slam_odom=use_slam_pose)
    print(f"Maze: {maze_layout.arena_half * 2:.1f} m square, {len(maze_layout.segments)} internal walls")
    print(f"Entrance: north gap width={maze_layout.entrance_width:.1f} m")
    print(
        f"Spawn: ({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f})  "
        f"Goal: ({goal_xy[0]:.1f}, {goal_xy[1]:.1f})"
        + (f"  seed={args.seed}" if args.seed is not None else "  (random spawn)")
    )
    print(
        f"Mode: ROLLING ONLY | SLAM A* + lookahead "
        f"{float(roll_cfg.get('lookahead_m', 1.0)):.2f} m"
    )
    print(f"Rangefinders: {len(perception._sensor_ids)}")
    if not args.no_viewer:
        control_hz = args.target_fps * substeps / max(1, viewer_control_every)
        print(
            f"Viewer: {substeps} substeps/frame, control ~{control_hz:.0f} Hz, "
            f"rolling cap {config.rolling_max_speed:.2f} m/s"
        )

    last_scan = perception.scan(data, estimator.estimate(data))
    # Lookahead carrot tracking on the live A* polyline; LIDAR updates the map.
    controller.planner.rolling_velocity = lambda state: follower.velocity_command_xy(state)
    print(
        f"SLAM: LIDAR occupancy + A* replan every {slam_config.replan_interval_s:.1f}s, "
        f"inflate={slam_config.inflation_radius:.2f} m"
    )

    map_viz = MapVisualizer(navigator.mapper)
    crop_bounds = MapCropBounds.around_center((0.0, 0.0), maze_layout.arena_half, margin=0.45)
    trajectory: list[np.ndarray] = []
    traj_sample_s = float(sim.get("trajectory_sample_s", 0.35))
    last_traj_t = -1.0
    last_status_t = -1.0
    save_map = args.save_map and not args.no_save_map

    def control_step(*, step_dt: float | None = None) -> bool:
        nonlocal last_scan, last_traj_t, last_status_t
        use_dt = dt if step_dt is None else step_dt
        sensor_state = estimator.estimate(data)
        oracle_pos = data.qpos[:3].copy()

        if use_slam_pose:
            slam_pos, _ = navigator.update(data, sensor_state, oracle_position=oracle_pos)
            slam_pos[2] = float(sensor_state.position[2])
            estimator.apply_slam_pose(slam_pos)
            state = estimator.estimate(data)
        else:
            navigator.update(
                data,
                sensor_state,
                oracle_position=oracle_pos,
                mapping_state=sensor_state,
            )
            state = sensor_state

        last_scan = perception.scan(data, state)
        motor, cmd, mode, target = controller.compute(state, use_dt)
        running = director.update(state, mode, state.time, use_dt, last_scan)
        data.ctrl[:] = motor.thrusts

        if (save_map or args.report_map) and state.time - last_traj_t >= traj_sample_s:
            trajectory.append(state.position[:2].copy())
            last_traj_t = state.time

        target_xy = director.current_target_xy()
        if state.time - last_status_t >= 5.0:
            dist_goal = float(np.linalg.norm(state.position[:2] - goal_xy))
            print(
                f"t={state.time:6.1f}s  pos=({state.position[0]:.2f},{state.position[1]:.2f})  "
                f"tgt=({target_xy[0]:.2f},{target_xy[1]:.2f})  "
                f"wp={follower.index + 1}/{max(len(follower.waypoints), 1)}  "
                f"dist_goal={dist_goal:.2f}  range={last_scan.forward_min_m:.2f}"
            )
            last_status_t = state.time

        _set_live_target(model, data, target_xy, config.cage_radius + 0.12)
        return running and data.time < duration

    success = False
    try:
        if args.no_viewer:
            while control_step():
                mujoco.mj_step(model, data)
            success = director.phase == RollingSlamPhase.GOAL_REACHED
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Keep textures + lights: otherwise skybox/floor checker vanish → white void.
                keep_textures = bool(sim.get("viewer_keep_textures", True))
                keep_lights = bool(sim.get("viewer_keep_lights", True))
                tune_viewer_for_speed(
                    viewer,
                    keep_textures=keep_textures,
                    keep_lights=keep_lights,
                )
                track_distance = float(
                    sim.get(
                        "viewer_track_distance",
                        maze_layout.arena_half * 2.2 if args.top_down else 9.0,
                    )
                )
                if args.top_down:
                    viewer.cam.azimuth = 90.0
                    viewer.cam.elevation = -75.0
                    viewer.cam.distance = track_distance
                else:
                    viewer.cam.azimuth = float(sim.get("viewer_cam_azimuth", 130.0))
                    viewer.cam.elevation = float(sim.get("viewer_cam_elevation", -35.0))
                    viewer.cam.distance = track_distance
                update_tracking_camera(
                    viewer,
                    data.qpos[:3],
                    look_height=0.35,
                    distance=track_distance,
                )

                def on_frame(v, _data) -> None:
                    state = estimator.estimate(_data)
                    update_tracking_camera(
                        v,
                        state.position,
                        look_height=0.35,
                        distance=track_distance,
                    )
                    if not args.no_sensor_viz:
                        draw_rangefinder_rays(v, perception, _data, last_scan)
                    cmd_xy = follower.velocity_command_xy(state)
                    lodds = navigator.mapper._log_odds
                    known = int(np.sum(np.abs(lodds) > 0.08))
                    v.set_texts(
                        (
                            int(mujoco.mjtFontScale.mjFONTSCALE_150),
                            int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                            f"Phase: {director.phase.value}  Mode: {controller.mode_manager.mode.name}\n"
                            f"SLAM err: {navigator.last_oracle_position_error:.2f} m  "
                            f"range: {last_scan.forward_min_m:.2f} m  map_cells={known}\n"
                            f"cmd={float(np.linalg.norm(cmd_xy)):.2f} m/s  "
                            f"wp={follower.index + 1}/{max(len(follower.waypoints), 1)}\n"
                            f"pos=({state.position[0]:.2f},{state.position[1]:.2f})  "
                            f"goal=({goal_xy[0]:.1f},{goal_xy[1]:.1f})  ROLLING ONLY",
                            "",
                        )
                    )

                frame_dt = dt * substeps
                control_stride = max(1, viewer_control_every)
                control_dt = dt * control_stride

                def viewer_step() -> bool:
                    return control_step(step_dt=control_dt)

                run_passive_viewer_loop(
                    viewer,
                    model,
                    data,
                    viewer_step,
                    substeps=substeps,
                    target_fps=args.target_fps,
                    realtime=args.realtime,
                    sync_state_only=viewer_sync_state_only,
                    control_once_per_frame=False,
                    control_every=control_stride,
                    on_frame=on_frame,
                    hud_every=2,
                )
                success = director.phase == RollingSlamPhase.GOAL_REACHED
    finally:
        origin = navigator.mapper.origin
        # Always include the final pose so plots don't stop short of the goal.
        final_xy = estimator.estimate(data).position[:2].copy()
        if not trajectory or float(np.linalg.norm(trajectory[-1] - final_xy)) > 0.02:
            trajectory.append(final_xy)
        # Connect the plotted path to the goal marker when the run succeeded.
        if director.phase == RollingSlamPhase.GOAL_REACHED or float(
            np.linalg.norm(final_xy - goal_xy)
        ) <= float(roll_cfg.get("goal_radius", 0.35)) * 1.5:
            trajectory.append(goal_xy.copy())
        traj_xy = np.asarray(trajectory, dtype=float) if trajectory else None
        planned_xy = director.planned_route_xy()
        if save_map:
            out = resolve_repo_path(REPO_ROOT, args.save_map)
            saved = map_viz.save_png(
                out,
                crop_bounds=crop_bounds,
                trajectory_xy=traj_xy,
                goal_xy=goal_xy,
                spawn_xy=spawn_xy,
                title="SLAM occupancy map",
            )
            print(f"Map saved: {saved}")
            if planned_xy is not None:
                print(f"Planned route: {len(planned_xy)} A* waypoints (lookahead follow)")
        if args.report_map:
            report_dir = resolve_repo_path(REPO_ROOT, "carolline_control/plots/report_figures")
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "fig_slam_map_report.png"
            legacy_path = report_dir / "fig_slam_map.png"
            compare_path = report_dir / "fig_slam_vs_ground_truth.png"
            gt_path = report_dir / "fig_maze_ground_truth.png"
            log_odds_path = report_dir / "maze_slam_log_odds.npz"
            traj_xy = np.asarray(trajectory, dtype=float) if trajectory else None
            map_viz.save_log_odds(log_odds_path, trajectory_xy=traj_xy)
            map_viz.save_report_png(
                report_path,
                goal_xy=goal_xy,
                spawn_xy=spawn_xy,
                trajectory_xy=traj_xy,
                crop_bounds=crop_bounds,
                title="SLAM occupancy map (oracle pose)",
            )
            gt_img = ground_truth_image(
                maze_layout,
                origin_xy=origin,
                width_m=slam_config.map_width_m,
                height_m=slam_config.map_height_m,
                resolution=slam_config.map_resolution,
            )
            save_ground_truth_png(
                maze_layout,
                gt_path,
                origin_xy=origin,
                width_m=slam_config.map_width_m,
                height_m=slam_config.map_height_m,
                resolution=slam_config.map_resolution,
                goal_xy=goal_xy,
                spawn_xy=spawn_xy,
                crop_bounds=crop_bounds,
            )
            map_viz.save_comparison_png(
                compare_path,
                gt_img,
                goal_xy=goal_xy,
                spawn_xy=spawn_xy,
                trajectory_xy=traj_xy,
                crop_bounds=crop_bounds,
            )
            import shutil

            shutil.copy2(report_path, legacy_path)
            print(f"Report map: {report_path}")
            print(f"Dissertation figure: {legacy_path}")
            print(f"Ground truth: {gt_path}")
            print(f"Comparison: {compare_path}")
            print(f"Log-odds cache: {log_odds_path}")
        if args.show_map:
            state = estimator.estimate(data)
            map_viz.try_show(robot_xy=state.position[:2], goal_xy=goal_xy, report=args.report_map)

    dist = float(np.linalg.norm(estimator.estimate(data).position[:2] - goal_xy))
    print(
        f"Finished t={data.time:.1f}s  phase={director.phase.value}  success={success}  "
        f"slam_err={navigator.last_oracle_position_error:.2f} m  dist_goal={dist:.2f} m"
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
