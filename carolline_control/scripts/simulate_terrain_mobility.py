"""CAROLLINE mobility test on mixed terrain (flat meadow, hills, mountains).

The scene uses a procedural MuJoCo heightfield:
  west  — flat meadow (spawn)
  center — rolling hills
  east  — mountainous peaks and valleys

Run from repo root:
    python carolline_control/scripts/simulate_terrain_mobility.py
    python carolline_control/scripts/simulate_terrain_mobility.py --mode patrol
    python carolline_control/scripts/simulate_terrain_mobility.py --mode manual --no-viewer

Manual mode controls match manual_teleop.py (arrows, 1-8 modes, R hold, 8 idle).
"""

from __future__ import annotations

import argparse
import math
import random
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
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.scripts.manual_teleop import (
    HELP_TEXT,
    KeyInputState,
    TeleopState,
    _apply_teleop_tuning,
    _random_ground_qpos,
    _settle,
    _sim_step,
    camera_ground_basis,
    update_tracking_camera,
)
from carolline_control.terrain.heightmap import sample_terrain_height
from carolline_control.terrain.scene import apply_terrain_mobility_tuning, compile_terrain_scene, load_terrain_config
from carolline_control.utils.types import ControlMode


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _spawn_on_terrain(
    data: mujoco.MjData,
    *,
    spawn_xy: list[float],
    cage_radius: float,
    heights: np.ndarray,
    layout,
    random_orientation: bool,
) -> None:
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


def _run_patrol(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CarollineController,
    estimator: StateEstimator,
    checkpoints: list[list[float]],
    speed: float,
    arrival_radius: float,
    duration: float,
    use_viewer: bool,
) -> None:
    if not checkpoints:
        raise ValueError("Patrol mode requires at least one checkpoint in terrain_config.yaml")

    target_idx = 0
    dt = float(model.opt.timestep)
    controller.mode_manager.request_roll()

    def step_once() -> dict:
        nonlocal target_idx
        state = estimator.estimate(data)
        target_xy = np.asarray(checkpoints[target_idx][:2], dtype=float)
        delta = target_xy - state.position[:2]
        dist = float(np.linalg.norm(delta))
        if dist < arrival_radius:
            target_idx = (target_idx + 1) % len(checkpoints)
            target_xy = np.asarray(checkpoints[target_idx][:2], dtype=float)
            delta = target_xy - state.position[:2]
            dist = float(np.linalg.norm(delta))

        direction = delta / dist if dist > 1e-3 else np.zeros(2)
        slowdown = float(arrival_radius) * 3.0
        if dist >= slowdown:
            cruise = speed
        else:
            cruise = speed * max(dist / slowdown, 0.25)
        vel_cmd = direction * cruise
        motor, _, mode, _ = controller.compute_manual(
            state,
            dt,
            ControlMode.ROLLING,
            velocity_xy=vel_cmd,
            velocity_world=np.zeros(3),
            yaw_rate=0.0,
        )
        data.ctrl[:] = motor.thrusts
        mujoco.mj_step(model, data)
        return {
            "mode": mode.name,
            "dist": dist,
            "target_idx": target_idx,
            "pos": state.position.copy(),
            "speed": float(np.linalg.norm(state.velocity)),
        }

    if not use_viewer:
        while data.time < duration:
            step_once()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 22.0
        viewer.cam.azimuth = 125.0
        viewer.cam.elevation = -28.0
        while viewer.is_running() and data.time < duration:
            info = step_once()
            viewer.set_texts(
                (
                    int(mujoco.mjtFontScale.mjFONTSCALE_150),
                    int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                    "Terrain patrol\nMode\nCheckpoint / dist [m]\nSpeed [m/s]\nPosition [m]",
                    (
                        "\n"
                        f"{info['mode']}\n"
                        f"{info['target_idx'] + 1}/{len(checkpoints)} / {info['dist']:.2f}\n"
                        f"{info['speed']:.2f}\n"
                        f"[{info['pos'][0]:+.1f}, {info['pos'][1]:+.1f}, {info['pos'][2]:+.2f}]"
                    ),
                )
            )
            viewer.sync()


def _run_manual(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CarollineController,
    estimator: StateEstimator,
    config,
    substeps: int,
    realtime: bool,
    teleop_ground_speed: float,
    teleop_time_constant: float,
) -> None:
    dt = float(model.opt.timestep)
    keys = KeyInputState()
    teleop = TeleopState(mode=ControlMode.ROLLING, ground_speed=teleop_ground_speed)
    teleop.vel_time_constant = teleop_time_constant
    last_mode = teleop.mode.name

    def on_key(keycode: int) -> None:
        keys.on_key(keycode)

    print(HELP_TEXT)
    print("Terrain manual mode — explore flat / hills / mountains.")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        track_distance = 20.0
        viewer.cam.distance = track_distance
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -25.0
        update_tracking_camera(viewer, data.qpos[:3], look_height=0.35, distance=track_distance)
        wall_t0 = time.perf_counter()
        sim_t0 = data.time
        move_basis: tuple[np.ndarray, np.ndarray] | None = None

        while viewer.is_running():
            pending_mode, speed_down, speed_up, toggle_help = keys.consume_pending()
            if pending_mode is not None and pending_mode != teleop.mode:
                if pending_mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
                    keys.clear_motion()
                    teleop.stop()
                    controller.pre_takeoff.reset(estimator.estimate(data))
                elif pending_mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                    keys.clear_motion()
                    teleop.stop()
                    state = estimator.estimate(data)
                    controller._hover_anchor_xy = state.position[:2].copy()
                    controller._flight_yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
                teleop.mode = pending_mode
            if speed_down:
                teleop.speed_scale = max(0.3, teleop.speed_scale * 0.85)
            if speed_up:
                teleop.speed_scale = min(2.5, teleop.speed_scale * 1.15)
            if toggle_help:
                teleop.show_help = not teleop.show_help

            vx, vy, vz, yaw, roll_hold = keys.direction()
            if vx != 0 or vy != 0:
                move_basis = camera_ground_basis(viewer.cam.azimuth, viewer.cam.elevation)
            else:
                move_basis = None
            frame_dt = dt * substeps
            keys.release_stale()
            teleop.update(
                vx, vy, vz, yaw, roll_hold, frame_dt, config.rolling_max_speed, move_basis=move_basis
            )

            for _ in range(substeps):
                mode = _sim_step(controller, estimator, teleop, model, data, dt)
                if mode.name != last_mode:
                    print(f"t={data.time:6.2f}s  {last_mode} -> {mode.name}")
                    last_mode = mode.name

            state = estimator.estimate(data)
            update_tracking_camera(viewer, state.position, look_height=0.35)

            if teleop.show_help:
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        HELP_TEXT
                        + f"\n\nMode: {teleop.mode.name}  key: {keys.last_label}\n"
                        f"cmd [{teleop.velocity_xy[0]:+.1f},{teleop.velocity_xy[1]:+.1f}] "
                        f"z={teleop.velocity_world[2]:+.1f}  pos z={state.position[2]:.2f}",
                        "",
                    )
                )
            else:
                viewer.clear_texts()
            viewer.sync()

            if realtime:
                delay = (data.time - sim_t0) - (time.perf_counter() - wall_t0)
                if delay > 0.0:
                    time.sleep(min(delay, 0.02))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE mixed-terrain mobility test")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "config.yaml"),
        help="Controller config",
    )
    parser.add_argument(
        "--terrain-config",
        default=str(REPO_ROOT / "carolline_control" / "terrain_config.yaml"),
        help="Terrain layout and checkpoints",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "patrol"],
        default="manual",
        help="manual keyboard teleop or autonomous rolling patrol",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--regenerate-heightmap", action="store_true")
    parser.add_argument("--speed", type=float, default=None, help="Override rolling / patrol speed [m/s]")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    terrain_raw = load_terrain_config(Path(args.terrain_config))
    if args.regenerate_heightmap:
        from carolline_control.terrain.scene import TERRAIN_ASSET_DIR, terrain_layout_from_config
        from carolline_control.terrain.heightmap import write_heightmap_png

        layout = terrain_layout_from_config(terrain_raw)
        png = TERRAIN_ASSET_DIR / f"heightmap_seed{layout.seed}_{layout.nrow}x{layout.ncol}.png"
        if png.exists():
            png.unlink()
        write_heightmap_png(png, layout)

    config = load_config(args.config)
    _apply_teleop_tuning(config)
    mob = apply_terrain_mobility_tuning(config, terrain_raw)
    if args.speed is not None:
        config.rolling_max_speed = float(args.speed)
        mob["patrol_speed"] = float(args.speed)
        mob["teleop_ground_speed"] = float(args.speed)

    base_path = _resolve_model_path(terrain_raw.get("model_path", config.model_path))
    model, layout, heights = compile_terrain_scene(
        base_path,
        config=terrain_raw,
        cage_radius=config.cage_radius,
    )
    data = mujoco.MjData(model)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)

    spawn = terrain_raw.get("spawn", {})
    spawn_xy = spawn.get("xy", [-10.0, 0.0])
    _spawn_on_terrain(
        data,
        spawn_xy=spawn_xy,
        cage_radius=config.cage_radius,
        heights=heights,
        layout=layout,
        random_orientation=bool(spawn.get("random_orientation", True)),
    )
    _settle(model, data, steps=int(spawn.get("settle_steps", 400)))

    sim = terrain_raw.get("simulation", {})
    duration = float(args.duration if args.duration is not None else sim.get("duration", 180.0))

    print("=" * 60)
    print("CAROLLINE terrain mobility test")
    print("=" * 60)
    print(f"  terrain: flat (x < {layout.flat_x_end:.0f}) | hills | mountains (east)")
    print(f"  extent: {layout.size_x:.0f} x {layout.size_y:.0f} m, max height {layout.max_height:.2f} m")
    print(f"  spawn: [{spawn_xy[0]:+.1f}, {spawn_xy[1]:+.1f}]  mode={args.mode}")
    print(f"  rolling max speed: {config.rolling_max_speed:.2f} m/s")
    print(f"  checkpoints: {len(terrain_raw.get('checkpoints', []))}")

    if args.mode == "patrol":
        _run_patrol(
            model=model,
            data=data,
            controller=controller,
            estimator=estimator,
            checkpoints=terrain_raw.get("checkpoints", []),
            speed=float(args.speed if args.speed is not None else sim.get("patrol_speed", mob.get("patrol_speed", 3.25))),
            arrival_radius=float(sim.get("patrol_arrival_radius", 0.55)),
            duration=duration,
            use_viewer=not args.no_viewer,
        )
    elif args.no_viewer:
        raise SystemExit("Manual mode requires the MuJoCo viewer (omit --no-viewer).")
    else:
        _run_manual(
            model=model,
            data=data,
            controller=controller,
            estimator=estimator,
            config=config,
            substeps=max(1, args.substeps),
            realtime=args.realtime,
            teleop_ground_speed=float(mob.get("teleop_ground_speed", config.rolling_max_speed)),
            teleop_time_constant=float(mob.get("teleop_time_constant", 0.030)),
        )

    state = estimator.estimate(data)
    print(
        f"Done. pos=[{state.position[0]:+.2f}, {state.position[1]:+.2f}, {state.position[2]:+.2f}] "
        f"speed={np.linalg.norm(state.velocity):.2f} m/s mode={controller.mode_manager.mode.name}"
    )


if __name__ == "__main__":
    main()
