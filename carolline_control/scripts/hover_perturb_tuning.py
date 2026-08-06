"""
Hover stabilization tuning harness.

Spawns the caged drone in HOVER, runs the geometric flight controller, and lets
you nudge the body in the MuJoCo viewer to see how fast it recenters.

Run from repo root:
    python carolline_control/scripts/hover_perturb_tuning.py

Viewer nudges (MuJoCo simulate):
  1. Double-click the drone body (x2 / cage) if it is not already selected.
  2. Hold Ctrl + right mouse — translate (push/pull).
  3. Hold Ctrl + left mouse — rotate (tilt).
  4. Release Ctrl — controller returns to hover point and level (upright) attitude.

Keyboard (instant impulses, good for A/B gain tests):
  Space     pause / resume physics
  R         reset pose to hover anchor
  I/K/J/L   +X / -X / +Y / -Y velocity impulse
  U/O       +Z / -Z velocity impulse
  Q/E       +yaw / -yaw rate impulse
  Tab       toggle left UI panel (Simulation controls)
"""

from __future__ import annotations

import argparse
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
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.sim_estimator import build_estimator, seed_estimator_from_sim
from carolline_control.utils.so3 import attitude_error, body_z_world, quat_to_rot, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


def _hover_qpos(anchor_xy: np.ndarray, height: float, yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array(
        [
            float(anchor_xy[0]),
            float(anchor_xy[1]),
            float(height),
            float(np.cos(half)),
            0.0,
            0.0,
            float(np.sin(half)),
        ],
        dtype=float,
    )


def _reset_hover(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    anchor_xy: np.ndarray,
    height: float,
    yaw: float,
) -> None:
    data.qpos[:7] = _hover_qpos(anchor_xy, height, yaw)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Body '{name}' not found in model.")
    return int(body_id)


def _select_body_for_perturb(
    perturb: mujoco.MjvPerturb,
    body_id: int,
) -> None:
    """Pre-select the drone so Ctrl+drag works immediately."""
    perturb.select = body_id
    perturb.flexselect = -1
    perturb.skinselect = -1
    perturb.localpos[:] = 0.0


def _apply_kinematic_perturb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perturb: mujoco.MjvPerturb,
) -> bool:
    """Apply mouse nudge as a direct pose edit (passive viewer default is force-only).

    MuJoCo's passive viewer calls mjv_applyPerturbPose(..., flg=0), which only
    moves mocap bodies. The caged drone is a free joint, so Ctrl+drag only injected
    weak xfrc_applied forces that the hover controller cancelled — hence the laggy
    nudge. flg=1 moves dynamic bodies kinematically while Ctrl is held.
    """
    if int(perturb.active) == 0:
        return False
    data.xfrc_applied[:] = 0.0
    mujoco.mjv_applyPerturbPose(model, data, perturb, 1)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Hover + perturbation controller tuning harness")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--height", type=float, default=None, help="Hover center height [m]")
    parser.add_argument("--x", type=float, default=None, help="Hover anchor X [m]")
    parser.add_argument("--y", type=float, default=None, help="Hover anchor Y [m]")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Initial / hold yaw [deg]")
    parser.add_argument(
        "--impulse",
        type=float,
        default=0.8,
        help="Keyboard velocity impulse magnitude [m/s or rad/s for Q/E]",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="Seconds of hover before keyboard impulses (viewer drag works immediately)",
    )
    parser.add_argument("--duration", type=float, default=600.0, help="Max sim time [s]")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.004,
        help="MuJoCo dt [s]; 4 ms gives smoother viewer than default 2 ms",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=10,
        help="Physics substeps per viewer frame (higher = faster sim, smoother motion)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Cap sim to wall clock (default: uncapped for responsive viewer)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wall-clock multiplier when --realtime is set",
    )
    args = parser.parse_args()

    raw = load_raw_config(args.config)
    config = load_config(args.config)

    model = compile_model_with_markers(config.model_path, config)
    model.opt.timestep = float(args.timestep)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    steps_per_frame = max(int(args.steps_per_frame), 1)

    # Tuning harness: disable motor slew so recovery is not artificially sluggish.
    config.motor_slew_rate = 1.0e9

    estimator = build_estimator(model, config, sensor_only=False)
    controller = CarollineController(config)

    mission = raw.get("mission", {})
    spawn_xy = np.asarray(mission.get("spawn_xy", [0.0, 0.0]), dtype=float)
    anchor_xy = np.array(
        [
            float(args.x if args.x is not None else spawn_xy[0]),
            float(args.y if args.y is not None else spawn_xy[1]),
        ],
        dtype=float,
    )
    hover_height = float(args.height if args.height is not None else config.hover_height)
    yaw = float(np.radians(args.yaw_deg))
    drone_body_id = _body_id(model, "x2")

    _reset_hover(model, data, anchor_xy, hover_height, yaw)
    seed_estimator_from_sim(estimator, data)

    controller.mode_manager.mode = ControlMode.HOVER
    controller._hover_anchor_xy = anchor_xy.copy()
    controller._flight_yaw = yaw

    paused = False
    dragging = False
    pending_impulse = np.zeros(3, dtype=float)
    pending_yaw_rate = 0.0
    state = estimator.estimate(data)

    def on_key(keycode: int) -> None:
        nonlocal paused, pending_yaw_rate
        try:
            key = chr(keycode)
        except ValueError:
            return
        key = key.lower()
        imp = float(args.impulse)
        if key == " ":
            paused = not paused
        elif key == "r":
            _reset_hover(model, data, anchor_xy, hover_height, yaw)
            seed_estimator_from_sim(estimator, data)
            controller._last_motors[:] = 0.0
        elif key == "i":
            pending_impulse[0] += imp
        elif key == "k":
            pending_impulse[0] -= imp
        elif key == "l":
            pending_impulse[1] += imp
        elif key == "j":
            pending_impulse[1] -= imp
        elif key == "u":
            pending_impulse[2] += imp
        elif key == "o":
            pending_impulse[2] -= imp
        elif key == "q":
            pending_yaw_rate += imp
        elif key == "e":
            pending_yaw_rate -= imp

    print("Hover perturbation tuning harness")
    print(f"  model   : {config.model_path}")
    print(f"  anchor  : ({anchor_xy[0]:.2f}, {anchor_xy[1]:.2f}) @ z={hover_height:.2f} m")
    print(f"  mass    : {config.mass:.3f} kg")
    print(
        f"  gains   : kx={config.kx} kv={config.kv} kx_z={config.kx_z} kv_z={config.kv_z} "
        f"kR={config.kR} kOmega={config.kOmega}"
    )
    print(f"  sim dt  : {dt:.4f} s, {steps_per_frame} steps/frame")
    print(f"  pacing  : {'realtime x' + str(args.speed) if args.realtime else 'uncapped (smooth viewer)'}")
    print("  Release Ctrl after a nudge: drone should return to anchor + upright.")
    print("  Keys: Space pause, R reset, I/K/J/L/U/O/Q/E impulses.")

    settle_until = args.settle
    start_time = data.time
    speed = max(float(args.speed), 0.05)
    hover_rd = controller.flight._orientation.hover_yaw(yaw)
    target_pos = np.array([anchor_xy[0], anchor_xy[1], hover_height], dtype=float)
    frame_idx = 0

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        _select_body_for_perturb(viewer.perturb, drone_body_id)
        viewer.cam.lookat[:] = [anchor_xy[0], anchor_xy[1], hover_height]
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 135.0

        wall_t0 = time.perf_counter()
        sim_t0 = data.time

        while viewer.is_running() and data.time - start_time < args.duration:
            if not paused and not dragging:
                for sub in range(steps_per_frame):
                    if data.time - start_time >= args.duration:
                        break
                    if args.realtime:
                        wall_elapsed = time.perf_counter() - wall_t0
                        sim_target = sim_t0 + wall_elapsed * speed
                        if data.time >= sim_target:
                            break

                    if sub == 0 and data.time >= settle_until:
                        if np.linalg.norm(pending_impulse) > 1e-9:
                            data.qvel[:3] += pending_impulse
                            pending_impulse[:] = 0.0
                        if abs(pending_yaw_rate) > 1e-9:
                            data.qvel[5] += pending_yaw_rate
                            pending_yaw_rate = 0.0

                    state = estimator.estimate(data)
                    cmd = controller.flight.hover(state, hover_height, anchor_xy, yaw)
                    motor = controller._allocate(cmd, state, ControlMode.HOVER)
                    data.ctrl[:] = motor.thrusts
                    mujoco.mj_step(model, data)

            viewer.sync(state_only=(not dragging and not paused))
            with viewer.lock():
                dragging = _apply_kinematic_perturb(model, data, viewer.perturb)
            if dragging:
                data.ctrl[:] = 0.0
                controller._last_motors[:] = 0.0

            frame_idx += 1
            if frame_idx % 4 == 0:
                pos_err = float(np.linalg.norm(data.qpos[:3] - target_pos))
                vel = float(np.linalg.norm(data.qvel[:3]))
                rotation = quat_to_rot(data.qpos[3:7])
                roll, pitch, _ = rot_to_euler_zyx(rotation)
                body_z_up = float(body_z_world(rotation)[2])
                tilt = float(np.degrees(np.arccos(np.clip(body_z_up, -1.0, 1.0))))
                diag = controller.last_diagnostics
                att_err = float(np.linalg.norm(attitude_error(rotation, hover_rd)))
                sat = "SAT" if diag.motor_saturated and not dragging else "ok"
                if dragging:
                    phase = "DRAG"
                elif data.time < settle_until:
                    phase = "SETTLE"
                elif paused:
                    phase = "PAUSED"
                else:
                    phase = "HOVER"

                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        (
                            "Hover perturb tuning\n"
                            "Phase\n"
                            "Pos err [m]\n"
                            "Tilt [deg] / bz\n"
                            "Att err |eR|\n"
                            "Speed [m/s]\n"
                            "Roll / Pitch [deg]\n"
                            "Motors\n"
                            "Nudge"
                        ),
                        (
                            "\n"
                            f"{phase}\n"
                            f"{pos_err:.3f}\n"
                            f"{tilt:.2f} / {body_z_up:+.3f}\n"
                            f"{att_err:.3f}\n"
                            f"{vel:.3f}\n"
                            f"{np.degrees(roll):+.1f} / {np.degrees(pitch):+.1f}\n"
                            f"{sat}  spread={diag.motor_spread:.2f} N\n"
                            "Ctrl+drag, release to recover"
                        ),
                    )
                )

            if args.realtime:
                wall_elapsed = time.perf_counter() - wall_t0
                sim_target = sim_t0 + wall_elapsed * speed
                delay = (data.time - sim_target)
                if delay > 0.0:
                    time.sleep(min(delay, 0.05))

    print("Done.")


if __name__ == "__main__":
    main()
