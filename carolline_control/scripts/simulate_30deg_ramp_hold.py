"""Visualize CAROLLINE rolling up a 30-degree ramp and holding position.

The vehicle rolls to a target shortly before the upper ramp edge, then remains
in ROLLING with zero desired velocity. Contact-frame gravity compensation and
angular-rate feedback keep the cage stationary without forcing it upright.

Run from the repository root:
    python carolline_control/scripts/simulate_30deg_ramp_hold.py
    python carolline_control/scripts/simulate_30deg_ramp_hold.py --no-viewer
"""

from __future__ import annotations

import argparse
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
from carolline_control.utils.types import ControlMode

ROLLING_RAMP_CONFIG = REPO_ROOT / "carolline_control" / "rolling_ramp_config.yaml"

# Defaults for the simple single-ramp teleop scene (overridden by rolling_ramp_config.yaml)
_DEFAULT_RAMP = {
    "angle_deg": 5.0,
    "flat_start": 1.4,
    "ramp_length": 4.0,
    "width": 3.0,
}
RAMP_THICKNESS = 0.10
TARGET_EDGE_CLEARANCE = 0.55


def load_ramp_scene_params(
    *,
    angle_deg: float | None = None,
    ramp_length: float | None = None,
    flat_start: float | None = None,
    width: float | None = None,
) -> dict[str, float]:
    """Load ramp geometry; rolling_ramp_config.yaml overrides script defaults."""
    params = dict(_DEFAULT_RAMP)
    if ROLLING_RAMP_CONFIG.exists():
        with ROLLING_RAMP_CONFIG.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        ramp = raw.get("ramp", {})
        for key in _DEFAULT_RAMP:
            if key in ramp:
                params[key] = float(ramp[key])
    if angle_deg is not None:
        params["angle_deg"] = float(angle_deg)
    if ramp_length is not None:
        params["ramp_length"] = float(ramp_length)
    if flat_start is not None:
        params["flat_start"] = float(flat_start)
    if width is not None:
        params["width"] = float(width)
    return params


def _ramp_geometry(cage_radius: float, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    angle = np.radians(params["angle_deg"])
    normal = np.array([-np.sin(angle), 0.0, np.cos(angle)])
    along = params["ramp_length"] - TARGET_EDGE_CLEARANCE
    surface_target = np.array(
        [
            params["flat_start"] + along * np.cos(angle),
            0.0,
            along * np.sin(angle),
        ]
    )
    return surface_target + cage_radius * normal, normal


def compile_scene(
    model_path: Path,
    cage_radius: float,
    *,
    angle_deg: float | None = None,
    ramp_length: float | None = None,
    flat_start: float | None = None,
    width: float | None = None,
) -> mujoco.MjModel:
    params = load_ramp_scene_params(
        angle_deg=angle_deg,
        ramp_length=ramp_length,
        flat_start=flat_start,
        width=width,
    )
    spec = mujoco.MjSpec.from_file(str(model_path))
    world = spec.worldbody
    angle = np.radians(params["angle_deg"])
    normal = np.array([-np.sin(angle), 0.0, np.cos(angle)])
    surface_midpoint = np.array(
        [
            params["flat_start"] + 0.5 * params["ramp_length"] * np.cos(angle),
            0.0,
            0.5 * params["ramp_length"] * np.sin(angle),
        ]
    )
    center = surface_midpoint - RAMP_THICKNESS * normal
    half = -0.5 * angle
    ramp_quat = [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]

    world.add_geom(
        name="ramp_30deg",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=center.tolist(),
        quat=ramp_quat,
        size=[0.5 * params["ramp_length"] + 0.12, 0.5 * params["width"], RAMP_THICKNESS],
        rgba=[0.42, 0.52, 0.32, 1.0],
        friction=[1.20, 0.02, 0.01],
        contype=1,
        conaffinity=1,
        condim=6,
    )

    target, _ = _ramp_geometry(cage_radius, params)
    marker = world.add_body(name="ramp_hold_target", mocap=True, pos=target.tolist())
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.09, 0.0, 0.0],
        rgba=[0.10, 0.90, 0.95, 0.95],
        contype=0,
        conaffinity=0,
    )

    model = spec.compile()
    model.opt.timestep = 0.004
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="30-degree CAROLLINE ramp hold test")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "config.yaml"),
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ramp_params = load_ramp_scene_params()
    target, expected_normal = _ramp_geometry(config.cage_radius, ramp_params)
    config.initial_mode = ControlMode.ROLLING
    config.spawn_xy = np.array([0.0, 0.0])
    config.roll_target = target[:2].copy()
    config.roll_position_tolerance = 0.08

    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model = compile_scene(model_path, config.cage_radius)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)

    tilt = np.radians(50.0) * 0.5
    data.qpos[:7] = [0.0, 0.0, config.cage_radius, np.cos(tilt), 0.0, np.sin(tilt), 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    hold_time = 0.0
    hold_announced = False
    telemetry: dict[str, np.ndarray | float | bool] = {}

    print("CAROLLINE ramp hold test")
    print(f"  ramp angle: {ramp_params['angle_deg']:.1f} deg  (rolling_ramp_config.yaml)")
    print(f"  target center: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}] m")
    print(f"  expected normal: {expected_normal}")

    def control_step() -> None:
        nonlocal hold_time, hold_announced
        state = estimator.estimate(data)
        distance = float(np.linalg.norm(state.position[:2] - target[:2]))
        controller.mode_manager.request_roll_hold(distance < 0.80)
        motor, _, mode, _ = controller.compute(state, dt)
        diagnostics = controller.last_diagnostics
        data.ctrl[:] = motor.thrusts

        holding = (
            distance < config.roll_position_tolerance
            and float(np.linalg.norm(state.velocity)) < 0.12
            and state.contact_valid
        )
        hold_time = hold_time + dt if holding else 0.0
        if hold_time >= 2.0 and not hold_announced:
            print(
                f"t={state.time:.2f}s  HOLD STABLE  "
                f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})"
            )
            hold_announced = True

        telemetry.update(
            mode=mode.name,
            distance=distance,
            speed=float(np.linalg.norm(state.velocity)),
            normal=state.contact_normal_world.copy(),
            requested=diagnostics.moment_body.copy(),
            achieved=diagnostics.achievable_contact_torque.copy(),
            motors=motor.thrusts.copy(),
            scale=diagnostics.allocation_scale,
            stable=hold_time >= 2.0,
        )

    if args.no_viewer:
        while data.time < args.duration:
            control_step()
            mujoco.mj_step(model, data)
    else:
        angle = np.radians(ramp_params["angle_deg"])
        ramp_mid_x = ramp_params["flat_start"] + 0.5 * ramp_params["ramp_length"] * np.cos(angle)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [ramp_mid_x, 0.0, 1.0]
            viewer.cam.distance = 8.0
            viewer.cam.azimuth = 115.0
            viewer.cam.elevation = -18.0
            while viewer.is_running() and data.time < args.duration:
                control_step()
                normal = np.asarray(telemetry["normal"])
                requested = np.asarray(telemetry["requested"])
                achieved = np.asarray(telemetry["achieved"])
                motors = np.asarray(telemetry["motors"])
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        (
                            "30 deg ramp hold\n"
                            "Mode / stable\n"
                            "Target error / speed [m, m/s]\n"
                            "Contact normal\n"
                            "Requested torque [N.m]\n"
                            "Achieved torque [N.m]\n"
                            "Motor thrusts [N]\n"
                            "Allocation scale"
                        ),
                        (
                            "\n"
                            f"{telemetry['mode']} / {'YES' if telemetry['stable'] else 'no'}\n"
                            f"{float(telemetry['distance']):.3f} / {float(telemetry['speed']):.3f}\n"
                            f"[{normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f}]\n"
                            f"[{requested[0]:+.2f}, {requested[1]:+.2f}, {requested[2]:+.2f}]\n"
                            f"[{achieved[0]:+.2f}, {achieved[1]:+.2f}, {achieved[2]:+.2f}]\n"
                            f"[{motors[0]:+.2f}, {motors[1]:+.2f}, "
                            f"{motors[2]:+.2f}, {motors[3]:+.2f}]\n"
                            f"{float(telemetry['scale']):.2f}"
                        ),
                    )
                )
                mujoco.mj_step(model, data)
                viewer.sync()

    final_state = estimator.estimate(data)
    print(
        f"Final: mode={controller.mode_manager.mode.name} stable={hold_announced} "
        f"pos=({final_state.position[0]:.2f},{final_state.position[1]:.2f},"
        f"{final_state.position[2]:.2f}) speed={np.linalg.norm(final_state.velocity):.3f} m/s"
    )


if __name__ == "__main__":
    main()
