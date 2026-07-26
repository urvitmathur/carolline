"""
Energy efficiency benchmark (CAROLLINE paper, Sec. V-C, Table II).

Compares cost of transport (COT) for straight-line locomotion at 0.5 m/s over
10 m on the ground (rolling) versus in the air (flight).

Usage (from repo root):
    python carolline_control/scripts/validate_energy_cot.py
    python carolline_control/scripts/validate_energy_cot.py --speed 0.5 --distance 10
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.types import ControlMode, TrajectoryTarget
from carolline_control.visualization.markers import compile_model_with_markers


@dataclass
class LocomotionResult:
    mode: str
    distance_m: float
    duration_s: float
    avg_speed_mps: float
    total_energy_j: float
    avg_power_w: float
    cot: float


def _rolling_cruise_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    """Side-roll attitude stable for +X translation (paper Sec. V-C setup)."""
    half = math.pi * 0.25  # 90 deg about body Y
    return [
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        float(ground_z),
        float(math.cos(half)),
        0.0,
        float(math.sin(half)),
        0.0,
    ]


def _tilted_ground_qpos(spawn_xy: list[float], ground_z: float, tilt_deg: float = 55.0) -> list[float]:
    half = math.radians(tilt_deg) * 0.5
    return [
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        float(ground_z),
        float(math.cos(half)),
        float(math.sin(half)),
        0.0,
        0.0,
    ]


def _integrate_energy(motor_thrusts: np.ndarray, dt: float) -> float:
    """Mechanical thrust proxy consistent with validation/metrics.py."""
    return float(np.sum(np.abs(motor_thrusts)) * dt)


def _run_rolling(
    *,
    config_path: Path,
    speed: float,
    distance: float,
    spawn_xy: list[float],
    ground_z: float,
    warmup_distance: float = 1.0,
) -> LocomotionResult:
    config = load_config(config_path)
    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.ROLLING
    controller.rolling.reset()

    data.qpos[:7] = np.asarray(_tilted_ground_qpos(spawn_xy, ground_z, tilt_deg=55.0), dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    direction = np.array([1.0, 0.0], dtype=float)
    start_xy = np.array(spawn_xy[:2], dtype=float)
    energy = 0.0
    measure_energy = 0.0
    duration = 0.0
    measure_duration = 0.0
    timeout = max(120.0, 4.0 * distance / max(speed, 0.05))
    measure_start = warmup_distance
    measure_end = warmup_distance + distance
    step = 0

    while duration < timeout:
        state = estimator.estimate(data)
        traveled = float(np.dot(state.position[:2] - start_xy, direction))
        if step >= int(0.5 / dt) and traveled >= measure_end:
            break
        step += 1

        velocity_cmd = direction * speed
        cmd = controller.rolling.compute(state, velocity_cmd, yaw=0.0, dt=dt)
        motor = controller.rolling.allocate(state, cmd).motor
        if config.esc_mapping_enabled:
            motor = controller.esc_mapper.apply(motor)
        data.ctrl[:] = motor.thrusts
        step_energy = _integrate_energy(motor.thrusts, dt)
        energy += step_energy
        if traveled >= measure_start:
            measure_energy += step_energy
            measure_duration += dt
        mujoco.mj_step(model, data)
        duration += dt

    final_state = estimator.estimate(data)
    traveled = float(np.dot(final_state.position[:2] - start_xy, direction))
    measured_distance = max(traveled - measure_start, 1e-6)
    cot = measure_energy / max(config.mass * config.gravity * measured_distance, 1e-6)
    return LocomotionResult(
        mode="terrestrial",
        distance_m=measured_distance,
        duration_s=measure_duration if measure_duration > 0 else duration,
        avg_speed_mps=measured_distance / max(measure_duration, 1e-6),
        total_energy_j=measure_energy if measure_energy > 0 else energy,
        avg_power_w=(measure_energy if measure_energy > 0 else energy)
        / max(measure_duration if measure_duration > 0 else duration, 1e-6),
        cot=cot,
    )


def _run_flight(
    *,
    config_path: Path,
    speed: float,
    distance: float,
    spawn_xy: list[float],
    cruise_height: float,
    warmup_distance: float = 1.0,
) -> LocomotionResult:
    config = load_config(config_path)
    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.FLIGHT
    controller._flight_yaw = 0.0
    controller._hover_anchor_xy = np.array(spawn_xy[:2], dtype=float)

    start = np.array([float(spawn_xy[0]), float(spawn_xy[1]), cruise_height], dtype=float)
    data.qpos[:3] = start
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    energy = 0.0
    measure_energy = 0.0
    duration = 0.0
    measure_duration = 0.0
    timeout = max(60.0, 3.0 * distance / max(speed, 0.05))
    measure_start = warmup_distance
    measure_end = warmup_distance + distance

    while duration < timeout:
        state = estimator.estimate(data)
        traveled = float(np.dot(state.position - start, direction))
        if traveled >= measure_end:
            break

        target_pos = start + direction * min(traveled + speed * dt, measure_end)
        target = TrajectoryTarget(
            position=target_pos,
            velocity=direction * speed,
            acceleration=np.zeros(3),
            yaw=0.0,
        )
        cmd = controller.flight.compute(state, target, ControlMode.FLIGHT)
        omega_d = controller.attitude.compute_desired_omega(
            state.rotation,
            cmd.desired_rotation,
            config.kR,
        )
        moment = controller.torque.compute(
            state.rotation,
            cmd.desired_rotation,
            state.omega_body,
            omega_d,
            config.inertia,
            config.kR,
            config.kOmega,
        )
        motor = controller.mixer.mix(cmd.thrust, moment)
        if config.esc_mapping_enabled:
            motor = controller.esc_mapper.apply(motor)
        data.ctrl[:] = motor.thrusts
        step_energy = _integrate_energy(motor.thrusts, dt)
        energy += step_energy
        if traveled >= measure_start:
            measure_energy += step_energy
            measure_duration += dt
        mujoco.mj_step(model, data)
        duration += dt

    final_state = estimator.estimate(data)
    traveled = float(np.dot(final_state.position - start, direction))
    measured_distance = max(traveled - measure_start, 1e-6)
    cot = measure_energy / max(config.mass * config.gravity * measured_distance, 1e-6)
    return LocomotionResult(
        mode="aerial",
        distance_m=measured_distance,
        duration_s=measure_duration if measure_duration > 0 else duration,
        avg_speed_mps=measured_distance / max(measure_duration, 1e-6),
        total_energy_j=measure_energy if measure_energy > 0 else energy,
        avg_power_w=(measure_energy if measure_energy > 0 else energy)
        / max(measure_duration if measure_duration > 0 else duration, 1e-6),
        cot=cot,
    )


def _print_result(result: LocomotionResult) -> None:
    print(
        f"  {result.mode:11s}  dist={result.distance_m:5.2f} m  "
        f"speed={result.avg_speed_mps:4.2f} m/s  "
        f"energy={result.total_energy_j:8.1f} J  "
        f"power={result.avg_power_w:5.2f} W  "
        f"COT={result.cot:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground vs flight energy / COT benchmark")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--speed", type=float, default=0.5, help="Target speed [m/s]")
    parser.add_argument("--distance", type=float, default=10.0, help="Travel distance [m]")
    parser.add_argument(
        "--warmup-distance",
        type=float,
        default=1.0,
        help="Distance [m] before steady-state COT measurement begins",
    )
    parser.add_argument(
        "--cruise-height",
        type=float,
        default=1.0,
        help="Flight cruise height [m]",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = load_raw_config(config_path)
    mission = raw.get("mission", {})
    spawn_xy = mission.get("spawn_xy", [0.0, 0.0])
    ground_z = float(raw.get("ground_z", 0.40))

    print("CAROLLINE energy efficiency benchmark (paper Sec. V-C)")
    print(f"  speed={args.speed} m/s  distance={args.distance} m  warmup={args.warmup_distance} m")
    print()

    ground = _run_rolling(
        config_path=config_path,
        speed=args.speed,
        distance=args.distance,
        spawn_xy=spawn_xy,
        ground_z=ground_z,
        warmup_distance=args.warmup_distance,
    )
    flight = _run_flight(
        config_path=config_path,
        speed=args.speed,
        distance=args.distance,
        spawn_xy=spawn_xy,
        cruise_height=args.cruise_height,
        warmup_distance=args.warmup_distance,
    )

    print("Results:")
    _print_result(ground)
    _print_result(flight)
    ratio = flight.cot / max(ground.cot, 1e-9)
    print()
    print(f"  Aerial / terrestrial COT ratio: {ratio:.2f}x")
    print(f"  (Paper reports ~12.6x lower COT on ground at 0.5 m/s over 10 m)")


if __name__ == "__main__":
    main()
