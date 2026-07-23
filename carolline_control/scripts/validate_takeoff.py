"""
Validate takeoff: UPRIGHT -> TAKEOFF -> HOVER.

Per-trial metrics:
  Takeoff Success, Takeoff Delay, Time to Hover,
  Max Tilt / Roll / Pitch during takeoff,
  Max Horizontal Drift, Max Vertical Overshoot,
  Max Vertical Velocity, Max Angular Velocity,
  Peak Motor Thrust, Motor Saturation %,
  Final Hover Altitude, Hover Achieved

Usage (from repo root):
    python carolline_control/scripts/validate_takeoff.py
    python carolline_control/scripts/validate_takeoff.py --trials 30 --seed 42 --hover-hold 3
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.so3 import body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


@dataclass
class TakeoffTrial:
    trial: int
    seed: int
    initial_tilt_deg: float
    takeoff_success: bool
    hover_achieved: bool
    takeoff_delay_s: float
    time_to_hover_s: float
    max_tilt_deg: float
    max_roll_deg: float
    max_pitch_deg: float
    max_horizontal_drift_m: float
    max_vertical_overshoot_m: float
    max_vertical_velocity: float
    max_angular_velocity: float
    peak_motor_thrust: float
    motor_saturation_pct: float
    final_hover_altitude_m: float
    liftoff_time_s: float
    recovery_time_s: float
    timeout: bool


def _random_ground_qpos(
    spawn_xy: list[float],
    ground_z: float,
    rng: random.Random,
    tilt_range: tuple[float, float],
) -> tuple[list[float], float]:
    tilt_deg = rng.uniform(tilt_range[0], tilt_range[1])
    heading = rng.uniform(0.0, 2.0 * math.pi)
    axis = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
    half = math.radians(tilt_deg) * 0.5
    qw = math.cos(half)
    qv = axis * math.sin(half)
    return (
        [
            float(spawn_xy[0]),
            float(spawn_xy[1]),
            float(ground_z),
            float(qw),
            float(qv[0]),
            float(qv[1]),
            float(qv[2]),
        ],
        tilt_deg,
    )


def _upright_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    return [float(spawn_xy[0]), float(spawn_xy[1]), float(ground_z), 1.0, 0.0, 0.0, 0.0]


def run_trial(
    *,
    trial: int,
    seed: int,
    config,
    raw: dict,
    timeout_s: float,
    hover_hold_s: float,
    start_tilted: bool,
    tilt_range: tuple[float, float],
) -> TakeoffTrial:
    rng = random.Random(seed)
    np.random.seed(seed)

    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    config.initial_mode = ControlMode.PRETAKEOFF
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.PRETAKEOFF

    spawn = np.asarray(raw.get("mission", {}).get("spawn_xy", [0.0, 0.0]), dtype=float)
    ground_z = float(raw.get("ground_z", 0.40))
    if start_tilted:
        qpos, init_tilt = _random_ground_qpos(spawn.tolist(), ground_z, rng, tilt_range)
    else:
        qpos = _upright_qpos(spawn.tolist(), ground_z)
        init_tilt = 0.0

    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    hover_h = float(config.hover_height)
    hover_alt_tol = float(config.hover_altitude_tolerance)

    # Timeline markers
    t_upright_entry: float | None = None
    t_takeoff_entry: float | None = None
    t_liftoff: float | None = None
    t_hover_entry: float | None = None
    hover_hold_start: float | None = None

    # Takeoff-window accumulators (TAKEOFF mode only, until HOVER)
    rolls: list[float] = []
    pitches: list[float] = []
    tilts: list[float] = []
    drifts: list[float] = []
    vz_list: list[float] = []
    omegas: list[float] = []
    peak_motor = 0.0
    sat_steps = 0
    takeoff_steps = 0
    max_z = ground_z
    takeoff_xy0: np.ndarray | None = None

    last_mode = ControlMode.PRETAKEOFF
    timed_out = False
    final_alt = float("nan")

    while data.time < timeout_s:
        state = estimator.estimate(data)
        motor, cmd, mode, _ = controller.compute(state, dt)

        # Freeze at HOVER — do not enter FLIGHT
        if mode == ControlMode.FLIGHT:
            controller.mode_manager.mode = ControlMode.HOVER
            mode = ControlMode.HOVER

        # Mode transition timestamps
        if mode == ControlMode.UPRIGHT and last_mode != ControlMode.UPRIGHT:
            t_upright_entry = float(state.time)
        if mode == ControlMode.TAKEOFF and last_mode != ControlMode.TAKEOFF:
            t_takeoff_entry = float(state.time)
            takeoff_xy0 = state.position[:2].copy()
        if mode == ControlMode.HOVER and last_mode != ControlMode.HOVER:
            t_hover_entry = float(state.time)
            hover_hold_start = float(state.time)

        # Liftoff: leave ground after takeoff starts
        if (
            t_takeoff_entry is not None
            and t_liftoff is None
            and not state.on_ground
            and state.position[2] > ground_z + 0.05
        ):
            t_liftoff = float(state.time)

        thrusts = np.asarray(motor.thrusts, dtype=float)

        # Metrics during TAKEOFF (and brief first moments of HOVER entry)
        if mode == ControlMode.TAKEOFF or (
            mode == ControlMode.HOVER and t_hover_entry is not None and state.time - t_hover_entry < 0.05
        ):
            rpy = rot_to_euler_zyx(state.rotation)
            roll_deg = float(np.degrees(rpy[0]))
            pitch_deg = float(np.degrees(rpy[1]))
            bz = float(body_z_world(state.rotation)[2])
            tilt_deg = float(np.degrees(np.arccos(np.clip(bz, -1.0, 1.0))))
            rolls.append(abs(roll_deg))
            pitches.append(abs(pitch_deg))
            tilts.append(tilt_deg)
            if takeoff_xy0 is not None:
                drifts.append(float(np.linalg.norm(state.position[:2] - takeoff_xy0)))
            else:
                drifts.append(float(np.linalg.norm(state.position[:2] - spawn)))
            vz_list.append(float(state.velocity[2]))
            omegas.append(float(np.linalg.norm(state.omega_body)))
            peak_motor = max(peak_motor, float(np.max(np.abs(thrusts))))
            if np.any(thrusts >= config.motor_max - 0.05) or np.any(thrusts <= config.motor_min + 0.05):
                sat_steps += 1
            takeoff_steps += 1
            max_z = max(max_z, float(state.position[2]))

        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)
        last_mode = mode

        if mode == ControlMode.HOVER and hover_hold_start is not None:
            if state.time - hover_hold_start >= hover_hold_s:
                final_alt = float(state.position[2])
                break

        if state.position[2] < -0.5 or state.position[2] > hover_h + 2.0:
            timed_out = True
            final_alt = float(state.position[2])
            break
    else:
        timed_out = True
        final_alt = float(estimator.estimate(data).position[2])

    hover_achieved = t_hover_entry is not None and not timed_out
    # Success: reached hover and stayed near target altitude at end of hold
    takeoff_success = (
        hover_achieved
        and math.isfinite(final_alt)
        and abs(final_alt - hover_h) < max(hover_alt_tol * 2.0, 0.2)
    )

    takeoff_delay = (
        float(t_takeoff_entry - t_upright_entry)
        if t_takeoff_entry is not None and t_upright_entry is not None
        else float("nan")
    )
    time_to_hover = (
        float(t_hover_entry - t_takeoff_entry)
        if t_hover_entry is not None and t_takeoff_entry is not None
        else float("nan")
    )
    liftoff_time = (
        float(t_liftoff - t_takeoff_entry)
        if t_liftoff is not None and t_takeoff_entry is not None
        else float("nan")
    )
    recovery_time = float(t_upright_entry) if t_upright_entry is not None else float("nan")

    overshoot = max(0.0, max_z - hover_h) if takeoff_steps else float("nan")

    return TakeoffTrial(
        trial=trial,
        seed=seed,
        initial_tilt_deg=init_tilt,
        takeoff_success=takeoff_success,
        hover_achieved=hover_achieved,
        takeoff_delay_s=takeoff_delay,
        time_to_hover_s=time_to_hover,
        max_tilt_deg=float(np.max(tilts)) if tilts else float("nan"),
        max_roll_deg=float(np.max(rolls)) if rolls else float("nan"),
        max_pitch_deg=float(np.max(pitches)) if pitches else float("nan"),
        max_horizontal_drift_m=float(np.max(drifts)) if drifts else float("nan"),
        max_vertical_overshoot_m=overshoot,
        max_vertical_velocity=float(np.max(vz_list)) if vz_list else float("nan"),
        max_angular_velocity=float(np.max(omegas)) if omegas else float("nan"),
        peak_motor_thrust=peak_motor if takeoff_steps else float("nan"),
        motor_saturation_pct=100.0 * sat_steps / max(takeoff_steps, 1),
        final_hover_altitude_m=final_alt,
        liftoff_time_s=liftoff_time,
        recovery_time_s=recovery_time,
        timeout=timed_out,
    )


def _print_trial(r: TakeoffTrial) -> None:
    status = "PASS" if r.takeoff_success else "FAIL"
    hover = "Yes" if r.hover_achieved else "No"
    delay = f"{r.takeoff_delay_s:5.2f}" if math.isfinite(r.takeoff_delay_s) else "  n/a"
    tth = f"{r.time_to_hover_s:5.2f}" if math.isfinite(r.time_to_hover_s) else "  n/a"
    print(
        f"  [{r.trial:03d}] {status}  hover={hover}  "
        f"delay={delay}s  t_hover={tth}s  "
        f"tilt_max={r.max_tilt_deg:5.1f}  "
        f"R/P_max={r.max_roll_deg:5.1f}/{r.max_pitch_deg:5.1f}  "
        f"drift={r.max_horizontal_drift_m:5.3f}m  "
        f"ovshoot={r.max_vertical_overshoot_m:5.3f}m  "
        f"vz_max={r.max_vertical_velocity:5.2f}  "
        f"|w|_max={r.max_angular_velocity:5.2f}  "
        f"peakMot={r.peak_motor_thrust:5.1f}  sat={r.motor_saturation_pct:4.1f}%  "
        f"alt={r.final_hover_altitude_m:5.2f}m"
        + ("  TIMEOUT" if r.timeout else "")
    )


def _agg(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return (
        f"mean={a.mean():.3f}  median={np.median(a):.3f}  "
        f"std={std:.3f}  min={a.min():.3f}  max={a.max():.3f}"
    )


def _summary(results: list[TakeoffTrial], hover_h: float) -> None:
    n = len(results)
    ok = [r for r in results if r.takeoff_success]
    hover_yes = sum(1 for r in results if r.hover_achieved)
    print("\n" + "=" * 78)
    print("TAKEOFF VALIDATION SUMMARY")
    print("=" * 78)
    print(f"  Trials                 : {n}")
    print(f"  Takeoff success        : {100.0 * len(ok) / max(n, 1):.1f}%  ({len(ok)}/{n})")
    print(f"  Hover achieved         : {100.0 * hover_yes / max(n, 1):.1f}%  ({hover_yes}/{n})")
    print(f"  Timeouts               : {sum(1 for r in results if r.timeout)}/{n}")
    print(f"  Target hover altitude  : {hover_h:.2f} m")
    print("")
    print(f"  Takeoff delay [s]      : {_agg([r.takeoff_delay_s for r in results])}")
    print(f"  Time to hover [s]      : {_agg([r.time_to_hover_s for r in results])}")
    print(f"  Liftoff delay [s]      : {_agg([r.liftoff_time_s for r in results])}")
    print("")
    print(f"  Max tilt [deg]         : {_agg([r.max_tilt_deg for r in results])}")
    print(f"  Max |roll| [deg]       : {_agg([r.max_roll_deg for r in results])}")
    print(f"  Max |pitch| [deg]      : {_agg([r.max_pitch_deg for r in results])}")
    print(f"  Max horiz. drift [m]   : {_agg([r.max_horizontal_drift_m for r in results])}")
    print(f"  Max vert. overshoot [m]: {_agg([r.max_vertical_overshoot_m for r in results])}")
    print(f"  Max vz [m/s]           : {_agg([r.max_vertical_velocity for r in results])}")
    print(f"  Max |w| [rad/s]        : {_agg([r.max_angular_velocity for r in results])}")
    print("")
    print(f"  Peak motor thrust [N]  : {_agg([r.peak_motor_thrust for r in results])}")
    print(f"  Motor saturation [%]   : {_agg([r.motor_saturation_pct for r in results])}")
    print(f"  Final hover alt [m]    : {_agg([r.final_hover_altitude_m for r in results])}")
    print("=" * 78)

    # Compact table header reminder
    print("\nMetric definitions:")
    print("  Takeoff Delay     = UPRIGHT entry -> TAKEOFF entry (settle gate)")
    print("  Time to Hover     = TAKEOFF entry -> HOVER entry")
    print("  Liftoff Delay     = TAKEOFF entry -> leave ground")
    print("  Vertical Overshoot= max(z) - hover_height during takeoff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate takeoff into hover")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--hover-hold", type=float, default=3.0, help="Hold in HOVER before ending trial [s]")
    parser.add_argument(
        "--start",
        choices=["tilted", "upright", "mixed"],
        default="tilted",
    )
    parser.add_argument("--tilt-min", type=float, default=20.0)
    parser.add_argument("--tilt-max", type=float, default=120.0)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "carolline_control" / "logs" / "takeoff_validation.csv"),
    )
    args = parser.parse_args()

    raw = load_raw_config(args.config)
    cfg0 = load_config(args.config)

    print("CAROLLINE takeoff validation")
    print(f"  model        : {cfg0.model_path}")
    print(f"  trials       : {args.trials}")
    print(f"  hover height : {cfg0.hover_height} m")
    print(f"  hover hold   : {args.hover_hold}s")
    print(f"  start        : {args.start}")
    print(f"  takeoff cos  : {cfg0.takeoff_cos_threshold}  settle={cfg0.upright_settle_time}s")
    print("-" * 78)

    results: list[TakeoffTrial] = []
    for i in range(args.trials):
        cfg = load_config(args.config)
        cfg.initial_mode = ControlMode.PRETAKEOFF
        if args.start == "upright":
            start_tilted = False
        elif args.start == "tilted":
            start_tilted = True
        else:
            start_tilted = i % 2 == 0

        r = run_trial(
            trial=i + 1,
            seed=args.seed + i,
            config=cfg,
            raw=raw,
            timeout_s=args.timeout,
            hover_hold_s=args.hover_hold,
            start_tilted=start_tilted,
            tilt_range=(args.tilt_min, args.tilt_max),
        )
        results.append(r)
        _print_trial(r)

    _summary(results, cfg0.hover_height)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["hover_achieved"] = "Yes" if r.hover_achieved else "No"
            writer.writerow(row)
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
