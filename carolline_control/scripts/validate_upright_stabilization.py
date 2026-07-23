"""
Validate upright stabilization after recovery.

Holds the cage in UPRIGHT (no takeoff) for a fixed window and reports:
  - Roll / pitch RMS and maxima [deg]
  - Angular velocity RMS and maximum [rad/s]
  - Motor command variance and saturation %

Usage (from repo root):
    python carolline_control/scripts/validate_upright_stabilization.py
    python carolline_control/scripts/validate_upright_stabilization.py --trials 30 --hold 8 --seed 42
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
class StabTrial:
    trial: int
    seed: int
    start_mode: str
    initial_tilt_deg: float
    recovered: bool
    recovery_time_s: float
    hold_time_s: float
    success: bool
    # Attitude [deg]
    roll_rms_deg: float
    pitch_rms_deg: float
    max_roll_deg: float
    max_pitch_deg: float
    mean_tilt_deg: float
    max_tilt_deg: float
    # Angular rate [rad/s]
    omega_rms: float
    omega_x_rms: float
    omega_y_rms: float
    omega_z_rms: float
    max_omega: float
    max_omega_x: float
    max_omega_y: float
    max_omega_z: float
    # Motors
    motor_variance: float
    motor_var_m1: float
    motor_var_m2: float
    motor_var_m3: float
    motor_var_m4: float
    motor_sat_pct: float
    peak_motor_abs: float
    mean_motor_spread: float
    # Contact / pose quality
    mean_bz: float
    min_bz: float
    contact_lost: bool


def _random_ground_qpos(spawn_xy: list[float], ground_z: float, rng: random.Random, tilt_range: tuple[float, float]) -> tuple[list[float], float]:
    tilt_deg = rng.uniform(tilt_range[0], tilt_range[1])
    heading = rng.uniform(0.0, 2.0 * math.pi)
    axis = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
    half = math.radians(tilt_deg) * 0.5
    qw = math.cos(half)
    qv = axis * math.sin(half)
    qpos = [
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        float(ground_z),
        float(qw),
        float(qv[0]),
        float(qv[1]),
        float(qv[2]),
    ]
    return qpos, tilt_deg


def _upright_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    return [float(spawn_xy[0]), float(spawn_xy[1]), float(ground_z), 1.0, 0.0, 0.0, 0.0]


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def run_trial(
    *,
    trial: int,
    seed: int,
    config,
    raw: dict,
    hold_s: float,
    recover_timeout_s: float,
    start_tilted: bool,
    tilt_range: tuple[float, float],
) -> StabTrial:
    rng = random.Random(seed)
    np.random.seed(seed)

    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    config.initial_mode = ControlMode.PRETAKEOFF
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.PRETAKEOFF

    spawn = raw.get("mission", {}).get("spawn_xy", [0.0, 0.0])
    ground_z = float(raw.get("ground_z", 0.40))
    if start_tilted:
        qpos, init_tilt = _random_ground_qpos(spawn, ground_z, rng, tilt_range)
        start_mode = "TILTED"
    else:
        qpos = _upright_qpos(spawn, ground_z)
        init_tilt = 0.0
        start_mode = "UPRIGHT"

    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    takeoff_cos = float(config.takeoff_cos_threshold)
    omega_tol = float(config.upright_omega_tolerance)

    # Phase 1: recover until upright-ready (or already upright)
    recovered = False
    recovery_time = float("nan")
    settle_hold = 0.0
    while data.time < recover_timeout_s:
        state = estimator.estimate(data)
        motor, cmd, mode, _ = controller.compute(state, dt)

        # Freeze at UPRIGHT — never take off during this validation
        if mode == ControlMode.TAKEOFF:
            controller.mode_manager.mode = ControlMode.UPRIGHT
            mode = ControlMode.UPRIGHT

        bz = float(body_z_world(state.rotation)[2])
        omega_n = float(np.linalg.norm(state.omega_body))
        if bz >= takeoff_cos and omega_n < omega_tol and state.on_ground:
            settle_hold += dt
            if settle_hold >= config.upright_settle_time:
                recovered = True
                recovery_time = float(state.time)
                controller.mode_manager.mode = ControlMode.UPRIGHT
                break
        else:
            settle_hold = 0.0

        data.ctrl[:] = motor.thrusts
        mujoco.mj_step(model, data)

    if not recovered:
        # Still collect a short window for diagnostics, but mark failure
        controller.mode_manager.mode = ControlMode.UPRIGHT

    # Phase 2: hold upright and measure
    rolls: list[float] = []
    pitches: list[float] = []
    tilts: list[float] = []
    bzs: list[float] = []
    omegas: list[np.ndarray] = []
    motors_hist: list[np.ndarray] = []
    sat_steps = 0
    steps = 0
    contact_lost = False
    t0 = float(data.time)

    while data.time - t0 < hold_s:
        state = estimator.estimate(data)
        # Keep mode frozen in UPRIGHT
        controller.mode_manager.mode = ControlMode.UPRIGHT
        motor, cmd, mode, _ = controller.compute(state, dt)
        if mode == ControlMode.TAKEOFF:
            controller.mode_manager.mode = ControlMode.UPRIGHT

        rpy = rot_to_euler_zyx(state.rotation)
        roll_deg = float(np.degrees(rpy[0]))
        pitch_deg = float(np.degrees(rpy[1]))
        bz = float(body_z_world(state.rotation)[2])
        tilt_deg = float(np.degrees(np.arccos(np.clip(bz, -1.0, 1.0))))

        rolls.append(roll_deg)
        pitches.append(pitch_deg)
        tilts.append(tilt_deg)
        bzs.append(bz)
        omegas.append(state.omega_body.copy())

        thrusts = np.asarray(motor.thrusts, dtype=float)
        motors_hist.append(thrusts.copy())
        if np.any(thrusts >= config.motor_max - 0.05) or np.any(thrusts <= config.motor_min + 0.05):
            sat_steps += 1
        steps += 1

        if not state.on_ground and state.position[2] > ground_z + 0.15:
            contact_lost = True

        data.ctrl[:] = thrusts
        mujoco.mj_step(model, data)

    rolls_a = np.asarray(rolls, dtype=float)
    pitches_a = np.asarray(pitches, dtype=float)
    tilts_a = np.asarray(tilts, dtype=float)
    bzs_a = np.asarray(bzs, dtype=float)
    omega_a = np.asarray(omegas, dtype=float) if omegas else np.zeros((0, 3))
    motors_a = np.asarray(motors_hist, dtype=float) if motors_hist else np.zeros((0, 4))

    omega_norm = np.linalg.norm(omega_a, axis=1) if omega_a.size else np.array([])
    motor_vars = np.var(motors_a, axis=0) if motors_a.size else np.full(4, np.nan)
    spreads = np.max(motors_a, axis=1) - np.min(motors_a, axis=1) if motors_a.size else np.array([])

    # Stabilization success: recovered + stayed near upright during hold
    success = (
        recovered
        and not contact_lost
        and float(np.mean(bzs_a)) >= takeoff_cos
        and float(np.max(tilts_a)) < 15.0
        and float(np.max(omega_norm)) < 1.0
    )

    return StabTrial(
        trial=trial,
        seed=seed,
        start_mode=start_mode,
        initial_tilt_deg=init_tilt,
        recovered=recovered,
        recovery_time_s=recovery_time,
        hold_time_s=float(data.time - t0),
        success=success,
        roll_rms_deg=_rms(rolls_a),
        pitch_rms_deg=_rms(pitches_a),
        max_roll_deg=float(np.max(np.abs(rolls_a))) if rolls_a.size else float("nan"),
        max_pitch_deg=float(np.max(np.abs(pitches_a))) if pitches_a.size else float("nan"),
        mean_tilt_deg=float(np.mean(tilts_a)) if tilts_a.size else float("nan"),
        max_tilt_deg=float(np.max(tilts_a)) if tilts_a.size else float("nan"),
        omega_rms=_rms(omega_norm),
        omega_x_rms=_rms(omega_a[:, 0]) if omega_a.size else float("nan"),
        omega_y_rms=_rms(omega_a[:, 1]) if omega_a.size else float("nan"),
        omega_z_rms=_rms(omega_a[:, 2]) if omega_a.size else float("nan"),
        max_omega=float(np.max(omega_norm)) if omega_norm.size else float("nan"),
        max_omega_x=float(np.max(np.abs(omega_a[:, 0]))) if omega_a.size else float("nan"),
        max_omega_y=float(np.max(np.abs(omega_a[:, 1]))) if omega_a.size else float("nan"),
        max_omega_z=float(np.max(np.abs(omega_a[:, 2]))) if omega_a.size else float("nan"),
        motor_variance=float(np.mean(motor_vars)),
        motor_var_m1=float(motor_vars[0]),
        motor_var_m2=float(motor_vars[1]),
        motor_var_m3=float(motor_vars[2]),
        motor_var_m4=float(motor_vars[3]),
        motor_sat_pct=100.0 * sat_steps / max(steps, 1),
        peak_motor_abs=float(np.max(np.abs(motors_a))) if motors_a.size else float("nan"),
        mean_motor_spread=float(np.mean(spreads)) if spreads.size else float("nan"),
        mean_bz=float(np.mean(bzs_a)) if bzs_a.size else float("nan"),
        min_bz=float(np.min(bzs_a)) if bzs_a.size else float("nan"),
        contact_lost=contact_lost,
    )


def _print_trial(r: StabTrial) -> None:
    status = "PASS" if r.success else "FAIL"
    print(
        f"  [{r.trial:03d}] {status}  tilt0={r.initial_tilt_deg:5.1f}deg  "
        f"rollRMS={r.roll_rms_deg:5.2f}  pitchRMS={r.pitch_rms_deg:5.2f}  "
        f"maxR/P={r.max_roll_deg:5.2f}/{r.max_pitch_deg:5.2f}  "
        f"wRMS={r.omega_rms:5.3f}  max|w|={r.max_omega:5.3f}  "
        f"motorVar={r.motor_variance:6.3f}  sat={r.motor_sat_pct:4.1f}%"
        + ("  NO_RECOVERY" if not r.recovered else "")
        + ("  CONTACT_LOST" if r.contact_lost else "")
    )


def _agg(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return f"mean={a.mean():.3f}  median={np.median(a):.3f}  std={std:.3f}  min={a.min():.3f}  max={a.max():.3f}"


def _summary(results: list[StabTrial]) -> None:
    n = len(results)
    ok = [r for r in results if r.success]
    print("\n" + "=" * 78)
    print("UPRIGHT STABILIZATION VALIDATION SUMMARY")
    print("=" * 78)
    print(f"  Trials              : {n}")
    print(f"  Success rate        : {100.0 * len(ok) / max(n, 1):.1f}%  ({len(ok)}/{n})")
    print(f"  Recovered           : {sum(1 for r in results if r.recovered)}/{n}")
    print(f"  Contact lost        : {sum(1 for r in results if r.contact_lost)}/{n}")
    print("")
    print("  Attitude [deg]")
    print(f"    Roll RMS          : {_agg([r.roll_rms_deg for r in results])}")
    print(f"    Pitch RMS         : {_agg([r.pitch_rms_deg for r in results])}")
    print(f"    Max |roll|        : {_agg([r.max_roll_deg for r in results])}")
    print(f"    Max |pitch|       : {_agg([r.max_pitch_deg for r in results])}")
    print(f"    Mean tilt         : {_agg([r.mean_tilt_deg for r in results])}")
    print(f"    Max tilt          : {_agg([r.max_tilt_deg for r in results])}")
    print("")
    print("  Angular velocity [rad/s]")
    print(f"    |w| RMS           : {_agg([r.omega_rms for r in results])}")
    print(f"    wx / wy / wz RMS  : "
          f"{np.nanmean([r.omega_x_rms for r in results]):.3f} / "
          f"{np.nanmean([r.omega_y_rms for r in results]):.3f} / "
          f"{np.nanmean([r.omega_z_rms for r in results]):.3f}")
    print(f"    Max |w|           : {_agg([r.max_omega for r in results])}")
    print(f"    Max |wx|/|wy|/|wz|: "
          f"{np.nanmean([r.max_omega_x for r in results]):.3f} / "
          f"{np.nanmean([r.max_omega_y for r in results]):.3f} / "
          f"{np.nanmean([r.max_omega_z for r in results]):.3f}")
    print("")
    print("  Motors")
    print(f"    Variance (mean)   : {_agg([r.motor_variance for r in results])}")
    print(f"    Var m1..m4 (mean) : "
          f"{np.nanmean([r.motor_var_m1 for r in results]):.3f}, "
          f"{np.nanmean([r.motor_var_m2 for r in results]):.3f}, "
          f"{np.nanmean([r.motor_var_m3 for r in results]):.3f}, "
          f"{np.nanmean([r.motor_var_m4 for r in results]):.3f}")
    print(f"    Saturation %      : {_agg([r.motor_sat_pct for r in results])}")
    print(f"    Peak |motor|      : {_agg([r.peak_motor_abs for r in results])}")
    print(f"    Mean motor spread : {_agg([r.mean_motor_spread for r in results])}")
    print("")
    print(f"  Body-z up mean/min  : {_agg([r.mean_bz for r in results])} / {_agg([r.min_bz for r in results])}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate upright stabilization metrics")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hold", type=float, default=8.0, help="Upright hold window [s]")
    parser.add_argument("--recover-timeout", type=float, default=20.0)
    parser.add_argument(
        "--start",
        choices=["tilted", "upright", "mixed"],
        default="tilted",
        help="Initial pose: random tilt, already upright, or mix",
    )
    parser.add_argument("--tilt-min", type=float, default=25.0)
    parser.add_argument("--tilt-max", type=float, default=160.0)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "carolline_control" / "logs" / "upright_stabilization.csv"),
    )
    args = parser.parse_args()

    raw = load_raw_config(args.config)
    cfg0 = load_config(args.config)

    print("CAROLLINE upright stabilization validation")
    print(f"  model       : {cfg0.model_path}")
    print(f"  trials      : {args.trials}")
    print(f"  hold window : {args.hold}s")
    print(f"  start       : {args.start}")
    print(f"  gains       : kR_pre={cfg0.kR_pre}  kOmega_pre={cfg0.kOmega_pre}  thrust_frac={cfg0.pre_takeoff_thrust_fraction}")
    print(f"  takeoff cos : {cfg0.takeoff_cos_threshold}  settle={cfg0.upright_settle_time}s")
    print("-" * 78)

    results: list[StabTrial] = []
    for i in range(args.trials):
        cfg = load_config(args.config)
        cfg.initial_mode = ControlMode.PRETAKEOFF
        if args.start == "upright":
            start_tilted = False
        elif args.start == "tilted":
            start_tilted = True
        else:
            start_tilted = (i % 2 == 0)

        r = run_trial(
            trial=i + 1,
            seed=args.seed + i,
            config=cfg,
            raw=raw,
            hold_s=args.hold,
            recover_timeout_s=args.recover_timeout,
            start_tilted=start_tilted,
            tilt_range=(args.tilt_min, args.tilt_max),
        )
        results.append(r)
        _print_trial(r)

    _summary(results)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
