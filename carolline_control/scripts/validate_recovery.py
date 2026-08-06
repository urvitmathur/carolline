"""
Validate upright recovery (PRETAKEOFF controller) from arbitrary orientations.

Spawns the cage at many random ground attitudes, runs recovery until upright
(or timeout), and prints per-trial + aggregate metrics.

Usage (from repo root):
    python carolline_control/scripts/validate_recovery.py
    python carolline_control/scripts/validate_recovery.py --trials 50 --seed 42 --no-viewer
"""

from __future__ import annotations

import argparse
import csv
import math
import random
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
from carolline_control.sim_estimator import (
    add_sensor_only_argument,
    build_estimator,
    print_estimator_mode,
    seed_estimator_from_sim,
)
from carolline_control.utils.so3 import attitude_error, body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


@dataclass
class TrialResult:
    trial: int
    seed: int
    initial_tilt_deg: float
    initial_bz: float
    initial_roll_deg: float
    initial_pitch_deg: float
    initial_yaw_deg: float
    success: bool
    recovery_time_s: float
    settle_time_s: float
    final_tilt_deg: float
    final_bz: float
    final_omega_norm: float
    max_omega_norm: float
    max_att_error: float
    mean_att_error: float
    peak_motor_abs: float
    motor_sat_pct: float
    contact_lost: bool
    final_mode: str
    timeout: bool


def _random_ground_qpos(spawn_xy: list[float], ground_z: float, rng: random.Random) -> tuple[list[float], float]:
    """Random cage attitude on the ground; returns (qpos, initial_tilt_deg)."""
    tilt_deg = rng.uniform(20.0, 175.0)
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


def _euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    # MuJoCo quat is [w,x,y,z]
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    rpy = rot_to_euler_zyx(R)
    return float(np.degrees(rpy[0])), float(np.degrees(rpy[1])), float(np.degrees(rpy[2]))


def run_trial(
    *,
    trial: int,
    seed: int,
    config,
    raw: dict,
    timeout_s: float,
    sensor_only: bool = False,
) -> TrialResult:
    rng = random.Random(seed)
    np.random.seed(seed)

    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    estimator = build_estimator(model, config, sensor_only=sensor_only)

    # Force recovery mission: start in PRETAKEOFF and freeze flight transitions.
    config.initial_mode = ControlMode.PRETAKEOFF
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.PRETAKEOFF

    spawn = raw.get("mission", {}).get("spawn_xy", [0.0, 0.0])
    ground_z = float(raw.get("ground_z", 0.40))
    qpos, init_tilt = _random_ground_qpos(spawn, ground_z, rng)
    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    seed_estimator_from_sim(estimator, data)

    roll0, pitch0, yaw0 = _euler_from_quat(data.qpos[3:7])
    state0 = estimator.estimate(data)
    bz0 = float(body_z_world(state0.rotation)[2])

    dt = float(model.opt.timestep)
    upright_cos = float(config.upright_cos_threshold)
    takeoff_cos = float(config.takeoff_cos_threshold)
    omega_tol = float(config.upright_omega_tolerance)

    recovery_time: float | None = None  # first time bz >= upright_cos & omega ok
    settle_time: float | None = None  # first time bz >= takeoff_cos & omega ok for settle window
    settle_hold = 0.0
    max_omega = 0.0
    att_errs: list[float] = []
    peak_motor = 0.0
    sat_steps = 0
    steps = 0
    contact_lost = False
    timed_out = False

    while data.time < timeout_s:
        state = estimator.estimate(data)
        motor, cmd, mode, target = controller.compute(state, dt)

        bz = float(body_z_world(state.rotation)[2])
        omega_n = float(np.linalg.norm(state.omega_body))
        max_omega = max(max_omega, omega_n)
        e_R = attitude_error(state.rotation, cmd.desired_rotation)
        att_errs.append(float(np.linalg.norm(e_R)))
        thrusts = np.asarray(motor.thrusts, dtype=float)
        peak_motor = max(peak_motor, float(np.max(np.abs(thrusts))))
        if np.any(thrusts >= config.motor_max - 0.05) or np.any(thrusts <= config.motor_min + 0.05):
            sat_steps += 1
        steps += 1

        if not state.on_ground and state.position[2] > ground_z + 0.15:
            contact_lost = True

        omega_ok = omega_n < omega_tol
        if recovery_time is None and bz >= upright_cos and omega_ok:
            recovery_time = float(state.time)

        # Mode manager advances PRETAKEOFF -> UPRIGHT -> TAKEOFF after settle.
        if mode == ControlMode.TAKEOFF:
            if settle_time is None:
                settle_time = float(state.time)
            break

        if bz >= takeoff_cos and omega_ok and state.on_ground:
            settle_hold += dt
            if settle_time is None and settle_hold >= config.upright_settle_time:
                settle_time = float(state.time)
                break
        else:
            settle_hold = 0.0

        data.ctrl[:] = motor.thrusts
        mujoco.mj_step(model, data)

        if state.position[2] < -0.5:
            timed_out = True
            break
    else:
        timed_out = True

    state_f = estimator.estimate(data)
    bz_f = float(body_z_world(state_f.rotation)[2])
    tilt_f = float(np.degrees(np.arccos(np.clip(bz_f, -1.0, 1.0))))
    omega_f = float(np.linalg.norm(state_f.omega_body))
    success = settle_time is not None and bz_f >= takeoff_cos * 0.99 and not timed_out

    return TrialResult(
        trial=trial,
        seed=seed,
        initial_tilt_deg=init_tilt,
        initial_bz=bz0,
        initial_roll_deg=roll0,
        initial_pitch_deg=pitch0,
        initial_yaw_deg=yaw0,
        success=success,
        recovery_time_s=float(recovery_time) if recovery_time is not None else float("nan"),
        settle_time_s=float(settle_time) if settle_time is not None else float("nan"),
        final_tilt_deg=tilt_f,
        final_bz=bz_f,
        final_omega_norm=omega_f,
        max_omega_norm=max_omega,
        max_att_error=float(np.max(att_errs)) if att_errs else float("nan"),
        mean_att_error=float(np.mean(att_errs)) if att_errs else float("nan"),
        peak_motor_abs=peak_motor,
        motor_sat_pct=100.0 * sat_steps / max(steps, 1),
        contact_lost=contact_lost,
        final_mode=controller.mode_manager.mode.name,
        timeout=timed_out,
    )


def _print_trial(r: TrialResult) -> None:
    status = "PASS" if r.success else "FAIL"
    rec = f"{r.recovery_time_s:6.2f}s" if math.isfinite(r.recovery_time_s) else "   n/a"
    stl = f"{r.settle_time_s:6.2f}s" if math.isfinite(r.settle_time_s) else "   n/a"
    print(
        f"  [{r.trial:03d}] {status}  "
        f"tilt0={r.initial_tilt_deg:6.1f}°  bz0={r.initial_bz:+.3f}  "
        f"t_rec={rec}  t_settle={stl}  "
        f"tilt_f={r.final_tilt_deg:5.1f}°  bz_f={r.final_bz:+.3f}  "
        f"|w|_max={r.max_omega_norm:5.2f}  sat={r.motor_sat_pct:4.1f}%"
        + ("  CONTACT_LOST" if r.contact_lost else "")
        + ("  TIMEOUT" if r.timeout else "")
    )


def _summary(results: list[TrialResult]) -> None:
    n = len(results)
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    rec_ok = np.array([r.recovery_time_s for r in ok if math.isfinite(r.recovery_time_s)])
    set_ok = np.array([r.settle_time_s for r in ok if math.isfinite(r.settle_time_s)])
    tilt0 = np.array([r.initial_tilt_deg for r in results])
    bz_f = np.array([r.final_bz for r in results])

    # Bin by initial tilt severity
    bins = [(0, 60), (60, 90), (90, 120), (120, 180)]
    print("\n" + "=" * 72)
    print("UPRIGHT RECOVERY VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  Trials            : {n}")
    print(f"  Success rate      : {100.0 * len(ok) / max(n, 1):.1f}%  ({len(ok)}/{n})")
    print(f"  Failure rate      : {100.0 * len(fail) / max(n, 1):.1f}%  ({len(fail)}/{n})")
    if rec_ok.size:
        print(
            f"  Recovery time     : mean={rec_ok.mean():.2f}s  "
            f"median={np.median(rec_ok):.2f}s  "
            f"std={rec_ok.std(ddof=1) if rec_ok.size > 1 else 0:.2f}s  "
            f"min={rec_ok.min():.2f}s  max={rec_ok.max():.2f}s"
        )
    if set_ok.size:
        print(
            f"  Settle time       : mean={set_ok.mean():.2f}s  "
            f"median={np.median(set_ok):.2f}s  "
            f"std={set_ok.std(ddof=1) if set_ok.size > 1 else 0:.2f}s  "
            f"min={set_ok.min():.2f}s  max={set_ok.max():.2f}s"
        )
    print(
        f"  Final body-z up   : mean={bz_f.mean():+.3f}  "
        f"min={bz_f.min():+.3f}  max={bz_f.max():+.3f}"
    )
    print(
        f"  Max |w| (all)     : mean={np.mean([r.max_omega_norm for r in results]):.2f}  "
        f"max={max(r.max_omega_norm for r in results):.2f} rad/s"
    )
    print(
        f"  Peak |motor|      : mean={np.mean([r.peak_motor_abs for r in results]):.2f}  "
        f"max={max(r.peak_motor_abs for r in results):.2f} N"
    )
    print(
        f"  Motor sat %       : mean={np.mean([r.motor_sat_pct for r in results]):.1f}%  "
        f"max={max(r.motor_sat_pct for r in results):.1f}%"
    )
    print(f"  Contact lost      : {sum(1 for r in results if r.contact_lost)}/{n}")
    print(f"  Timeouts          : {sum(1 for r in results if r.timeout)}/{n}")

    print("\n  Success by initial tilt bin:")
    for lo, hi in bins:
        subset = [r for r in results if lo <= r.initial_tilt_deg < hi]
        if not subset:
            continue
        s = sum(1 for r in subset if r.success)
        print(f"    [{lo:3d},{hi:3d})° : {100.0 * s / len(subset):5.1f}%  ({s}/{len(subset)})")

    if fail:
        print("\n  Failures:")
        for r in fail[:15]:
            reason = []
            if r.timeout:
                reason.append("TIMEOUT")
            if r.contact_lost:
                reason.append("CONTACT_LOST")
            if r.final_bz < 0.98:
                reason.append(f"bz={r.final_bz:+.3f}")
            print(
                f"    trial {r.trial:03d}  tilt0={r.initial_tilt_deg:6.1f}°  "
                f"mode={r.final_mode}  {' '.join(reason) or 'INCOMPLETE'}"
            )
        if len(fail) > 15:
            print(f"    ... and {len(fail) - 15} more")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate upright recovery from arbitrary orientations")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--trials", type=int, default=40, help="Number of random orientations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=25.0, help="Per-trial timeout [s]")
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "carolline_control" / "logs" / "recovery_validation.csv"),
        help="Output CSV path",
    )
    add_sensor_only_argument(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_config(args.config)
    # Recovery-only: do not start in ROLLING
    config.initial_mode = ControlMode.PRETAKEOFF

    print("CAROLLINE upright recovery validation")
    print_estimator_mode(sensor_only=args.sensor_only)
    print(f"  model      : {config.model_path}")
    print(f"  trials     : {args.trials}")
    print(f"  timeout    : {args.timeout}s / trial")
    print(f"  upright cos: {config.upright_cos_threshold}  (~{np.degrees(np.arccos(config.upright_cos_threshold)):.1f}° tilt)")
    print(f"  takeoff cos: {config.takeoff_cos_threshold}  (~{np.degrees(np.arccos(config.takeoff_cos_threshold)):.1f}° tilt)")
    print(f"  settle     : {config.upright_settle_time}s  omega_tol={config.upright_omega_tolerance} rad/s")
    print(f"  gains      : kR_pre={config.kR_pre}  kOmega_pre={config.kOmega_pre}  thrust_frac={config.pre_takeoff_thrust_fraction}")
    print("-" * 72)

    results: list[TrialResult] = []
    for i in range(args.trials):
        # Fresh config each trial (mass filled from MuJoCo inside run_trial)
        cfg = load_config(args.config)
        cfg.initial_mode = ControlMode.PRETAKEOFF
        r = run_trial(
            trial=i + 1,
            seed=args.seed + i,
            config=cfg,
            raw=raw,
            timeout_s=args.timeout,
            sensor_only=args.sensor_only,
        )
        results.append(r)
        _print_trial(r)

    _summary(results)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
