"""
Monte Carlo full-mission validation for CAROLLINE.

Evaluates robustness of the COMPLETE mission over randomized initial
orientations. Does NOT modify controller logic, gains, or the state machine.

Mission sequence per trial:
  Recovery -> Upright Stabilization -> Takeoff -> Hover -> Waypoints -> Landing

Usage (from repo root):
    python carolline_control/scripts/validate_monte_carlo.py
    python carolline_control/scripts/validate_monte_carlo.py --trials 30 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
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
from carolline_control.logging.pipeline_tracer import PipelineAbort
from carolline_control.utils.so3 import body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode
from carolline_control.visualization.markers import compile_model_with_markers


# Failure reason tags (observer classification only)
FAIL_NONE = ""
FAIL_RECOVERY_TIMEOUT = "Recovery timeout"
FAIL_STABILIZATION = "Stabilization failure"
FAIL_TAKEOFF = "Takeoff failure"
FAIL_HOVER = "Hover instability"
FAIL_WAYPOINT = "Waypoint timeout"
FAIL_LANDING = "Landing failure"
FAIL_CONTACT = "Contact loss"
FAIL_NUMERICAL = "Numerical instability"
FAIL_UNKNOWN = "Unknown"


@dataclass
class MissionTrial:
    trial: int
    seed: int
    initial_roll_deg: float
    initial_pitch_deg: float
    initial_yaw_deg: float
    initial_tilt_deg: float
    recovery_success: bool
    recovery_time_s: float
    stabilization_success: bool
    takeoff_success: bool
    hover_success: bool
    waypoint_success: bool
    landing_success: bool
    mission_completed: bool
    mission_result: str
    total_mission_duration_s: float
    max_tilt_deg: float
    max_position_error_m: float
    max_waypoint_tracking_error_m: float
    mean_hover_error_m: float
    mean_waypoint_error_m: float
    landing_position_error_m: float
    max_motor_saturation_pct: float
    failure_reason: str
    final_mode: str


def _random_ground_qpos(
    spawn_xy: list[float],
    ground_z: float,
    rng: random.Random,
    tilt_range: tuple[float, float],
) -> tuple[list[float], float]:
    """Same style as recovery validation: random tilt about a horizontal axis."""
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


def _euler_from_qpos(qpos: list[float]) -> tuple[float, float, float]:
    w, x, y, z = qpos[3:7]
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


def _classify_failure(
    *,
    recovery_success: bool,
    stabilization_success: bool,
    takeoff_success: bool,
    hover_success: bool,
    waypoint_success: bool,
    landing_success: bool,
    contact_loss: bool,
    numerical: bool,
    final_mode: str,
) -> str:
    if landing_success:
        return FAIL_NONE
    if numerical:
        return FAIL_NUMERICAL
    if contact_loss and not recovery_success:
        return FAIL_CONTACT
    if not recovery_success:
        return FAIL_RECOVERY_TIMEOUT
    if not stabilization_success:
        return FAIL_STABILIZATION
    if not takeoff_success:
        return FAIL_TAKEOFF
    if not hover_success:
        return FAIL_HOVER
    if not waypoint_success:
        return FAIL_WAYPOINT
    if not landing_success:
        return FAIL_LANDING
    return FAIL_UNKNOWN


def run_trial(
    *,
    trial: int,
    seed: int,
    config,
    raw: dict,
    timeout_s: float,
    tilt_range: tuple[float, float],
) -> MissionTrial:
    rng = random.Random(seed)
    np.random.seed(seed)

    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    # Full airborne mission starts with recovery (same as recovery validator).
    config.initial_mode = ControlMode.PRETAKEOFF
    controller = CarollineController(config)
    controller.mode_manager.mode = ControlMode.PRETAKEOFF

    spawn = np.asarray(raw.get("mission", {}).get("spawn_xy", [0.0, 0.0]), dtype=float)
    ground_z = float(raw.get("ground_z", 0.40))
    qpos, init_tilt = _random_ground_qpos(spawn.tolist(), ground_z, rng, tilt_range)
    roll0, pitch0, yaw0 = _euler_from_qpos(qpos)

    data.qpos[:7] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)

    # Phase timestamps
    t_upright: float | None = None
    t_takeoff: float | None = None
    t_hover: float | None = None
    t_flight: float | None = None
    t_landing: float | None = None
    t_idle: float | None = None

    # Accumulators
    max_tilt = 0.0
    max_pos_err = 0.0
    max_wp_err = 0.0
    hover_errs: list[float] = []
    wp_errs: list[float] = []
    sat_steps = 0
    steps = 0
    contact_loss = False
    numerical = False
    pipeline_abort = False
    landing_err = float("nan")
    last_mode = ControlMode.PRETAKEOFF
    idle_since: float | None = None

    while data.time < timeout_s:
        state = estimator.estimate(data)
        try:
            motor, cmd, mode, target = controller.compute(state, dt)
        except PipelineAbort:
            pipeline_abort = True
            numerical = True
            break
        except Exception:
            numerical = True
            break

        thrusts = np.asarray(motor.thrusts, dtype=float)
        data.ctrl[:] = thrusts

        # Mode transition markers
        if mode == ControlMode.UPRIGHT and last_mode != ControlMode.UPRIGHT:
            t_upright = float(state.time)
        if mode == ControlMode.TAKEOFF and last_mode != ControlMode.TAKEOFF:
            t_takeoff = float(state.time)
        if mode == ControlMode.HOVER and last_mode != ControlMode.HOVER:
            t_hover = float(state.time)
        if mode == ControlMode.FLIGHT and last_mode != ControlMode.FLIGHT:
            t_flight = float(state.time)
        if mode == ControlMode.LANDING and last_mode != ControlMode.LANDING:
            t_landing = float(state.time)
        if mode == ControlMode.IDLE and last_mode != ControlMode.IDLE:
            t_idle = float(state.time)
            landing_err = float(np.linalg.norm(state.position[:2] - spawn))

        # Metrics
        bz = float(body_z_world(state.rotation)[2])
        tilt = float(np.degrees(np.arccos(np.clip(bz, -1.0, 1.0))))
        max_tilt = max(max_tilt, tilt)

        pos_err = float(np.linalg.norm(state.position - target.position))
        max_pos_err = max(max_pos_err, pos_err)

        if mode == ControlMode.HOVER:
            hover_errs.append(pos_err)
        if mode == ControlMode.FLIGHT:
            wp_err = float(np.linalg.norm(state.position - target.position))
            wp_errs.append(wp_err)
            max_wp_err = max(max_wp_err, wp_err)

        if np.any(thrusts >= config.motor_max - 0.05) or np.any(thrusts <= config.motor_min + 0.05):
            sat_steps += 1
        steps += 1

        # Contact loss after airborne (takeoff onward)
        if t_takeoff is not None and mode in (
            ControlMode.TAKEOFF,
            ControlMode.HOVER,
            ControlMode.FLIGHT,
        ):
            if state.position[2] < ground_z + 0.05 and float(np.linalg.norm(state.velocity)) > 1.0:
                contact_loss = True

        # Numerical instability heuristics
        if (
            not np.isfinite(state.position).all()
            or not np.isfinite(thrusts).all()
            or abs(state.position[2]) > 50.0
            or float(np.linalg.norm(state.omega_body)) > 80.0
        ):
            numerical = True
            break

        last_mode = mode
        mujoco.mj_step(model, data)

        # End after IDLE settled briefly (same idea as main.py)
        if mode == ControlMode.IDLE:
            if idle_since is None:
                idle_since = float(state.time)
            elif float(state.time) - idle_since > 1.0:
                break
        else:
            idle_since = None

    final_mode = controller.mode_manager.mode.name
    duration = float(data.time)

    recovery_success = t_upright is not None
    stabilization_success = t_takeoff is not None
    takeoff_success = t_hover is not None
    hover_success = t_flight is not None
    waypoint_success = t_landing is not None
    landing_success = t_idle is not None
    mission_completed = landing_success and not numerical and not pipeline_abort

    failure_reason = _classify_failure(
        recovery_success=recovery_success,
        stabilization_success=stabilization_success,
        takeoff_success=takeoff_success,
        hover_success=hover_success,
        waypoint_success=waypoint_success,
        landing_success=landing_success,
        contact_loss=contact_loss,
        numerical=numerical or pipeline_abort,
        final_mode=final_mode,
    )
    if mission_completed:
        failure_reason = FAIL_NONE

    # If timed out without IDLE, refine reason by last reached phase
    if not mission_completed and failure_reason == FAIL_NONE:
        failure_reason = FAIL_UNKNOWN

    return MissionTrial(
        trial=trial,
        seed=seed,
        initial_roll_deg=roll0,
        initial_pitch_deg=pitch0,
        initial_yaw_deg=yaw0,
        initial_tilt_deg=init_tilt,
        recovery_success=recovery_success,
        recovery_time_s=float(t_upright) if t_upright is not None else float("nan"),
        stabilization_success=stabilization_success,
        takeoff_success=takeoff_success,
        hover_success=hover_success,
        waypoint_success=waypoint_success,
        landing_success=landing_success,
        mission_completed=mission_completed,
        mission_result="PASS" if mission_completed else "FAIL",
        total_mission_duration_s=duration,
        max_tilt_deg=max_tilt,
        max_position_error_m=max_pos_err,
        max_waypoint_tracking_error_m=max_wp_err if wp_errs else float("nan"),
        mean_hover_error_m=float(np.mean(hover_errs)) if hover_errs else float("nan"),
        mean_waypoint_error_m=float(np.mean(wp_errs)) if wp_errs else float("nan"),
        landing_position_error_m=landing_err if math.isfinite(landing_err) else float("nan"),
        max_motor_saturation_pct=100.0 * sat_steps / max(steps, 1),
        failure_reason=failure_reason if not mission_completed else FAIL_NONE,
        final_mode=final_mode,
    )


def _print_trial(r: MissionTrial) -> None:
    flags = (
        f"R={'Y' if r.recovery_success else 'N'} "
        f"S={'Y' if r.stabilization_success else 'N'} "
        f"T={'Y' if r.takeoff_success else 'N'} "
        f"H={'Y' if r.hover_success else 'N'} "
        f"W={'Y' if r.waypoint_success else 'N'} "
        f"L={'Y' if r.landing_success else 'N'}"
    )
    rec = f"{r.recovery_time_s:5.2f}s" if math.isfinite(r.recovery_time_s) else "  n/a"
    land = f"{r.landing_position_error_m:5.3f}m" if math.isfinite(r.landing_position_error_m) else "  n/a"
    fail = f"  fail={r.failure_reason}" if r.failure_reason else ""
    print(
        f"  [{r.trial:03d}] {r.mission_result}  "
        f"rpy=({r.initial_roll_deg:+6.1f},{r.initial_pitch_deg:+6.1f},{r.initial_yaw_deg:+6.1f})  "
        f"tilt0={r.initial_tilt_deg:5.1f}  "
        f"{flags}  "
        f"t_rec={rec}  dur={r.total_mission_duration_s:6.1f}s  "
        f"tilt_max={r.max_tilt_deg:5.1f}  land_err={land}  "
        f"sat={r.max_motor_saturation_pct:4.1f}%  mode={r.final_mode}"
        f"{fail}"
    )


def _agg(vals: list[float]) -> tuple[float, float, float]:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(a.mean()), float(np.median(a)), float(a.max())


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def _summary(results: list[MissionTrial]) -> None:
    n = len(results)
    ok = [r for r in results if r.mission_completed]
    fail = [r for r in results if not r.mission_completed]

    mean_dur, med_dur, max_dur = _agg([r.total_mission_duration_s for r in results])
    mean_rec, _, _ = _agg([r.recovery_time_s for r in results])
    mean_hov, _, _ = _agg([r.mean_hover_error_m for r in results])
    mean_wp, _, _ = _agg([r.mean_waypoint_error_m for r in results])
    mean_land, _, _ = _agg([r.landing_position_error_m for r in results])
    mean_tilt, _, _ = _agg([r.max_tilt_deg for r in results])
    mean_sat, _, max_sat = _agg([r.max_motor_saturation_pct for r in results])

    print("\n" + "=" * 78)
    print("MONTE CARLO VALIDATION SUMMARY")
    print("=" * 78)
    print(f"  Trials                 : {n}")
    print(f"  Successful Missions    : {len(ok)}")
    print(f"  Mission Success Rate   : {100.0 * len(ok) / max(n, 1):.1f}%")
    print(f"  Failure Rate           : {100.0 * len(fail) / max(n, 1):.1f}%")
    print("")
    print(f"  Mean Mission Time      : {_fmt(mean_dur, 2)} s")
    print(f"  Median Mission Time    : {_fmt(med_dur, 2)} s")
    print(f"  Maximum Mission Time   : {_fmt(max_dur, 2)} s")
    print("")
    print(f"  Mean Recovery Time     : {_fmt(mean_rec, 2)} s")
    print(f"  Mean Hover Error       : {_fmt(mean_hov, 4)} m")
    print(f"  Mean Waypoint Error    : {_fmt(mean_wp, 4)} m")
    print(f"  Mean Landing Error     : {_fmt(mean_land, 4)} m")
    print(f"  Mean Maximum Tilt      : {_fmt(mean_tilt, 2)} deg")
    print(f"  Mean Motor Saturation  : {_fmt(mean_sat, 2)} %")
    print(f"  Maximum Motor Saturation: {_fmt(max_sat, 2)} %")
    print("")
    print("  Failure Counts by Category")
    counts = Counter(r.failure_reason for r in fail)
    if not counts:
        print("    (none)")
    else:
        for reason, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            label = reason if reason else "Unknown"
            print(f"    {label:28s}: {count}")

    print("")
    print("  Mission Success by Initial Tilt Range")
    bins = [(0, 60), (60, 90), (90, 120), (120, 180)]
    for lo, hi in bins:
        subset = [r for r in results if lo <= r.initial_tilt_deg < hi]
        if not subset:
            print(f"    [{lo:3d},{hi:3d}) deg : n/a  (0 trials)")
            continue
        s = sum(1 for r in subset if r.mission_completed)
        print(f"    [{lo:3d},{hi:3d}) deg : {100.0 * s / len(subset):5.1f}%  ({s}/{len(subset)})")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo full-mission validation")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-trial timeout [s] (default: sim_duration from config)",
    )
    parser.add_argument("--tilt-min", type=float, default=20.0)
    parser.add_argument("--tilt-max", type=float, default=175.0)
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "carolline_control" / "logs" / "monte_carlo_validation.csv"),
    )
    args = parser.parse_args()

    raw = load_raw_config(args.config)
    cfg0 = load_config(args.config)
    timeout = float(args.timeout) if args.timeout is not None else float(raw.get("sim_duration", 300.0))

    print("CAROLLINE Monte Carlo full-mission validation")
    print(f"  model       : {cfg0.model_path}")
    print(f"  trials      : {args.trials}")
    print(f"  timeout     : {timeout}s / trial")
    print(f"  tilt range  : [{args.tilt_min}, {args.tilt_max}] deg")
    print(f"  sequence    : Recovery -> Upright -> Takeoff -> Hover -> Waypoints -> Landing")
    print("-" * 78)

    results: list[MissionTrial] = []
    for i in range(args.trials):
        cfg = load_config(args.config)
        cfg.initial_mode = ControlMode.PRETAKEOFF
        r = run_trial(
            trial=i + 1,
            seed=args.seed + i,
            config=cfg,
            raw=raw,
            timeout_s=timeout,
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
