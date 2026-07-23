"""Aggregate per-run metrics from observer records."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from carolline_control.validation.failures import FailureCause, classify_failure
from carolline_control.validation.observer import SimulationObserver
from carolline_control.validation.randomization import RunParameters


@dataclass
class RunMetrics:
    run_id: int
    seed: int
    campaign: str
    mission_success: bool
    mission_time: float
    mode_transitions: int
    max_position_error: float
    mean_position_error: float
    max_attitude_error: float
    max_tilt_deg: float
    max_angular_velocity: float
    avg_motor_utilization: float
    peak_motor_utilization: float
    motor_saturation_pct: float
    total_energy_j: float
    avg_power_w: float
    max_control_effort: float
    recovery_time_s: float
    waypoint_success_rate: float
    landing_accuracy_m: float
    contact_events: int
    max_contact_force: float
    oscillation_metric: float
    numerical_stability_metric: float
    max_quat_error: float
    max_orthogonality_error: float
    max_force_residual: float
    max_torque_residual: float
    control_smoothness: float
    motor_smoothness: float
    dominant_roll_freq_hz: float
    failure_cause: str
    final_mode: str
    mass_scale: float
    motor_effectiveness: float
    wind_magnitude: float
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_run_metrics(
    *,
    run_params: RunParameters,
    observer: SimulationObserver,
    final_mode: str,
    mission_success: bool,
    mission_time: float,
    mode_transitions: int,
    mode_history: list[str],
    motor_min: float,
    motor_max: float,
    mission_timeout: float,
    pipeline_aborted: bool,
    spawn_xy: np.ndarray,
    max_omega: float,
    landing_pos: np.ndarray | None,
) -> RunMetrics:
    recs = observer.records
    if not recs:
        return RunMetrics(
            run_id=run_params.run_id,
            seed=run_params.seed,
            campaign=run_params.campaign,
            mission_success=False,
            mission_time=0.0,
            mode_transitions=0,
            max_position_error=0.0,
            mean_position_error=0.0,
            max_attitude_error=0.0,
            max_tilt_deg=0.0,
            max_angular_velocity=0.0,
            avg_motor_utilization=0.0,
            peak_motor_utilization=0.0,
            motor_saturation_pct=0.0,
            total_energy_j=0.0,
            avg_power_w=0.0,
            max_control_effort=0.0,
            recovery_time_s=0.0,
            waypoint_success_rate=0.0,
            landing_accuracy_m=999.0,
            contact_events=0,
            max_contact_force=0.0,
            oscillation_metric=0.0,
            numerical_stability_metric=0.0,
            max_quat_error=0.0,
            max_orthogonality_error=0.0,
            max_force_residual=0.0,
            max_torque_residual=0.0,
            control_smoothness=0.0,
            motor_smoothness=0.0,
            dominant_roll_freq_hz=0.0,
            failure_cause=FailureCause.PIPELINE_ABORT.value,
            final_mode=final_mode,
            mass_scale=run_params.mass_scale,
            motor_effectiveness=run_params.motor_effectiveness,
            wind_magnitude=float(np.linalg.norm(run_params.wind_vector)),
            params=run_params.to_dict(),
        )

    pos_errs = np.array([r.pos_error_norm for r in recs])
    att_errs = np.array([r.att_error_norm for r in recs])
    tilts = np.array([r.tilt_deg for r in recs])
    moments = np.array([r.moment_norm for r in recs])
    motors_all = np.array([r.motors for r in recs])
    motor_abs = np.abs(motors_all)
    motor_range = float(max(abs(motor_min), abs(motor_max)))
    utilization = motor_abs / max(motor_range, 1e-6)
    sat_mask = (motor_abs >= motor_max - 0.05) | (motor_abs <= motor_min + 0.05)

    dt = np.median(np.diff([r.time for r in recs[: min(200, len(recs))]])) if len(recs) > 1 else 0.004
    power = np.array([np.sum(np.abs(r.motors)) for r in recs])
    energy = float(np.sum(power) * dt)

    # Waypoint proxy: fraction of FLIGHT with pos error < 0.25 m
    flight_recs = [r for r in recs if r.mode == "FLIGHT"]
    wp_rate = (
        float(np.mean([r.pos_error_norm < 0.25 for r in flight_recs])) if flight_recs else 0.0
    )

    # Recovery time: time in PRETAKEOFF+UPRIGHT
    recovery_time = sum(
        dt for r in recs if r.mode in ("PRETAKEOFF", "UPRIGHT")
    )

    # Stuck mode detection
    stuck = None
    if len(mode_history) > 100:
        last_modes = mode_history[-int(5.0 / dt) :]
        if last_modes and all(m == last_modes[0] for m in last_modes) and last_modes[0] != "IDLE":
            stuck = last_modes[0]

    phys_flags = sum(len(r.physics.flags) for r in recs)  # logged via numerical_stability_metric
    osc = observer.oscillation_metric()

    landing_acc = float(np.linalg.norm(landing_pos[:2] - spawn_xy)) if landing_pos is not None else 999.0

    failure = classify_failure(
        mission_success=mission_success,
        final_mode=final_mode,
        sim_time=mission_time,
        mission_timeout=mission_timeout,
        max_pos_error=float(pos_errs.max()),
        max_tilt=float(tilts.max()),
        motor_sat_pct=100.0 * float(np.mean(sat_mask)),
        oscillation_metric=osc,
        max_quat_error=max(r.physics.quat_norm_error for r in recs),
        max_orthogonality_error=max(r.physics.orthogonality_error for r in recs),
        mode_transitions=mode_transitions,
        stuck_in_mode=stuck,
        pipeline_aborted=pipeline_aborted,
        contact_events=sum(r.contact_count for r in recs),
        waypoint_success_rate=wp_rate,
    )

    return RunMetrics(
        run_id=run_params.run_id,
        seed=run_params.seed,
        campaign=run_params.campaign,
        mission_success=mission_success,
        mission_time=mission_time,
        mode_transitions=mode_transitions,
        max_position_error=float(pos_errs.max()),
        mean_position_error=float(pos_errs.mean()),
        max_attitude_error=float(att_errs.max()),
        max_tilt_deg=float(tilts.max()),
        max_angular_velocity=max_omega,
        avg_motor_utilization=float(utilization.mean()),
        peak_motor_utilization=float(utilization.max()),
        motor_saturation_pct=100.0 * float(np.mean(sat_mask)),
        total_energy_j=energy,
        avg_power_w=energy / max(mission_time, 1e-6),
        max_control_effort=float(moments.max()),
        recovery_time_s=float(recovery_time),
        waypoint_success_rate=wp_rate,
        landing_accuracy_m=landing_acc,
        contact_events=sum(r.contact_count for r in recs),
        max_contact_force=max(r.contact_force_max for r in recs),
        oscillation_metric=osc,
        numerical_stability_metric=float(phys_flags / max(len(recs), 1)),
        max_quat_error=max(r.physics.quat_norm_error for r in recs),
        max_orthogonality_error=max(r.physics.orthogonality_error for r in recs),
        max_force_residual=max(r.physics.force_residual_norm for r in recs),
        max_torque_residual=max(r.physics.torque_residual_norm for r in recs),
        control_smoothness=observer.control_smoothness(),
        motor_smoothness=observer.motor_smoothness(),
        dominant_roll_freq_hz=observer.dominant_frequency(observer._roll_hist),
        failure_cause=failure.value,
        final_mode=final_mode,
        mass_scale=run_params.mass_scale,
        motor_effectiveness=run_params.motor_effectiveness,
        wind_magnitude=float(np.linalg.norm(run_params.wind_vector)),
        params=run_params.to_dict(),
    )
