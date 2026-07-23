"""Automatic failure classification for validation runs."""

from __future__ import annotations

from enum import Enum

from carolline_control.utils.types import ControlMode


class FailureCause(str, Enum):
    NONE = "NONE"
    FAILED_UPRIGHT = "FAILED_UPRIGHT"
    FAILED_TAKEOFF = "FAILED_TAKEOFF"
    FAILED_HOVER = "FAILED_HOVER"
    FAILED_WAYPOINT = "FAILED_WAYPOINT"
    FAILED_RECOVERY = "FAILED_RECOVERY"
    FAILED_LANDING = "FAILED_LANDING"
    MOTOR_SATURATION = "MOTOR_SATURATION"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    STATE_MACHINE_STUCK = "STATE_MACHINE_STUCK"
    OSCILLATION = "OSCILLATION"
    MISSION_TIMEOUT = "MISSION_TIMEOUT"
    CONTACT_FAILURE = "CONTACT_FAILURE"
    PIPELINE_ABORT = "PIPELINE_ABORT"


def classify_failure(
    *,
    mission_success: bool,
    final_mode: str,
    sim_time: float,
    mission_timeout: float,
    max_pos_error: float,
    max_tilt: float,
    motor_sat_pct: float,
    oscillation_metric: float,
    max_quat_error: float,
    max_orthogonality_error: float,
    mode_transitions: int,
    stuck_in_mode: str | None,
    pipeline_aborted: bool,
    contact_events: int,
    waypoint_success_rate: float,
) -> FailureCause:
    if mission_success:
        return FailureCause.NONE
    if pipeline_aborted:
        return FailureCause.PIPELINE_ABORT
    if sim_time >= mission_timeout:
        return FailureCause.MISSION_TIMEOUT
    if max_quat_error > 0.05 or max_orthogonality_error > 0.1:
        return FailureCause.NUMERICAL_INSTABILITY
    if motor_sat_pct > 25.0:
        return FailureCause.MOTOR_SATURATION
    if oscillation_metric > 0.25:
        return FailureCause.OSCILLATION
    if stuck_in_mode:
        return FailureCause.STATE_MACHINE_STUCK
    if final_mode in ("PRETAKEOFF", "UPRIGHT") or stuck_in_mode in ("PRETAKEOFF", "UPRIGHT"):
        return FailureCause.FAILED_UPRIGHT
    if final_mode == "TAKEOFF" or stuck_in_mode == "TAKEOFF":
        return FailureCause.FAILED_TAKEOFF
    if final_mode == "HOVER" or stuck_in_mode == "HOVER":
        return FailureCause.FAILED_HOVER
    if final_mode == "FLIGHT" and waypoint_success_rate < 0.5:
        return FailureCause.FAILED_WAYPOINT
    if final_mode == "LANDING" or stuck_in_mode == "LANDING":
        return FailureCause.FAILED_LANDING
    if contact_events > 500 and max_tilt > 45:
        return FailureCause.CONTACT_FAILURE
    if max_pos_error > 2.0:
        return FailureCause.FAILED_RECOVERY
    return FailureCause.FAILED_WAYPOINT
