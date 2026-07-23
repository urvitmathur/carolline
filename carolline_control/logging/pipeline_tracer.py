"""Event-driven debug tracer for the CAROLLINE control pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from carolline_control.utils.so3 import attitude_error
from carolline_control.utils.types import ControlCommand, ControlMode, MotorCommand, RobotState, TrajectoryTarget


class PipelineAbort(Exception):
    """Raised when the tracer detects the first invalid control value."""


@dataclass
class PipelineAbortReport:
    time: float
    variable: str
    value: str
    origin: str
    likely_cause: str


def rotation_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw [rad] from a body-to-world rotation matrix."""
    R = np.asarray(R, dtype=float)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _fmt_vec(v: np.ndarray, precision: int = 4) -> str:
    return "(" + ", ".join(f"{float(x):.{precision}f}" for x in v) + ")"


def _relative_change(previous: float, current: float, eps: float = 1e-6) -> float:
    return abs(current - previous) / max(abs(previous), eps)


def _active_controller(mode: ControlMode) -> str:
    if mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
        return "PreTakeoffController"
    if mode == ControlMode.ROLLING:
        return "RollingController"
    return "FlightController"


class PipelineTracer:
    """Print pipeline diagnostics only on mode change, saturation, or >10% value change."""

    TRACKED_SCALARS = ("thrust", "roll", "pitch", "yaw", "des_roll", "des_pitch", "des_yaw")
    TRACKED_VECTORS = ("e_R", "omega_body", "omega_d", "moment")

    def __init__(
        self,
        motor_min: float,
        motor_max: float,
        change_threshold: float = 0.10,
    ) -> None:
        self.motor_min = motor_min
        self.motor_max = motor_max
        self.change_threshold = change_threshold
        self._prev_mode: Optional[ControlMode] = None
        self._prev_values: dict[str, float | np.ndarray] = {}
        self._prev_motors: Optional[np.ndarray] = None
        self._first_invalid: Optional[PipelineAbortReport] = None
        self.aborted = False

    def observe(
        self,
        state: RobotState,
        prev_mode: ControlMode,
        mode: ControlMode,
        target: TrajectoryTarget,
        cmd: ControlCommand,
        motor: MotorCommand,
        omega_d: np.ndarray,
        moment: np.ndarray,
    ) -> bool:
        """Record one control cycle. Returns False if simulation should stop."""
        if self.aborted:
            return False

        warnings = self._validate(state, cmd, motor, omega_d, moment)
        if warnings and self._first_invalid is None:
            warning = warnings[0]
            self._first_invalid = PipelineAbortReport(
                time=state.time,
                variable=warning["variable"],
                value=warning["value"],
                origin=warning["origin"],
                likely_cause=warning["cause"],
            )
            self.aborted = True
            self._print_abort(self._first_invalid)
            return False

        roll, pitch, yaw = rotation_to_rpy(state.rotation)
        des_roll, des_pitch, des_yaw = rotation_to_rpy(cmd.desired_rotation)
        e_R = attitude_error(state.rotation, cmd.desired_rotation)

        current = {
            "thrust": float(cmd.thrust),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "des_roll": des_roll,
            "des_pitch": des_pitch,
            "des_yaw": des_yaw,
            "e_R": e_R.copy(),
            "omega_body": np.asarray(state.omega_body, dtype=float).copy(),
            "omega_d": np.asarray(omega_d, dtype=float).copy(),
            "moment": np.asarray(moment, dtype=float).copy(),
        }

        mode_changed = self._prev_mode is not None and mode != self._prev_mode
        motors = np.asarray(motor.thrusts, dtype=float)
        saturated = self._motors_saturated(motors)
        value_changed = self._values_changed(current, motors)
        first_sample = self._prev_mode is None

        if mode_changed or saturated or value_changed or first_sample:
            display_prev = "INIT" if first_sample else prev_mode.name
            self._print_event(
                time=state.time,
                prev_mode_name=display_prev,
                mode=mode,
                state=state,
                target=target,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                des_roll=des_roll,
                des_pitch=des_pitch,
                des_yaw=des_yaw,
                e_R=e_R,
                omega_d=omega_d,
                moment=moment,
                cmd=cmd,
                motors=motors,
                saturated=saturated,
                warnings=warnings,
            )

        self._prev_mode = mode
        self._prev_values = current
        self._prev_motors = motors.copy()
        return True

    def _validate(
        self,
        state: RobotState,
        cmd: ControlCommand,
        motor: MotorCommand,
        omega_d: np.ndarray,
        moment: np.ndarray,
    ) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []

        q = np.asarray(state.quaternion, dtype=float)
        q_norm = float(np.linalg.norm(q))
        if not np.all(np.isfinite(q)) or abs(q_norm - 1.0) > 1e-2:
            warnings.append(
                {
                    "variable": "quaternion",
                    "value": _fmt_vec(q, 6),
                    "origin": "StateEstimator.estimate",
                    "cause": "MuJoCo quaternion is non-finite or not unit length.",
                }
            )

        R = np.asarray(state.rotation, dtype=float)
        Rd = np.asarray(cmd.desired_rotation, dtype=float)
        for name, matrix, origin in (
            ("rotation", R, "StateEstimator.estimate / quat_to_rot"),
            ("desired_rotation", Rd, f"{_active_controller(cmd.mode)}.compute"),
        ):
            if not np.all(np.isfinite(matrix)):
                warnings.append(
                    {
                        "variable": name,
                        "value": "non-finite matrix",
                        "origin": origin,
                        "cause": "Rotation matrix contains NaN or Inf.",
                    }
                )
                continue
            det = float(np.linalg.det(matrix))
            ortho_err = float(np.linalg.norm(matrix.T @ matrix - np.eye(3)))
            if abs(det - 1.0) > 1e-2 or ortho_err > 1e-2:
                warnings.append(
                    {
                        "variable": name,
                        "value": f"det={det:.6f}, ortho_err={ortho_err:.6f}",
                        "origin": origin,
                        "cause": "Rotation matrix is not a valid SO(3) element.",
                    }
                )

        thrust = float(cmd.thrust)
        max_collective = 4.0 * self.motor_max * 1.05
        min_collective = 4.0 * self.motor_min * 1.05 if self.motor_min < 0.0 else 0.0
        if (
            not math.isfinite(thrust)
            or thrust > max_collective
            or thrust < min_collective
        ):
            warnings.append(
                {
                    "variable": "collective_thrust",
                    "value": f"{thrust:.6f}",
                    "origin": f"{_active_controller(cmd.mode)}.compute",
                    "cause": "Thrust is non-finite or outside actuator-achievable bounds.",
                }
            )

        moment_vec = np.asarray(moment, dtype=float)
        if not np.all(np.isfinite(moment_vec)):
            warnings.append(
                {
                    "variable": "body_moment",
                    "value": _fmt_vec(moment_vec, 6),
                    "origin": "TorqueController.compute",
                    "cause": "Body moment command is non-finite.",
                }
            )
        elif float(np.linalg.norm(moment_vec)) > 1e4:
            warnings.append(
                {
                    "variable": "body_moment",
                    "value": _fmt_vec(moment_vec, 4),
                    "origin": "TorqueController.compute",
                    "cause": "Body moment magnitude is unrealistically large.",
                }
            )

        motors = np.asarray(motor.thrusts, dtype=float)
        if not np.all(np.isfinite(motors)):
            warnings.append(
                {
                    "variable": "motor_thrusts",
                    "value": _fmt_vec(motors, 6),
                    "origin": "MotorMixer.mix",
                    "cause": "Motor allocation produced non-finite thrusts.",
                }
            )

        omega_d_vec = np.asarray(omega_d, dtype=float)
        if not np.all(np.isfinite(omega_d_vec)):
            warnings.append(
                {
                    "variable": "desired_omega_body",
                    "value": _fmt_vec(omega_d_vec, 6),
                    "origin": "AttitudeController.compute_desired_omega",
                    "cause": "Desired body angular velocity is non-finite.",
                }
            )

        return warnings

    def _motors_saturated(self, motors: np.ndarray) -> bool:
        return bool(
            np.any(motors >= self.motor_max - 1e-3)
            or np.any(motors <= self.motor_min + 1e-3)
        )

    def _values_changed(self, current: dict, motors: np.ndarray) -> bool:
        if not self._prev_values:
            return False

        for key in self.TRACKED_SCALARS:
            if _relative_change(float(self._prev_values[key]), float(current[key])) > self.change_threshold:
                return True

        for key in self.TRACKED_VECTORS:
            prev = np.asarray(self._prev_values[key], dtype=float)
            curr = np.asarray(current[key], dtype=float)
            for i in range(len(curr)):
                if _relative_change(float(prev[i]), float(curr[i])) > self.change_threshold:
                    return True
            if _relative_change(float(np.linalg.norm(prev)), float(np.linalg.norm(curr))) > self.change_threshold:
                return True

        if self._prev_motors is not None:
            for prev, curr in zip(self._prev_motors, motors):
                delta = abs(float(curr) - float(prev))
                scale = max(abs(float(prev)), 0.1 * self.motor_max, 1e-6)
                if delta / scale > self.change_threshold:
                    return True

        return False

    def _print_event(
        self,
        *,
        time: float,
        prev_mode_name: str,
        mode: ControlMode,
        state: RobotState,
        target: TrajectoryTarget,
        roll: float,
        pitch: float,
        yaw: float,
        des_roll: float,
        des_pitch: float,
        des_yaw: float,
        e_R: np.ndarray,
        omega_d: np.ndarray,
        moment: np.ndarray,
        cmd: ControlCommand,
        motors: np.ndarray,
        saturated: bool,
        warnings: list[dict[str, str]],
    ) -> None:
        active = _active_controller(mode)
        sat_note = " | MOTOR SATURATION" if saturated else ""
        print(f"\n--- PIPELINE EVENT t={time:.4f}s{sat_note} ---")
        print(
            "Pipeline: StateEstimator -> ModeManager -> Planner -> "
            f"{active} -> AttitudeController -> TorqueController -> MotorMixer"
        )
        print(f"Mode: {prev_mode_name} -> {mode.name}")
        print(f"Position (x,y,z): {_fmt_vec(state.position)}")
        print(
            "Roll, Pitch, Yaw [deg]: "
            f"({math.degrees(roll):.2f}, {math.degrees(pitch):.2f}, {math.degrees(yaw):.2f})"
        )
        print(
            "Desired Roll, Pitch, Yaw [deg]: "
            f"({math.degrees(des_roll):.2f}, {math.degrees(des_pitch):.2f}, {math.degrees(des_yaw):.2f})"
        )
        print(f"Attitude error e_R: {_fmt_vec(e_R)} | |e_R|={float(np.linalg.norm(e_R)):.4f}")
        print(f"Body angular velocity: {_fmt_vec(state.omega_body)}")
        print(f"Desired body angular velocity: {_fmt_vec(omega_d)}")
        print(f"Collective thrust [N]: {cmd.thrust:.4f}")
        print(f"Body moment (Mx, My, Mz): {_fmt_vec(moment)}")
        print(f"Motor thrusts [N]: {_fmt_vec(motors)}")
        print(
            "Planner target (x,y,z): "
            f"{_fmt_vec(target.position)} | yaw={math.degrees(target.yaw):.2f} deg"
        )
        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning['variable']}: {warning['value']} ({warning['origin']})")

    def _print_abort(self, report: PipelineAbortReport) -> None:
        print("\n=== PIPELINE ABORT: FIRST INVALID VALUE ===")
        print(f"Time: {report.time:.4f}s")
        print(f"1. Variable that first became incorrect: {report.variable}")
        print(f"2. Function where it originated: {report.origin}")
        print(f"3. Likely cause: {report.likely_cause}")
        print(f"   Observed value: {report.value}")
