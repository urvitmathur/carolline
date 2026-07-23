"""Per-timestep physics verification and controller health monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from carolline_control.utils.so3 import quat_to_rot
from carolline_control.validation.config import HealthConfig, PhysicsTolerances


@dataclass
class PhysicsSnapshot:
    quat_norm_error: float
    orthogonality_error: float
    force_residual_norm: float
    torque_residual_norm: float
    thrust_sum_error: float
    flags: list[str] = field(default_factory=list)


@dataclass
class StepRecord:
    time: float
    mode: str
    pos_error_norm: float
    tilt_deg: float
    att_error_norm: float
    motor_spread: float
    thrust: float
    moment_norm: float
    motors: np.ndarray
    wind_norm: float
    contact_count: int
    contact_force_max: float
    physics: PhysicsSnapshot


class SimulationObserver:
    """Collects timestep data, physics residuals, and health metrics."""

    def __init__(
        self,
        physics_tol: PhysicsTolerances,
        health_cfg: HealthConfig,
        motor_min: float,
        motor_max: float,
        mass: float,
        inertia: np.ndarray,
    ) -> None:
        self._physics_tol = physics_tol
        self._health_cfg = health_cfg
        self._motor_min = motor_min
        self._motor_max = motor_max
        self._mass = mass
        self._inertia = inertia
        self.records: list[StepRecord] = []
        self._roll_hist: list[float] = []
        self._pitch_hist: list[float] = []
        self._motor_hist: list[float] = []
        self._time_hist: list[float] = []

    def observe_step(
        self,
        *,
        time: float,
        mode: str,
        state,
        target,
        diagnostics,
        motor,
        cmd,
        data,
        model,
        body_id: int,
        wind: np.ndarray,
        qacc: np.ndarray | None = None,
    ) -> PhysicsSnapshot:
        motors = np.asarray(motor.thrusts, dtype=float)
        pos_err = state.position - target.position
        e_r = diagnostics.e_R
        R = state.rotation
        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
        quat_err = abs(float(np.linalg.norm(state.quaternion)) - 1.0)

        thrust_sum = float(np.sum(motors))
        thrust_err = abs(thrust_sum - float(cmd.thrust))
        flags: list[str] = []

        if quat_err > self._physics_tol.quat_norm_tol:
            flags.append("QUAT_NORM")
        if ortho_err > self._physics_tol.orthogonality_tol:
            flags.append("ORTHOGONALITY")
        if thrust_err > self._physics_tol.thrust_sum_tol:
            flags.append("THRUST_SUM")

        force_res = 0.0
        torque_res = 0.0
        if qacc is not None:
            lin_acc = qacc[:3] if len(qacc) >= 3 else np.zeros(3)
            net_force_est = self._mass * lin_acc
            thrust_world = R @ np.array([0.0, 0.0, thrust_sum])
            weight_world = np.array([0.0, 0.0, -self._mass * 9.81])
            force_res = float(np.linalg.norm(net_force_est - thrust_world - weight_world))
            if force_res > self._physics_tol.force_residual_tol:
                flags.append("FORCE_RESIDUAL")

        contact_count = int(data.ncon)
        contact_force_max = 0.0
        if data.ncon > 0 and hasattr(data, "cfrc_ext") and data.cfrc_ext.size:
            cf = np.abs(data.cfrc_ext[body_id])
            contact_force_max = float(np.max(cf)) if cf.size else 0.0

        snap = PhysicsSnapshot(
            quat_norm_error=quat_err,
            orthogonality_error=ortho_err,
            force_residual_norm=force_res,
            torque_residual_norm=torque_res,
            thrust_sum_error=thrust_err,
            flags=flags,
        )
        self.records.append(
            StepRecord(
                time=time,
                mode=mode,
                pos_error_norm=float(np.linalg.norm(pos_err)),
                tilt_deg=diagnostics.tilt_deg,
                att_error_norm=float(np.linalg.norm(e_r)),
                motor_spread=diagnostics.motor_spread,
                thrust=float(cmd.thrust),
                moment_norm=float(np.linalg.norm(diagnostics.moment_body)),
                motors=motors.copy(),
                wind_norm=float(np.linalg.norm(wind)),
                contact_count=contact_count,
                contact_force_max=contact_force_max,
                physics=snap,
            )
        )
        self._roll_hist.append(float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))))
        self._pitch_hist.append(float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))))
        self._motor_hist.append(float(np.mean(np.abs(motors))))
        self._time_hist.append(time)
        return snap

    def dominant_frequency(self, signal: list[float]) -> float:
        n = len(signal)
        if n < self._health_cfg.min_samples_fft:
            return 0.0
        y = np.asarray(signal[-512:], dtype=float)
        y = y - y.mean()
        spec = np.abs(np.fft.rfft(y))
        if spec.size <= 1:
            return 0.0
        spec[0] = 0.0
        idx = int(np.argmax(spec))
        dt = np.median(np.diff(self._time_hist[-512:])) if len(self._time_hist) > 1 else 0.004
        freqs = np.fft.rfftfreq(len(y), dt)
        return float(freqs[idx]) if idx < len(freqs) else 0.0

    def oscillation_metric(self) -> float:
        if len(self._roll_hist) < self._health_cfg.min_samples_fft:
            return 0.0
        y = np.asarray(self._roll_hist[-512:], dtype=float)
        y = y - y.mean()
        spec = np.abs(np.fft.rfft(y))
        if spec.size <= 1 or spec.max() < 1e-12:
            return 0.0
        spec[0] = 0.0
        return float(spec.max() / (np.sum(spec) + 1e-12))

    def control_smoothness(self) -> float:
        if len(self.records) < 3:
            return 0.0
        m = np.array([r.moment_norm for r in self.records])
        return float(np.mean(np.abs(np.diff(m))))

    def motor_smoothness(self) -> float:
        if len(self.records) < 3:
            return 0.0
        spreads = np.array([r.motor_spread for r in self.records])
        return float(np.mean(np.abs(np.diff(spreads))))
