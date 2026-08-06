"""Metrics for geometric vs LQR controller comparison studies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackingSample:
    time: float
    pos: np.ndarray
    vel: np.ndarray
    pos_ref: np.ndarray
    vel_ref: np.ndarray
    motors: np.ndarray
    tilt_deg: float


@dataclass
class ComparisonMetrics:
    controller: str
    domain: str
    rmse_position: float
    rmse_velocity: float
    max_position_error: float
    max_velocity_error: float
    max_tilt_deg: float
    settling_time_s: float
    peak_motor_thrust: float
    motor_saturation_pct: float
    control_effort: float
    stable: bool
    notes: str = ""

    def to_row(self) -> dict[str, float | str | bool]:
        return {
            "controller": self.controller,
            "domain": self.domain,
            "rmse_position": self.rmse_position,
            "rmse_velocity": self.rmse_velocity,
            "max_position_error": self.max_position_error,
            "max_velocity_error": self.max_velocity_error,
            "max_tilt_deg": self.max_tilt_deg,
            "settling_time_s": self.settling_time_s,
            "peak_motor_thrust": self.peak_motor_thrust,
            "motor_saturation_pct": self.motor_saturation_pct,
            "control_effort": self.control_effort,
            "stable": self.stable,
            "notes": self.notes,
        }


@dataclass
class MetricsAccumulator:
    motor_min: float
    motor_max: float
    samples: list[TrackingSample] = field(default_factory=list)

    def add(
        self,
        *,
        time: float,
        pos: np.ndarray,
        vel: np.ndarray,
        pos_ref: np.ndarray,
        vel_ref: np.ndarray,
        motors: np.ndarray,
        tilt_deg: float,
    ) -> None:
        self.samples.append(
            TrackingSample(
                time=time,
                pos=np.asarray(pos, dtype=float).copy(),
                vel=np.asarray(vel, dtype=float).copy(),
                pos_ref=np.asarray(pos_ref, dtype=float).copy(),
                vel_ref=np.asarray(vel_ref, dtype=float).copy(),
                motors=np.asarray(motors, dtype=float).copy(),
                tilt_deg=float(tilt_deg),
            )
        )

    def finalize(
        self,
        *,
        controller: str,
        domain: str,
        settle_axis: int = 0,
        settle_fraction: float = 0.02,
        settle_hold: float = 0.5,
        step_target: float | None = None,
        height_limit: float = 3.0,
    ) -> ComparisonMetrics:
        if not self.samples:
            return ComparisonMetrics(
                controller=controller,
                domain=domain,
                rmse_position=float("inf"),
                rmse_velocity=float("inf"),
                max_position_error=float("inf"),
                max_velocity_error=float("inf"),
                max_tilt_deg=0.0,
                settling_time_s=float("inf"),
                peak_motor_thrust=0.0,
                motor_saturation_pct=0.0,
                control_effort=0.0,
                stable=False,
                notes="no samples",
            )

        pos_err = [s.pos - s.pos_ref for s in self.samples]
        vel_err = [s.vel - s.vel_ref for s in self.samples]
        pos_norm = [float(np.linalg.norm(e)) for e in pos_err]
        vel_norm = [float(np.linalg.norm(e)) for e in vel_err]

        motors = np.vstack([s.motors for s in self.samples])
        motor_abs = np.abs(motors)
        sat = np.mean(
            (motors <= self.motor_min + 0.05) | (motors >= self.motor_max - 0.05)
        ) * 100.0

        max_tilt = max(s.tilt_deg for s in self.samples)
        max_pos = max(pos_norm)
        max_vel = max(vel_norm)
        z_max = max(float(s.pos[2]) for s in self.samples)
        stable = max_pos < 5.0 and z_max < height_limit and max_tilt < 85.0

        settling = float("inf")
        if step_target is not None and self.samples:
            band = max(settle_fraction * abs(step_target), 0.05)
            hold = 0.0
            t0 = self.samples[0].time
            for sample in self.samples:
                err = abs(float(sample.pos[settle_axis] - sample.pos_ref[settle_axis]))
                if err <= band:
                    hold += sample.time - t0 if len(self.samples) > 1 else 0.004
                    if hold >= settle_hold:
                        settling = sample.time - self.samples[0].time
                        break
                else:
                    hold = 0.0
                t0 = sample.time

        effort = float(np.sum(motor_abs)) / len(self.samples)

        return ComparisonMetrics(
            controller=controller,
            domain=domain,
            rmse_position=float(np.sqrt(np.mean(np.square(pos_norm)))),
            rmse_velocity=float(np.sqrt(np.mean(np.square(vel_norm)))),
            max_position_error=max_pos,
            max_velocity_error=max_vel,
            max_tilt_deg=max_tilt,
            settling_time_s=settling,
            peak_motor_thrust=float(np.max(motor_abs)),
            motor_saturation_pct=float(sat),
            control_effort=effort,
            stable=stable,
        )
