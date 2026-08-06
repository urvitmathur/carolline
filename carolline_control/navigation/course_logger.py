"""Extended flight logger for autonomous course runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from carolline_control.logging.logger import Logger
from carolline_control.utils.types import (
    ControlCommand,
    ControlDiagnostics,
    ControlMode,
    MotorCommand,
    RobotState,
    TrajectoryTarget,
)


class CourseFlightLogger(Logger):
    """Flight logger with mission phase and perception columns."""

    HEADER = [
        *Logger.HEADER,
        "phase",
        "checkpoint",
        "cmd_speed",
        "forward_min_m",
        "mission_blocked",
    ]

    def __init__(self, path: str | Path) -> None:
        self._path, self._file = self._open_log_file(Path(path))
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log_course(
        self,
        state: RobotState,
        mode: ControlMode,
        cmd: ControlCommand,
        motor: MotorCommand,
        target: TrajectoryTarget | None,
        diagnostics: ControlDiagnostics | None,
        *,
        phase: str,
        checkpoint: int,
        cmd_speed: float,
        forward_min_m: float,
        mission_blocked: bool,
    ) -> None:
        if target is None:
            desired_pos = state.position.copy()
            desired_vel = np.zeros(3)
        else:
            desired_pos = np.asarray(target.position, dtype=float)
            desired_vel = np.asarray(target.velocity, dtype=float)

        if diagnostics is None:
            pos_err = state.position - desired_pos
            vel_err = state.velocity - desired_vel
            motors = np.asarray(motor.thrusts, dtype=float)
            diagnostics = ControlDiagnostics(
                position_error=pos_err,
                velocity_error=vel_err,
                euler_rpy=np.zeros(3),
                des_euler_rpy=np.zeros(3),
                e_R=np.zeros(3),
                omega_d_body=np.zeros(3),
                moment_body=np.zeros(3),
                motor_spread=float(np.max(motors) - np.min(motors)),
                motor_saturated=False,
                body_z_up=0.0,
                tilt_deg=0.0,
            )

        rpy_deg = np.degrees(diagnostics.euler_rpy)
        des_rpy_deg = np.degrees(diagnostics.des_euler_rpy)
        row = [
            state.time,
            mode.name,
            *state.position.tolist(),
            *state.velocity.tolist(),
            *desired_pos.tolist(),
            *desired_vel.tolist(),
            *diagnostics.position_error.tolist(),
            *diagnostics.velocity_error.tolist(),
            *rpy_deg.tolist(),
            *des_rpy_deg.tolist(),
            diagnostics.body_z_up,
            diagnostics.tilt_deg,
            cmd.thrust,
            *diagnostics.moment_body.tolist(),
            *diagnostics.e_R.tolist(),
            *diagnostics.omega_d_body.tolist(),
            *motor.thrusts.tolist(),
            diagnostics.motor_spread,
            int(diagnostics.motor_saturated),
            int(state.on_ground),
            *state.quaternion.tolist(),
            phase,
            checkpoint,
            f"{cmd_speed:.4f}",
            f"{forward_min_m:.4f}",
            int(mission_blocked),
        ]
        self._writer.writerow(row)
