"""CSV flight logger with full control diagnostics."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from carolline_control.logging.paths import fallback_log_path

from carolline_control.utils.types import (
    ControlCommand,
    ControlDiagnostics,
    ControlMode,
    MotorCommand,
    RobotState,
    TrajectoryTarget,
)


class Logger:
    """Log state, targets, motor commands, and control pipeline metrics."""

    HEADER = [
        "time",
        "mode",
        "px",
        "py",
        "pz",
        "vx",
        "vy",
        "vz",
        "des_px",
        "des_py",
        "des_pz",
        "des_vx",
        "des_vy",
        "des_vz",
        "ex",
        "ey",
        "ez",
        "evx",
        "evy",
        "evz",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "des_roll_deg",
        "des_pitch_deg",
        "des_yaw_deg",
        "body_z_up",
        "tilt_deg",
        "thrust",
        "mx",
        "my",
        "mz",
        "eRx",
        "eRy",
        "eRz",
        "omega_d_x",
        "omega_d_y",
        "omega_d_z",
        "m1",
        "m2",
        "m3",
        "m4",
        "motor_spread",
        "motor_saturated",
        "on_ground",
        "qw",
        "qx",
        "qy",
        "qz",
    ]

    def __init__(self, path: str | Path) -> None:
        self._path, self._file = self._open_log_file(Path(path))
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    @staticmethod
    def _open_log_file(path: Path) -> tuple[Path, object]:
        """Open a CSV log file, with retry and timestamped fallback on Windows locks."""
        path = path.resolve()
        if path.exists() and path.is_dir():
            raise NotADirectoryError(
                f"Log path is a directory, not a file: {path}. "
                "Remove or rename it and retry."
            )

        candidates = [path, fallback_log_path(path)]
        last_error: OSError | None = None
        for attempt, candidate in enumerate(candidates):
            candidate.parent.mkdir(parents=True, exist_ok=True)
            for retry in range(3):
                try:
                    handle = candidate.open("w", newline="", encoding="utf-8")
                    if candidate != path:
                        hint = (
                            f" (close Excel/other programs using {path})"
                            if last_error is not None
                            else ""
                        )
                        print(
                            f"Warning: could not open log file {path}: {last_error}; "
                            f"using {candidate}{hint}"
                        )
                    return candidate, handle
                except OSError as exc:
                    last_error = exc
                    if retry < 2:
                        time.sleep(0.05)
            if attempt == 0:
                continue
        raise OSError(
            f"Could not open log file {path} or fallback {candidates[1]}: {last_error}. "
            "Close any program that has the CSV open (Excel, viewer) and retry, "
            "or pass --log path/to/other.csv"
        ) from last_error

    def log(
        self,
        state: RobotState,
        mode: ControlMode,
        cmd: ControlCommand,
        motor: MotorCommand,
        target: TrajectoryTarget | None = None,
        diagnostics: ControlDiagnostics | None = None,
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
        ]
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
