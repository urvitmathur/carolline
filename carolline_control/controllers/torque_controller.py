"""
Torque controller — rigid body moment on SO(3).

Implements Lee et al. (2010), Eq. 11 / Eq. 20:

    M = -kR*eR - kOmega*eOmega + Omega x J*Omega
        - J*(hat(Omega)*R^T*Rd*Omega_d - R^T*Rd*Omega_d_dot)

Assumption: Omega_d_dot = 0 unless a feedforward derivative is supplied.
"""

from __future__ import annotations

import numpy as np

from carolline_control.utils.so3 import attitude_error, hat, omega_error


class TorqueController:
    """Geometric torque controller with gyroscopic compensation."""

    def compute(
        self,
        R: np.ndarray,
        Rd: np.ndarray,
        omega: np.ndarray,
        omega_d: np.ndarray,
        inertia: np.ndarray,
        kR: float,
        kOmega: float,
        omega_d_dot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute desired body moment."""
        e_R = attitude_error(R, Rd)
        e_Omega = omega_error(omega, R, Rd, omega_d)
        omega_d_dot = (
            np.zeros(3) if omega_d_dot is None else np.asarray(omega_d_dot, dtype=float)
        )

        gyro = np.cross(omega, inertia @ omega)
        feedforward = inertia @ (
            hat(omega) @ R.T @ Rd @ omega_d - R.T @ Rd @ omega_d_dot
        )
        return -kR * e_R - kOmega * e_Omega + gyro - feedforward
