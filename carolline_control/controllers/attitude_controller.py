"""
Attitude controller — proportional law on SO(3).

Implements Lee et al. (2010), Eq. 11 first block:
    Omega_d = Omega_d_from_Rd  (via feedforward in torque controller)

This class supplies the desired angular velocity feedforward used in the
torque controller (Lee et al., Eq. 11). For flight modes, Omega_d is zero and
attitude regulation is handled by the -kR*eR term in TorqueController. Rolling
mode passes a non-zero body-rate feedforward from the rolling controller.
"""

from __future__ import annotations

import numpy as np


class AttitudeController:
    """Proportional SO(3) attitude controller."""

    def compute_desired_omega(
        self,
        R: np.ndarray,
        Rd: np.ndarray,
        kR: float,
        omega_d_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return desired body angular velocity.

        Parameters
        ----------
        R:
            Current rotation matrix (body -> world).
        Rd:
            Desired rotation matrix.
        kR:
            Proportional gain on e_R.
        omega_d_ff:
            Optional feedforward angular velocity (e.g., rolling mode).
        """
        if omega_d_ff is not None:
            return np.asarray(omega_d_ff, dtype=float).copy()
        return np.zeros(3)
