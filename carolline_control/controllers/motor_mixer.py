"""
Motor mixer — maps desired thrust and body moment to four rotor forces.

Uses thrust-preserving differential allocation so attitude torque does not
create phantom collective lift (CAROLLINE bidirectional motors).
"""

from __future__ import annotations

import numpy as np

from carolline_control.utils.types import ControllerConfig, MotorCommand


class MotorMixer:
    """Convert collective thrust and body torque to per-motor commands."""

    def __init__(self, config: ControllerConfig) -> None:
        self._min = config.motor_min
        self._max = config.motor_max
        self._alloc = self._build_allocation(config)
        self._moment_alloc = self._alloc[1:, :]

    @staticmethod
    def _build_allocation(config: ControllerConfig) -> np.ndarray:
        """Build 4x4 wrench matrix W such that wrench = W @ thrusts."""
        positions = config.rotor_positions
        yaw_coeff = config.yaw_drag_coeff
        W = np.zeros((4, 4))
        W[0, :] = 1.0
        for i in range(4):
            x, y, _ = positions[i]
            W[1, i] = y
            W[2, i] = -x
            W[3, i] = yaw_coeff[i]
        return W

    def mix(self, thrust: float, moment: np.ndarray) -> MotorCommand:
        """Allocate rotor thrusts; sum(thrusts) matches collective thrust."""
        moment = np.asarray(moment, dtype=float)
        if abs(thrust) <= 1e-6 and np.linalg.norm(moment) < 1e-6:
            return MotorCommand(thrusts=np.zeros(4))

        base = np.full(4, thrust / 4.0, dtype=float)
        moment_err = moment - self._moment_alloc @ base
        delta = np.linalg.pinv(self._moment_alloc) @ moment_err
        delta -= np.mean(delta)

        thrusts = np.clip(base + delta, self._min, self._max)
        for _ in range(4):
            collective_err = thrust - float(np.sum(thrusts))
            thrusts = np.clip(thrusts + collective_err / 4.0, self._min, self._max)
            moment_err = moment - self._moment_alloc @ thrusts
            if abs(collective_err) < 0.02 and np.linalg.norm(moment_err) < 0.05:
                break
            delta = np.linalg.pinv(self._moment_alloc) @ moment_err
            delta -= np.mean(delta)
            thrusts = np.clip(thrusts + 0.35 * delta, self._min, self._max)

        return MotorCommand(thrusts=thrusts)

    def mix_prioritize_moment(self, thrust: float, moment: np.ndarray) -> MotorCommand:
        """Allocate rotors favoring torque tracking (used on ground rolling)."""
        moment = np.asarray(moment, dtype=float)
        if abs(thrust) <= 1e-6 and np.linalg.norm(moment) < 1e-6:
            return MotorCommand(thrusts=np.zeros(4))

        wrench = np.array([thrust, moment[0], moment[1], moment[2]], dtype=float)
        thrusts = np.linalg.pinv(self._alloc) @ wrench
        return MotorCommand(thrusts=np.clip(thrusts, self._min, self._max))
