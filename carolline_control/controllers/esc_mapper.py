"""Bidirectional ESC thrust mapping (CAROLLINE paper, Sec. IV-C, Fig. 5).

Models BLHeli bidirectional PWM (center 1500 us, reverse 1040, forward 1960) as
piecewise-linear thrust curves. The mapper records equivalent ESC signals for
telemetry; MuJoCo actuators remain thrust-native [N], matching simulation practice
when the ground allocator already compensates reverse-thrust asymmetry.
"""

from __future__ import annotations

import numpy as np

from carolline_control.utils.types import MotorCommand


class EscThrustMapper:
    """Piecewise ESC signal <-> thrust conversion."""

    ESC_CENTER = 1500.0
    ESC_REVERSE = 1040.0
    ESC_FORWARD = 1960.0
    ESC_MID_LOW = 1450.0
    ESC_MID_HIGH = 1550.0

    def __init__(
        self,
        *,
        thrust_forward_max: float = 13.0,
        thrust_reverse_max: float = 10.0,
        reverse_efficiency: float = 0.72,
        enabled: bool = True,
    ) -> None:
        self.thrust_forward_max = float(thrust_forward_max)
        self.thrust_reverse_max = float(thrust_reverse_max)
        self.reverse_efficiency = float(max(reverse_efficiency, 1e-6))
        self.enabled = bool(enabled)
        self._last_esc_signals = np.zeros(4, dtype=float)

        span_f = self.ESC_FORWARD - self.ESC_CENTER
        span_r = self.ESC_CENTER - self.ESC_REVERSE
        self._forward_gain = self.thrust_forward_max / span_f
        self._reverse_gain = self.thrust_reverse_max / span_r

    def esc_to_thrust(self, esc_signal: float) -> float:
        """Physical thrust [N] from ESC PWM (piecewise linear, Fig. 5)."""
        signal = float(esc_signal)
        if signal >= self.ESC_MID_HIGH:
            return self._forward_gain * (signal - self.ESC_CENTER)
        if signal <= self.ESC_MID_LOW:
            return -self._reverse_gain * (self.ESC_CENTER - signal)
        return 0.0

    def thrust_to_esc(self, thrust: float) -> float:
        """Controller thrust [N] to ESC PWM, with reverse-thrust compensation."""
        thrust = float(np.clip(thrust, -self.thrust_reverse_max, self.thrust_forward_max))
        if abs(thrust) < 1e-9:
            return self.ESC_CENTER
        if thrust > 0.0:
            return self.ESC_CENTER + thrust / self._forward_gain
        commanded = abs(thrust) / self.reverse_efficiency
        return self.ESC_CENTER - commanded / self._reverse_gain

    def apply(self, motor: MotorCommand) -> MotorCommand:
        """Record ESC signals and pass thrust commands to MuJoCo actuators."""
        thrusts = np.asarray(motor.thrusts, dtype=float).copy()
        if self.enabled:
            for index, value in enumerate(thrusts):
                self._last_esc_signals[index] = self.thrust_to_esc(value)
        else:
            self._last_esc_signals[:] = self.ESC_CENTER
        return MotorCommand(thrusts=thrusts)

    @property
    def last_esc_signals(self) -> np.ndarray:
        return self._last_esc_signals.copy()
