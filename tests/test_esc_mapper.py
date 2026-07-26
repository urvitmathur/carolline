"""Tests for bidirectional ESC thrust mapping (paper Sec. IV-C)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.controllers.esc_mapper import EscThrustMapper
from carolline_control.utils.types import MotorCommand


def test_esc_center_is_zero_thrust() -> None:
    mapper = EscThrustMapper()
    assert abs(mapper.esc_to_thrust(1500.0)) < 1e-9
    assert abs(mapper.thrust_to_esc(0.0) - 1500.0) < 1e-9


def test_forward_and_reverse_extents() -> None:
    mapper = EscThrustMapper(reverse_efficiency=1.0)
    assert np.isclose(mapper.esc_to_thrust(1960.0), 13.0, atol=1e-6)
    assert np.isclose(mapper.esc_to_thrust(1040.0), -10.0, atol=1e-6)


def test_midband_continuity() -> None:
    mapper = EscThrustMapper(reverse_efficiency=1.0)
    low = mapper.esc_to_thrust(1450.0)
    high = mapper.esc_to_thrust(1550.0)
    assert abs(low) < 1.5
    assert abs(high) < 1.5


def test_reverse_compensation_increases_esc_deflection() -> None:
    ideal = EscThrustMapper(reverse_efficiency=1.0)
    compensated = EscThrustMapper(reverse_efficiency=0.72)
    esc_ideal = ideal.thrust_to_esc(-5.0)
    esc_comp = compensated.thrust_to_esc(-5.0)
    assert esc_comp < esc_ideal


def test_apply_updates_esc_signals() -> None:
    mapper = EscThrustMapper()
    motor = MotorCommand(thrusts=np.array([5.0, -4.0, 0.0, -7.0]))
    mapped = mapper.apply(motor)
    assert np.allclose(mapped.thrusts, motor.thrusts)
    assert np.all(mapper.last_esc_signals >= 1040.0)
    assert np.all(mapper.last_esc_signals <= 1960.0)


def test_disabled_mapper_is_passthrough() -> None:
    mapper = EscThrustMapper(enabled=False)
    motor = MotorCommand(thrusts=np.array([3.0, -6.0, 1.0, -2.0]))
    mapped = mapper.apply(motor)
    assert np.allclose(mapped.thrusts, motor.thrusts)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
