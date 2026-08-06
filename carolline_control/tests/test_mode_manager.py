"""Unit tests for ModeManager transitions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.controllers.mode_manager import ModeManager
from carolline_control.utils.types import ControlMode, RobotState


def _ground_state(*, on_ground: bool, xy=(0.0, 0.0)) -> RobotState:
    return RobotState(
        position=np.array([xy[0], xy[1], 0.4 if on_ground else 0.9]),
        velocity=np.zeros(3),
        quaternion=np.array([0.924, 0.0, 0.383, 0.0]),
        rotation=np.eye(3),
        omega_body=np.zeros(3),
        omega_world=np.zeros(3),
        accel_body=np.zeros(3),
        on_ground=on_ground,
        ground_contact_z=0.0,
        time=0.0,
        contact_point_world=np.zeros(3),
        contact_normal_world=np.array([0.0, 0.0, 1.0]),
        contact_force=0.0,
        contact_confidence=1.0 if on_ground else 0.0,
        contact_valid=on_ground,
    )


def test_rolling_contact_loss_switches_to_recovery():
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    config.initial_mode = ControlMode.ROLLING
    mgr = ModeManager(config)
    grace = config.contact_loss_grace
    dt = 0.004

    mgr.update(_ground_state(on_ground=True), dt)
    assert mgr.mode == ControlMode.ROLLING
    assert mgr.contact_loss_timer == 0.0

    steps = int(np.ceil(grace / dt)) + 1
    for _ in range(steps):
        mode = mgr.update(_ground_state(on_ground=False), dt)

    assert mode == ControlMode.PRETAKEOFF
    assert mgr.resume_rolling_after_recovery is True
    assert mgr.contact_loss_timer == 0.0


def test_takeoff_uses_soft_attitude_gains():
    from carolline_control.carolline_controller import CarollineController
    from carolline_control.utils.types import ControlCommand

    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    ctrl = CarollineController(config)
    state = _ground_state(on_ground=False)
    state.position[2] = config.hover_height * 0.5
    cmd = ControlCommand(
        thrust=config.mass * config.gravity,
        moment_body=np.zeros(3),
        desired_omega_body=np.zeros(3),
        desired_rotation=np.eye(3),
        mode=ControlMode.TAKEOFF,
    )

    captured: dict[str, float] = {}

    def spy(rotation, desired_rotation, kR, omega_d_ff=None):
        captured["kR"] = float(kR)
        return np.zeros(3)

    ctrl.attitude.compute_desired_omega = spy  # type: ignore[method-assign]
    ctrl._allocate(cmd, state, ControlMode.TAKEOFF)

    assert captured["kR"] == config.kR_pre
    assert config.kR_pre < config.kR


if __name__ == "__main__":
    test_rolling_contact_loss_switches_to_recovery()
    test_takeoff_uses_soft_attitude_gains()
    print("All mode manager tests passed.")
