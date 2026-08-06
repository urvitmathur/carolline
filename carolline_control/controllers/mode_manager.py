"""
Mode manager — finite state machine for CAROLLINE operating modes.

States: PRETAKEOFF, UPRIGHT, TAKEOFF, HOVER, FLIGHT, ROLLING, LANDING, IDLE

Transitions follow the CAROLLINE paper modular architecture (Sec. III).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode, ControllerConfig, RobotState


@dataclass
class ModeManager:
    """Automatic mode switching logic."""

    config: ControllerConfig
    mode: ControlMode = field(default=ControlMode.PRETAKEOFF)
    mission_phase: int = 0
    hover_timer: float = 0.0
    landing_timer: float = 0.0
    rolling_timer: float = 0.0
    upright_timer: float = 0.0
    pre_upright_timer: float = 0.0
    request_rolling: bool = False
    resume_rolling_after_recovery: bool = False
    rolling_stall_timer: float = 0.0
    contact_loss_timer: float = 0.0
    hold_upright_after_recovery: bool = False
    rolling_hold_requested: bool = False

    def __post_init__(self) -> None:
        self.mode = self.config.initial_mode

    def update(self, state: RobotState, dt: float, mission_complete: bool = False) -> ControlMode:
        """Evaluate transition rules and return active mode."""
        if self.mode == ControlMode.ROLLING:
            pos_err = state.position[:2] - self.config.roll_target
            dist = float(np.linalg.norm(pos_err))
            near_goal = state.on_ground and dist < self.config.roll_position_tolerance
            speed_ok = (
                float(np.linalg.norm(state.velocity[:2])) < self.config.roll_arrival_speed
            )
            if near_goal and speed_ok and not self.rolling_hold_requested:
                self.rolling_timer += dt
                if self.rolling_timer >= 0.5:
                    self.resume_rolling_after_recovery = False
                    self.mode = ControlMode.PRETAKEOFF
                    self.rolling_timer = 0.0
            else:
                self.rolling_timer = 0.0

            # Contact can disappear for a few solver frames on ramp seams. A
            # ground-mode vehicle must not silently enter an aerial hover mode.
            if state.on_ground:
                self.contact_loss_timer = 0.0
            else:
                self.contact_loss_timer += dt
                if self.contact_loss_timer >= self.config.contact_loss_grace:
                    self.resume_rolling_after_recovery = True
                    self.mode = ControlMode.PRETAKEOFF
                    self.contact_loss_timer = 0.0
                    self.pre_upright_timer = 0.0

            stalled = (
                not self.rolling_hold_requested
                and state.on_ground
                and float(np.linalg.norm(pos_err)) > 2.0 * self.config.roll_position_tolerance
                and float(np.linalg.norm(state.velocity[:2])) < self.config.rolling_stall_speed
            )
            self.rolling_stall_timer = self.rolling_stall_timer + dt if stalled else 0.0
            if self.rolling_stall_timer >= self.config.rolling_stall_time:
                self.resume_rolling_after_recovery = True
                self.mode = ControlMode.PRETAKEOFF
                self.pre_upright_timer = 0.0
                self.rolling_stall_timer = 0.0

        elif self.mode == ControlMode.PRETAKEOFF:
            bz = float(body_z_world(state.rotation)[2])
            omega_tol = max(self.config.upright_omega_tolerance, 0.55)
            omega_ok = float(np.linalg.norm(state.omega_body)) < omega_tol
            if bz >= self.config.upright_cos_threshold and omega_ok:
                self.pre_upright_timer += dt
                if self.pre_upright_timer >= self.config.pre_upright_settle_time:
                    self.mode = ControlMode.UPRIGHT
                    self.upright_timer = 0.0
            else:
                self.pre_upright_timer = 0.0

        elif self.mode == ControlMode.UPRIGHT:
            bz = float(body_z_world(state.rotation)[2])
            omega_ok = float(np.linalg.norm(state.omega_body)) < self.config.upright_omega_tolerance
            if self.request_rolling or self.resume_rolling_after_recovery:
                self.mode = ControlMode.ROLLING
                self.request_rolling = False
                self.resume_rolling_after_recovery = False
                self.upright_timer = 0.0
            elif (
                not self.hold_upright_after_recovery
                and state.on_ground
                and bz >= self.config.takeoff_cos_threshold
                and omega_ok
            ):
                self.upright_timer += dt
                if self.upright_timer >= self.config.upright_settle_time:
                    self.mode = ControlMode.TAKEOFF
                    self.upright_timer = 0.0
            else:
                self.upright_timer = 0.0

        elif self.mode == ControlMode.TAKEOFF:
            bz = float(body_z_world(state.rotation)[2])
            alt_err = abs(state.position[2] - self.config.hover_height)
            alt_ok = alt_err < self.config.hover_altitude_tolerance
            upright = bz >= self.config.takeoff_cos_threshold
            vel_z_ok = abs(state.velocity[2]) < self.config.hover_velocity_tolerance
            vel_xy = float(np.linalg.norm(state.velocity[:2]))
            vel_xy_ok = vel_xy < self.config.takeoff_horizontal_velocity_tolerance
            # Require near-target altitude from below (or settled) to avoid locking in during overshoot.
            at_or_below_hover = state.position[2] <= self.config.hover_height + 0.02
            if alt_ok and upright and vel_z_ok and vel_xy_ok and at_or_below_hover:
                self.mode = ControlMode.HOVER
                self.hover_timer = 0.0

        elif self.mode == ControlMode.HOVER:
            self.hover_timer += dt
            bz = float(body_z_world(state.rotation)[2])
            alt_err = abs(state.position[2] - self.config.hover_height)
            alt_ok = alt_err < self.config.hover_altitude_tolerance
            vel_ok = float(np.linalg.norm(state.velocity)) < self.config.hover_velocity_tolerance
            upright = bz >= self.config.upright_cos_threshold
            if self.hover_timer >= self.config.hover_before_flight_time and alt_ok and vel_ok and upright:
                self.mode = ControlMode.FLIGHT

        elif self.mode == ControlMode.FLIGHT:
            if mission_complete:
                self.mode = ControlMode.LANDING
                self.landing_timer = 0.0

        elif self.mode == ControlMode.LANDING:
            bz = body_z_world(state.rotation)[2]
            if state.on_ground and bz < self.config.upright_cos_threshold:
                self.hold_upright_after_recovery = True
                self.mode = ControlMode.PRETAKEOFF
                self.landing_timer = 0.0
            elif state.on_ground and bz >= self.config.upright_cos_threshold:
                settled = (
                    np.linalg.norm(state.velocity) < 0.05
                    and abs(state.position[2] - state.ground_contact_z) < 0.05
                )
                if settled:
                    self.landing_timer += dt
                    if self.landing_timer >= self.config.landing_settle_time:
                        self.mode = ControlMode.IDLE
                else:
                    self.landing_timer = 0.0

        return self.mode

    def set_mission_phase(self, phase: int) -> None:
        self.mission_phase = phase

    def request_ground_recovery(self, *, resume_rolling: bool) -> None:
        """Explicitly enter paper pre-takeoff recovery."""
        self.resume_rolling_after_recovery = bool(resume_rolling)
        self.hold_upright_after_recovery = False
        self.request_rolling = False
        self.pre_upright_timer = 0.0
        self.upright_timer = 0.0
        self.mode = ControlMode.PRETAKEOFF

    def request_roll(self) -> None:
        """Resume rolling after the vehicle reaches its upright ground state."""
        self.hold_upright_after_recovery = False
        if self.mode == ControlMode.ROLLING:
            self.request_rolling = False
            self.resume_rolling_after_recovery = False
            return
        if self.mode in (ControlMode.IDLE, ControlMode.LANDING):
            self.mode = ControlMode.ROLLING
            self.request_rolling = False
            return
        self.request_rolling = True

    def request_roll_hold(self, enabled: bool = True) -> None:
        """Keep ROLLING active at its target for contact-frame position holding."""
        self.rolling_hold_requested = bool(enabled)
        if enabled:
            self.request_roll()

    def request_takeoff(self) -> None:
        self.request_ground_recovery(resume_rolling=False)

    def request_flight(self) -> None:
        self.mode = ControlMode.FLIGHT

    def request_landing(self) -> None:
        self.mode = ControlMode.LANDING
        self.landing_timer = 0.0

    def request_idle(self) -> None:
        self.mode = ControlMode.IDLE
