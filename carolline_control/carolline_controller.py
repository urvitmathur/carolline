"""
CAROLLINE integrated control stack.

Control loop:
    Sensors -> StateEstimator -> ModeManager ->
    (RollingController | FlightController | PreTakeoffController) ->
    AttitudeController -> TorqueController -> MotorMixer -> MuJoCo
"""

from __future__ import annotations

import numpy as np

from carolline_control.controllers.attitude_controller import AttitudeController
from carolline_control.controllers.esc_mapper import EscThrustMapper
from carolline_control.controllers.flight_controller import FlightController
from carolline_control.controllers.mode_manager import ModeManager
from carolline_control.controllers.motor_mixer import MotorMixer
from carolline_control.controllers.planner import Planner
from carolline_control.controllers.pre_takeoff_controller import PreTakeoffController
from carolline_control.controllers.rolling_controller import RollingController
from carolline_control.controllers.torque_controller import TorqueController
from carolline_control.logging.pipeline_tracer import PipelineAbort, PipelineTracer
from carolline_control.utils.so3 import attitude_error, body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import (
    ControlCommand,
    ControlDiagnostics,
    ControlMode,
    ControllerConfig,
    MotorCommand,
    RobotState,
    TrajectoryTarget,
)


class CarollineController:
    """Top-level controller integrating all subsystems."""

    def __init__(self, config: ControllerConfig, pipeline_tracer: PipelineTracer | None = None) -> None:
        self.config = config
        self.mode_manager = ModeManager(config)
        self.pre_takeoff = PreTakeoffController(config)
        self.rolling = RollingController(config)
        self.flight = FlightController(config)
        self.planner = Planner(config)
        self.attitude = AttitudeController()
        self.torque = TorqueController()
        self.mixer = MotorMixer(config)
        self.esc_mapper = EscThrustMapper(
            thrust_forward_max=config.esc_thrust_forward_max,
            thrust_reverse_max=config.esc_thrust_reverse_max,
            reverse_efficiency=config.esc_reverse_efficiency,
            enabled=config.esc_mapping_enabled,
        )
        self._last_omega_d = np.zeros(3)
        self._control_dt = 0.004
        self._pipeline_tracer = pipeline_tracer
        self._trace_omega_d = np.zeros(3)
        self._trace_moment = np.zeros(3)
        self._last_motors = np.zeros(4)
        self._hover_anchor_xy = config.roll_target.copy()
        self._flight_yaw = config.mission_yaw
        self._last_diagnostics = ControlDiagnostics(
            position_error=np.zeros(3),
            velocity_error=np.zeros(3),
            euler_rpy=np.zeros(3),
            des_euler_rpy=np.zeros(3),
            e_R=np.zeros(3),
            omega_d_body=np.zeros(3),
            moment_body=np.zeros(3),
            motor_spread=0.0,
            motor_saturated=False,
            body_z_up=1.0,
            tilt_deg=0.0,
        )

    @property
    def last_diagnostics(self) -> ControlDiagnostics:
        return self._last_diagnostics

    def _apply_esc_mapping(self, motor: MotorCommand) -> MotorCommand:
        """Map ideal thrust commands through bidirectional ESC curves (paper Fig. 5)."""
        return self.esc_mapper.apply(motor)

    def _slew_motors(self, motor: MotorCommand, dt: float) -> MotorCommand:
        """Limit per-motor thrust rate to avoid abrupt saturation steps."""
        motor = self._apply_esc_mapping(motor)
        if np.allclose(self._last_motors, 0.0):
            self._last_motors = motor.thrusts.copy()
            return motor
        max_step = self.config.motor_slew_rate * max(dt, 1e-6)
        delta = np.clip(motor.thrusts - self._last_motors, -max_step, max_step)
        smoothed = self._last_motors + delta
        self._last_motors = smoothed.copy()
        return MotorCommand(thrusts=smoothed)

    def compute(
        self,
        state: RobotState,
        dt: float,
    ) -> tuple[MotorCommand, ControlCommand, ControlMode, TrajectoryTarget]:
        """Run one control iteration."""
        self._control_dt = max(dt, 1e-6)
        prev_mode = self.mode_manager.mode
        mode = self.mode_manager.update(
            state,
            dt,
            mission_complete=self.planner.mission_complete,
        )
        if mode == ControlMode.FLIGHT and prev_mode != ControlMode.FLIGHT:
            self.planner.begin_flight(state, yaw=self._flight_yaw)
        if prev_mode == ControlMode.UPRIGHT and mode == ControlMode.TAKEOFF:
            self._hover_anchor_xy = state.position[:2].copy()
            self._flight_yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            self.planner.set_flight_yaw(self._flight_yaw)
        if prev_mode != ControlMode.HOVER and mode == ControlMode.HOVER:
            self._hover_anchor_xy = state.position[:2].copy()
        if prev_mode == ControlMode.ROLLING and mode != ControlMode.ROLLING:
            self.rolling.reset()
        if mode == ControlMode.ROLLING and prev_mode != ControlMode.ROLLING:
            self.rolling.reset()
        if mode == ControlMode.PRETAKEOFF and prev_mode != ControlMode.PRETAKEOFF:
            self.pre_takeoff.reset(state)
        target = self._plan_target(state, mode, dt)
        cmd = self._mode_command(state, mode, target)
        motor = self._allocate(cmd, state, mode)
        self._last_diagnostics = self._build_diagnostics(state, target, cmd, motor)
        if self._pipeline_tracer is not None:
            if not self._pipeline_tracer.observe(
                state=state,
                prev_mode=prev_mode,
                mode=mode,
                target=target,
                cmd=cmd,
                motor=motor,
                omega_d=self._trace_omega_d,
                moment=self._trace_moment,
            ):
                raise PipelineAbort("Invalid control value detected by pipeline tracer.")
        return motor, cmd, mode, target

    def compute_manual(
        self,
        state: RobotState,
        dt: float,
        mode: ControlMode,
        *,
        velocity_xy: np.ndarray | None = None,
        velocity_world: np.ndarray | None = None,
        yaw_rate: float = 0.0,
    ) -> tuple[MotorCommand, ControlCommand, ControlMode, TrajectoryTarget]:
        """Run one control iteration with user-selected mode and velocity commands."""
        self._control_dt = max(dt, 1e-6)
        prev_mode = self.mode_manager.mode
        self.mode_manager.mode = mode
        self._apply_manual_mode_transition(state, prev_mode, mode)
        target = self._plan_target_manual(state, mode, velocity_xy, velocity_world, yaw_rate)
        cmd = self._mode_command_manual(
            state,
            mode,
            target,
            velocity_xy,
            velocity_world,
            yaw_rate,
        )
        motor = self._allocate(cmd, state, mode)
        self._last_diagnostics = self._build_diagnostics(state, target, cmd, motor)
        if self._pipeline_tracer is not None:
            if not self._pipeline_tracer.observe(
                state=state,
                prev_mode=prev_mode,
                mode=mode,
                target=target,
                cmd=cmd,
                motor=motor,
                omega_d=self._trace_omega_d,
                moment=self._trace_moment,
            ):
                raise PipelineAbort("Invalid control value detected by pipeline tracer.")
        return motor, cmd, mode, target

    def _apply_manual_mode_transition(
        self,
        state: RobotState,
        prev_mode: ControlMode,
        mode: ControlMode,
    ) -> None:
        if mode == ControlMode.FLIGHT and prev_mode != ControlMode.FLIGHT:
            self.planner.begin_flight(state, yaw=self._flight_yaw)
        if prev_mode == ControlMode.UPRIGHT and mode == ControlMode.TAKEOFF:
            self._hover_anchor_xy = state.position[:2].copy()
            self._flight_yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            self.planner.set_flight_yaw(self._flight_yaw)
        if prev_mode != ControlMode.HOVER and mode == ControlMode.HOVER:
            self._hover_anchor_xy = state.position[:2].copy()
        if prev_mode == ControlMode.ROLLING and mode != ControlMode.ROLLING:
            self.rolling.reset()
        if mode == ControlMode.ROLLING and prev_mode != ControlMode.ROLLING:
            self.rolling.reset()
        if mode == ControlMode.PRETAKEOFF and prev_mode != ControlMode.PRETAKEOFF:
            self.pre_takeoff.reset(state)
        if mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT) and prev_mode != mode:
            self._flight_yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            if mode in (ControlMode.TAKEOFF, ControlMode.HOVER):
                self._hover_anchor_xy = state.position[:2].copy()

    def _plan_target_manual(
        self,
        state: RobotState,
        mode: ControlMode,
        velocity_xy: np.ndarray | None,
        velocity_world: np.ndarray | None,
        yaw_rate: float,
    ) -> TrajectoryTarget:
        yaw = self._flight_yaw
        if mode in (ControlMode.ROLLING, ControlMode.IDLE, ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
        lead = 0.75
        vel_xy = np.zeros(2) if velocity_xy is None else np.asarray(velocity_xy, dtype=float)[:2]
        vel_w = np.zeros(3) if velocity_world is None else np.asarray(velocity_world, dtype=float)[:3]

        if mode == ControlMode.ROLLING:
            return TrajectoryTarget(
                position=state.position.copy(),
                velocity=np.array([vel_xy[0], vel_xy[1], 0.0]),
                acceleration=np.zeros(3),
                yaw=yaw,
                yaw_rate=yaw_rate,
            )
        if mode == ControlMode.TAKEOFF:
            return TrajectoryTarget(
                position=np.array(
                    [self._hover_anchor_xy[0], self._hover_anchor_xy[1], self.config.hover_height]
                ),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=yaw,
            )
        if mode == ControlMode.HOVER:
            if np.linalg.norm(vel_w) > 1e-6:
                return TrajectoryTarget(
                    position=state.position.copy(),
                    velocity=vel_w,
                    acceleration=np.zeros(3),
                    yaw=yaw,
                    yaw_rate=yaw_rate,
                )
            return TrajectoryTarget(
                position=state.position.copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=yaw,
            )
        if mode == ControlMode.FLIGHT:
            if np.linalg.norm(vel_w) > 1e-6:
                return TrajectoryTarget(
                    position=state.position.copy(),
                    velocity=vel_w,
                    acceleration=np.zeros(3),
                    yaw=yaw,
                    yaw_rate=yaw_rate,
                )
            return TrajectoryTarget(
                position=state.position.copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=yaw,
            )
        if mode == ControlMode.LANDING:
            return TrajectoryTarget(
                position=np.array(
                    [state.position[0], state.position[1], self.config.landing_height]
                ),
                velocity=vel_w,
                acceleration=np.zeros(3),
                yaw=yaw,
                yaw_rate=yaw_rate,
            )
        if mode == ControlMode.IDLE:
            return TrajectoryTarget(
                position=state.position.copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=yaw,
            )
        return TrajectoryTarget(
            position=np.array([state.position[0], state.position[1], self.config.hover_height]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=yaw,
        )

    def _mode_command_manual(
        self,
        state: RobotState,
        mode: ControlMode,
        target: TrajectoryTarget,
        velocity_xy: np.ndarray | None,
        velocity_world: np.ndarray | None,
        yaw_rate: float,
    ) -> ControlCommand:
        if mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
            return self.pre_takeoff.compute(state, mode)
        if mode == ControlMode.ROLLING:
            vel_xy = np.zeros(2) if velocity_xy is None else np.asarray(velocity_xy, dtype=float)[:2]
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            hold = self.mode_manager.rolling_hold_requested
            return self.rolling.compute(
                state, vel_xy, yaw=yaw, yaw_rate=yaw_rate, dt=self._control_dt, hold=hold
            )
        if mode == ControlMode.TAKEOFF:
            return self.flight.takeoff_thrust(
                state,
                self.config.hover_height,
                self._hover_anchor_xy,
                self._flight_yaw,
            )
        if mode == ControlMode.HOVER:
            vel_cmd = np.zeros(3) if velocity_world is None else np.asarray(velocity_world, dtype=float)[:3]
            if abs(yaw_rate) > 1e-3:
                self._flight_yaw += float(yaw_rate) * self._control_dt
            if np.linalg.norm(vel_cmd) > 0.05:
                return self.flight.compute_velocity(state, vel_cmd, self._flight_yaw, mode)
            return self.flight.compute(state, target, mode)
        if mode == ControlMode.LANDING:
            cmd = self.flight.compute(state, target, mode)
            if state.on_ground:
                weight = self.config.mass * self.config.gravity
                if body_z_world(state.rotation)[2] < self.config.upright_cos_threshold:
                    return self.pre_takeoff.compute(state, ControlMode.LANDING)
                yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
                return ControlCommand(
                    thrust=0.05 * weight,
                    moment_body=np.zeros(3),
                    desired_omega_body=np.zeros(3),
                    desired_rotation=self.flight._orientation.hover_yaw(yaw),
                    mode=mode,
                )
            return cmd
        if mode == ControlMode.IDLE:
            weight = self.config.mass * self.config.gravity
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            return ControlCommand(
                thrust=0.0,
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=self.flight._orientation.hover_yaw(yaw),
                mode=mode,
            )
        if mode == ControlMode.FLIGHT:
            vel_cmd = np.zeros(3) if velocity_world is None else np.asarray(velocity_world, dtype=float)[:3]
            if abs(yaw_rate) > 1e-3:
                self._flight_yaw += float(yaw_rate) * self._control_dt
            if np.linalg.norm(vel_cmd) > 0.05:
                return self.flight.compute_velocity(state, vel_cmd, self._flight_yaw, mode)
            return self.flight.compute(state, target, mode)
        return self.flight.compute(state, target, mode)

    def _plan_target(self, state: RobotState, mode: ControlMode, dt: float) -> TrajectoryTarget:
        if mode == ControlMode.TAKEOFF:
            return TrajectoryTarget(
                position=np.array(
                    [self._hover_anchor_xy[0], self._hover_anchor_xy[1], self.config.hover_height]
                ),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=self._flight_yaw,
            )
        if mode == ControlMode.HOVER:
            return TrajectoryTarget(
                position=np.array(
                    [self._hover_anchor_xy[0], self._hover_anchor_xy[1], self.config.hover_height]
                ),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=self._flight_yaw,
            )
        if mode in (ControlMode.FLIGHT,):
            return self.planner.update(state, dt)
        if mode == ControlMode.ROLLING:
            roll_pos = self.planner.rolling_target_position(state)
            return TrajectoryTarget(
                position=roll_pos,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0])),
            )
        if mode == ControlMode.LANDING:
            return TrajectoryTarget(
                position=np.array(
                    [self.config.spawn_xy[0], self.config.spawn_xy[1], self.config.landing_height]
                ),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=self._flight_yaw,
            )
        if mode == ControlMode.IDLE:
            return TrajectoryTarget(
                position=state.position.copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0])),
            )
        return TrajectoryTarget(
            position=np.array([state.position[0], state.position[1], self.config.hover_height]),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=0.0,
        )

    def _mode_command(
        self,
        state: RobotState,
        mode: ControlMode,
        target: TrajectoryTarget,
    ) -> ControlCommand:
        if mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
            return self.pre_takeoff.compute(state, mode)
        if mode == ControlMode.ROLLING:
            v_xy = self.planner.rolling_velocity(state)
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            hold = self.mode_manager.rolling_hold_requested
            return self.rolling.compute(state, v_xy, yaw=yaw, dt=self._control_dt, hold=hold)
        if mode == ControlMode.TAKEOFF:
            return self.flight.takeoff_thrust(
                state,
                self.config.hover_height,
                self._hover_anchor_xy,
                self._flight_yaw,
            )
        if mode == ControlMode.HOVER:
            return self.flight.hover(
                state,
                self.config.hover_height,
                self._hover_anchor_xy,
                self._flight_yaw,
            )
        if mode == ControlMode.LANDING:
            land_target = TrajectoryTarget(
                position=np.array(
                    [self.config.spawn_xy[0], self.config.spawn_xy[1], self.config.landing_height]
                ),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=self._flight_yaw,
            )
            cmd = self.flight.compute(state, land_target, mode)
            if state.on_ground:
                weight = self.config.mass * self.config.gravity
                if body_z_world(state.rotation)[2] < self.config.upright_cos_threshold:
                    return self.pre_takeoff.compute(state, ControlMode.LANDING)
                yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
                cmd = ControlCommand(
                    thrust=0.05 * weight,
                    moment_body=np.zeros(3),
                    desired_omega_body=np.zeros(3),
                    desired_rotation=self.flight._orientation.hover_yaw(yaw),
                    mode=mode,
                )
            return cmd
        if mode == ControlMode.IDLE:
            weight = self.config.mass * self.config.gravity
            yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
            return ControlCommand(
                thrust=0.0,
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=self.flight._orientation.hover_yaw(yaw),
                mode=mode,
            )
        return self.flight.compute(state, target, mode)

    def _allocate(self, cmd: ControlCommand, state: RobotState, mode: ControlMode) -> MotorCommand:
        if mode == ControlMode.IDLE:
            self._trace_omega_d = np.zeros(3)
            self._trace_moment = np.zeros(3)
            self._last_motors = np.zeros(4)
            return MotorCommand(thrusts=np.zeros(4))

        if mode == ControlMode.ROLLING:
            self._trace_omega_d = cmd.desired_omega_body.copy()
            allocation = self.rolling.allocate(state, cmd)
            self._trace_moment = allocation.requested_torque.copy()
            return self._slew_motors(allocation.motor, self._control_dt)

        ground_recovery = mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT) or (
            mode == ControlMode.LANDING
            and state.on_ground
            and body_z_world(state.rotation)[2] < self.config.upright_cos_threshold
        )
        if ground_recovery:
            self._trace_omega_d = cmd.desired_omega_body.copy()
            allocation = self.pre_takeoff.allocate(state, cmd)
            self._trace_moment = allocation.requested_torque.copy()
            return self._slew_motors(allocation.motor, self._control_dt)

        gains = (
            (self.config.kR_pre, self.config.kOmega_pre)
            if mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT, ControlMode.ROLLING)
            else (self.config.kR, self.config.kOmega)
        )
        omega_d = self.attitude.compute_desired_omega(
            state.rotation,
            cmd.desired_rotation,
            gains[0],
            omega_d_ff=cmd.desired_omega_body if np.linalg.norm(cmd.desired_omega_body) > 1e-9 else None,
        )
        omega_d_dot = np.zeros(3)
        self._last_omega_d = omega_d.copy()

        moment = self.torque.compute(
            state.rotation,
            cmd.desired_rotation,
            state.omega_body,
            omega_d,
            self.config.inertia,
            gains[0],
            gains[1],
            omega_d_dot=omega_d_dot,
        )
        if np.linalg.norm(cmd.moment_body) > 0:
            moment += cmd.moment_body
        self._trace_omega_d = omega_d.copy()
        self._trace_moment = moment.copy()
        motor = self.mixer.mix(cmd.thrust, moment)
        return self._slew_motors(motor, self._control_dt)

    def _build_diagnostics(
        self,
        state: RobotState,
        target: TrajectoryTarget,
        cmd: ControlCommand,
        motor: MotorCommand,
    ) -> ControlDiagnostics:
        pos_err = state.position - target.position
        vel_err = state.velocity - target.velocity
        bz = body_z_world(state.rotation)
        body_z_up = float(bz[2])
        tilt_deg = float(np.degrees(np.arccos(np.clip(body_z_up, -1.0, 1.0))))
        motors = np.asarray(motor.thrusts, dtype=float)
        sat_lo = np.any(motors <= self.config.motor_min + 0.05)
        sat_hi = np.any(motors >= self.config.motor_max - 0.05)
        allocation = None
        if cmd.mode == ControlMode.ROLLING:
            allocation = self.rolling.last_allocation
        elif cmd.mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT) or (
            cmd.mode == ControlMode.LANDING
            and state.on_ground
            and body_z_world(state.rotation)[2] < self.config.upright_cos_threshold
        ):
            allocation = self.pre_takeoff.last_allocation
        return ControlDiagnostics(
            position_error=pos_err,
            velocity_error=vel_err,
            euler_rpy=rot_to_euler_zyx(state.rotation),
            des_euler_rpy=rot_to_euler_zyx(cmd.desired_rotation),
            e_R=attitude_error(state.rotation, cmd.desired_rotation),
            omega_d_body=self._trace_omega_d.copy(),
            moment_body=self._trace_moment.copy(),
            motor_spread=float(np.max(motors) - np.min(motors)),
            motor_saturated=bool(sat_lo or sat_hi),
            body_z_up=body_z_up,
            tilt_deg=tilt_deg,
            achievable_contact_torque=(
                allocation.achieved_torque.copy()
                if allocation is not None
                else np.zeros(3)
            ),
            allocation_scale=(
                float(allocation.allocation_scale)
                if allocation is not None
                else 1.0
            ),
            contact_normal_world=state.contact_normal_world.copy(),
            ground_allocation_active=allocation is not None,
        )
