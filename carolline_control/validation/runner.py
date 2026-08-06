"""Headless simulation runner for validation (observer-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.logging.pipeline_tracer import PipelineAbort
from carolline_control.utils.types import ControlMode, ControllerConfig
from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.disturbances import DisturbanceManager
from carolline_control.validation.metrics import RunMetrics, compute_run_metrics
from carolline_control.validation.model_perturbation import apply_gain_overrides, apply_model_perturbations
from carolline_control.validation.observer import SimulationObserver
from carolline_control.validation.randomization import RunParameters
from carolline_control.validation.wrappers import ActuatorDelayBuffer, NoisyStateEstimator
from carolline_control.visualization.markers import compile_model_with_markers


@dataclass
class RunResult:
    metrics: RunMetrics
    observer: SimulationObserver | None
    mode_history: list[str]


class ValidationRunner:
    """Runs one Monte Carlo simulation without modifying controller internals."""

    def __init__(
        self,
        val_cfg: ValidationConfig,
        repo_root: Path | None = None,
        *,
        sensor_only: bool = False,
    ) -> None:
        self.val_cfg = val_cfg
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.sensor_only = sensor_only
        ctrl_path = self.repo_root / val_cfg.controller_config
        self.raw_config = load_raw_config(ctrl_path)
        self.base_config = load_config(ctrl_path)

    def _build_config(self, run_params: RunParameters) -> ControllerConfig:
        cfg = self.base_config
        if run_params.gain_overrides:
            cfg = apply_gain_overrides(cfg, run_params.gain_overrides)
        return cfg

    def run_single(
        self,
        run_params: RunParameters,
        *,
        observer_enabled: bool = True,
        campaign_mode: str = "nominal",
    ) -> RunResult:
        config = self._build_config(run_params)
        model = compile_model_with_markers(config.model_path, config)
        apply_model_perturbations(model, run_params)
        data = mujoco.MjData(model)

        base_estimator = StateEstimator(model, config, sensor_only=self.sensor_only)
        base_estimator.fill_inertial_params(config)
        estimator: StateEstimator | NoisyStateEstimator = base_estimator
        if any(
            [
                run_params.gyro_noise_std > 0,
                run_params.accel_noise_std > 0,
                np.any(run_params.gyro_bias),
                np.any(run_params.accel_bias),
                run_params.position_noise_std > 0,
            ]
        ):
            estimator = NoisyStateEstimator(base_estimator, run_params)

        controller = CarollineController(config)
        delay_buf = ActuatorDelayBuffer(run_params.actuator_delay_steps)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")

        observer: SimulationObserver | None = None
        if observer_enabled:
            observer = SimulationObserver(
                physics_tol=self.val_cfg.physics,
                health_cfg=self.val_cfg.health,
                motor_min=config.motor_min,
                motor_max=config.motor_max,
                mass=config.mass,
                inertia=config.inertia,
            )

        disturbances = DisturbanceManager(run_params, body_id)
        disturbances.set_campaign_mode(campaign_mode)

        data.qpos[:7] = np.asarray(run_params.initial_qpos, dtype=float)
        data.qvel[:] = np.asarray(run_params.initial_qvel, dtype=float)
        mujoco.mj_forward(model, data)
        if self.sensor_only:
            base_estimator.reset(data.qpos[:3].copy())

        duration = float(self.val_cfg.sim_duration)
        base_dt = float(model.opt.timestep)
        mission = self.raw_config.get("mission", {})
        spawn_xy = np.asarray(mission.get("spawn_xy", [0.0, 0.0]), dtype=float)

        last_mode = controller.mode_manager.mode.name
        mode_history: list[str] = [last_mode]
        mode_transitions = 0
        idle_since: float | None = None
        pipeline_aborted = False
        max_omega = 0.0
        landing_pos: np.ndarray | None = None
        rng = np.random.default_rng(run_params.seed)

        while data.time < duration and data.time < self.val_cfg.mission_timeout:
            dt = base_dt
            ctrl_dt = base_dt + float(rng.uniform(-run_params.control_dt_jitter, run_params.control_dt_jitter))
            ctrl_dt = max(ctrl_dt, 1e-6)

            state = estimator.estimate(data)
            max_omega = max(max_omega, float(np.linalg.norm(state.omega_body)))

            try:
                motor, cmd, mode, target = controller.compute(state, ctrl_dt)
            except PipelineAbort:
                pipeline_aborted = True
                break

            thrusts = np.asarray(motor.thrusts, dtype=float) * run_params.motor_effectiveness
            applied = delay_buf.push(thrusts)
            data.ctrl[:] = applied

            if observer is not None:
                wind = disturbances.apply(model, data, state.time, dt)
                observer.observe_step(
                    time=state.time,
                    mode=mode.name,
                    state=state,
                    target=target,
                    diagnostics=controller.last_diagnostics,
                    motor=motor,
                    cmd=cmd,
                    data=data,
                    model=model,
                    body_id=body_id,
                    wind=wind,
                    qacc=data.qacc.copy() if data.qacc.size else None,
                )
            else:
                disturbances.apply(model, data, state.time, dt)

            if mode.name != last_mode:
                mode_transitions += 1
                mode_history.append(mode.name)
                last_mode = mode.name

            if mode == ControlMode.LANDING:
                landing_pos = state.position.copy()

            if mode == ControlMode.IDLE:
                if idle_since is None:
                    idle_since = state.time
                elif state.time - idle_since > 2.0:
                    break
            else:
                idle_since = None

            mujoco.mj_step(model, data)

        final_mode = controller.mode_manager.mode.name
        mission_success = final_mode == "IDLE" and not pipeline_aborted

        metrics = compute_run_metrics(
            run_params=run_params,
            observer=observer or SimulationObserver(
                self.val_cfg.physics,
                self.val_cfg.health,
                config.motor_min,
                config.motor_max,
                config.mass,
                config.inertia,
            ),
            final_mode=final_mode,
            mission_success=mission_success,
            mission_time=float(data.time),
            mode_transitions=mode_transitions,
            mode_history=mode_history,
            motor_min=config.motor_min,
            motor_max=config.motor_max,
            mission_timeout=self.val_cfg.mission_timeout,
            pipeline_aborted=pipeline_aborted,
            spawn_xy=spawn_xy,
            max_omega=max_omega,
            landing_pos=landing_pos,
        )
        return RunResult(metrics=metrics, observer=observer, mode_history=mode_history)
