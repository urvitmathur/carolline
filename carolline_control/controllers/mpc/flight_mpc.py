"""Linear MPC for aerial hover / waypoint tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carolline_control.controllers.lqr.flight_lqr import build_hover_linearization
from carolline_control.controllers.mpc.finite_horizon import LinearMpcSolver, discretize_euler
from carolline_control.utils.so3 import desired_rotation_from_thrust_direction, rot_to_euler_zyx
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState, TrajectoryTarget


@dataclass
class FlightMpcWeights:
    q_pos: np.ndarray
    q_vel: np.ndarray
    q_att: np.ndarray
    q_rate: np.ndarray
    qf_scale: float
    r_thrust: float
    r_moment: np.ndarray
    horizon: int = 120
    dt: float = 0.004


def default_flight_mpc_weights() -> FlightMpcWeights:
    return FlightMpcWeights(
        q_pos=np.array([14.0, 14.0, 20.0]),
        q_vel=np.array([9.0, 9.0, 11.0]),
        q_att=np.array([16.0, 16.0, 7.0]),
        q_rate=np.array([3.5, 3.5, 2.5]),
        qf_scale=2.0,
        r_thrust=0.10,
        r_moment=np.array([0.14, 0.14, 0.07]),
        horizon=15,
        dt=0.004,
    )


class FlightMpcController:
    """Receding-horizon linear MPC on hover wrench with input bounds."""

    def __init__(self, config: ControllerConfig, weights: FlightMpcWeights | None = None) -> None:
        self._config = config
        self._weights = weights or default_flight_mpc_weights()
        self._Ad: np.ndarray | None = None
        self._Bd: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._Qf: np.ndarray | None = None
        self._solver: LinearMpcSolver | None = None
        self._u_eq = np.zeros(4)
        self._u_min = np.zeros(4)
        self._u_max = np.zeros(4)
        self._build_model()

    def _build_model(self) -> None:
        w = self._weights
        mass = self._config.mass
        Ac, Bc = build_hover_linearization(mass, self._config.gravity, self._config.inertia)
        self._Ad, self._Bd = discretize_euler(Ac, Bc, w.dt)
        self._Q = np.diag(np.concatenate([w.q_pos, w.q_vel, w.q_att, w.q_rate]))
        self._R = np.diag(np.concatenate([[w.r_thrust], w.r_moment]))
        self._Qf = w.qf_scale * self._Q
        self._u_eq = np.array([mass * self._config.gravity, 0.0, 0.0, 0.0])

        max_up = 4.0 * self._config.motor_max * 0.98
        max_down = 4.0 * abs(self._config.motor_min) * 0.98
        min_thrust = -max_down if self._config.motor_min < 0.0 else 0.0
        moment_limit = 0.45 * self._config.mass * self._config.gravity * self._config.cage_radius
        self._u_min = np.array([min_thrust, -moment_limit, -moment_limit, -moment_limit])
        self._u_max = np.array([max_up, moment_limit, moment_limit, moment_limit])
        du_min = self._u_min - self._u_eq
        du_max = self._u_max - self._u_eq
        self._solver = LinearMpcSolver(
            self._Ad,
            self._Bd,
            self._Q,
            self._R,
            self._Qf,
            w.horizon,
            u_min=du_min,
            u_max=du_max,
        )

    @staticmethod
    def _state_vector(state: RobotState) -> np.ndarray:
        rpy = rot_to_euler_zyx(state.rotation)
        return np.concatenate([state.position, state.velocity, rpy, state.omega_body])

    @staticmethod
    def _reference_vector(target: TrajectoryTarget) -> np.ndarray:
        return np.array(
            [
                target.position[0], target.position[1], target.position[2],
                target.velocity[0], target.velocity[1], target.velocity[2],
                0.0, 0.0, float(target.yaw),
                0.0, 0.0, float(target.yaw_rate),
            ],
            dtype=float,
        )

    def compute(
        self,
        state: RobotState,
        target: TrajectoryTarget,
        mode: ControlMode = ControlMode.HOVER,
    ) -> ControlCommand:
        assert self._solver is not None
        x = self._state_vector(state)
        x_ref = self._reference_vector(target)
        err = x - x_ref
        err[8] = float((err[8] + np.pi) % (2.0 * np.pi) - np.pi)

        du0 = self._solver.solve(err)
        wrench = self._u_eq + du0
        thrust = float(wrench[0])
        moment = wrench[1:4].copy()

        max_up = 4.0 * self._config.motor_max * 0.98
        max_down = 4.0 * abs(self._config.motor_min) * 0.98
        min_thrust = -max_down if self._config.motor_min < 0.0 else 0.0
        thrust = float(np.clip(thrust, min_thrust, max_up))
        moment_limit = 0.45 * self._config.mass * self._config.gravity * self._config.cage_radius
        moment = np.clip(moment, -moment_limit, moment_limit)

        yaw = float(target.yaw)
        Rd = desired_rotation_from_thrust_direction(np.array([0.0, 0.0, 1.0]), yaw)
        return ControlCommand(
            thrust=thrust,
            moment_body=moment,
            desired_omega_body=np.zeros(3),
            desired_rotation=Rd,
            mode=mode,
        )
