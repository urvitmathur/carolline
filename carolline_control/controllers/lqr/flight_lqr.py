"""Linear-quadratic regulator for aerial hover / waypoint tracking.

Linearizes a standard 12-state small-angle quadrotor about hover and regulates
position, velocity, attitude, and body rates using wrench inputs
[T, tau_x, tau_y, tau_z] before motor mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carolline_control.controllers.lqr.dare_solver import check_closed_loop_stable, discretize_euler, solve_dare
from carolline_control.utils.so3 import rot_to_euler_zyx
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState, TrajectoryTarget


@dataclass
class FlightLqrWeights:
    q_pos: np.ndarray
    q_vel: np.ndarray
    q_att: np.ndarray
    q_rate: np.ndarray
    r_thrust: float
    r_moment: np.ndarray
    dt: float = 0.004


def default_flight_lqr_weights() -> FlightLqrWeights:
    return FlightLqrWeights(
        q_pos=np.array([12.0, 12.0, 18.0]),
        q_vel=np.array([8.0, 8.0, 10.0]),
        q_att=np.array([14.0, 14.0, 6.0]),
        q_rate=np.array([3.0, 3.0, 2.0]),
        r_thrust=0.08,
        r_moment=np.array([0.12, 0.12, 0.06]),
        dt=0.004,
    )


def build_hover_linearization(mass: float, gravity: float, inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (A, B) for x=[p,v,rpy,omega], u=[T,tau]."""
    Iinv = np.linalg.inv(np.asarray(inertia, dtype=float))
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7] = gravity
    A[4, 6] = -gravity
    A[6:9, 9:12] = np.eye(3)
    B[5, 0] = 1.0 / mass
    B[9:12, 1:4] = Iinv
    return A, B


class FlightLqrController:
    """LQR wrench controller for HOVER / FLIGHT tracking."""

    def __init__(self, config: ControllerConfig, weights: FlightLqrWeights | None = None) -> None:
        self._config = config
        self._weights = weights or default_flight_lqr_weights()
        self._K: np.ndarray | None = None
        self._u_eq = np.zeros(4)
        self._build_gains()

    @property
    def gain_matrix(self) -> np.ndarray:
        assert self._K is not None
        return self._K

    def _build_gains(self) -> None:
        w = self._weights
        mass = self._config.mass
        Ac, Bc = build_hover_linearization(mass, self._config.gravity, self._config.inertia)
        Ad, Bd = discretize_euler(Ac, Bc, w.dt)
        Q = np.diag(np.concatenate([w.q_pos, w.q_vel, w.q_att, w.q_rate]))
        R = np.diag(np.concatenate([[w.r_thrust], w.r_moment]))
        K, _ = solve_dare(Ad, Bd, Q, R)
        if not check_closed_loop_stable(Ad, Bd, K):
            raise RuntimeError("Flight LQR closed loop is unstable; tune Q/R in lqr_config.yaml")
        self._K = K
        self._u_eq = np.array([mass * self._config.gravity, 0.0, 0.0, 0.0])

    @staticmethod
    def _state_vector(state: RobotState) -> np.ndarray:
        rpy = rot_to_euler_zyx(state.rotation)
        return np.concatenate(
            [
                state.position,
                state.velocity,
                rpy,
                state.omega_body,
            ]
        )

    @staticmethod
    def _reference_vector(target: TrajectoryTarget) -> np.ndarray:
        yaw = float(target.yaw)
        return np.array(
            [
                target.position[0],
                target.position[1],
                target.position[2],
                target.velocity[0],
                target.velocity[1],
                target.velocity[2],
                0.0,
                0.0,
                yaw,
                0.0,
                0.0,
                float(target.yaw_rate),
            ],
            dtype=float,
        )

    def compute(
        self,
        state: RobotState,
        target: TrajectoryTarget,
        mode: ControlMode = ControlMode.HOVER,
    ) -> ControlCommand:
        assert self._K is not None
        x = self._state_vector(state)
        x_ref = self._reference_vector(target)
        err = x - x_ref
        # Wrap yaw error to [-pi, pi]
        err[8] = float((err[8] + np.pi) % (2.0 * np.pi) - np.pi)

        wrench = self._u_eq - self._K @ err
        thrust = float(wrench[0])
        moment = wrench[1:4].copy()

        max_up = 4.0 * self._config.motor_max * 0.98
        max_down = 4.0 * abs(self._config.motor_min) * 0.98
        min_thrust = -max_down if self._config.motor_min < 0.0 else 0.0
        thrust = float(np.clip(thrust, min_thrust, max_up))

        moment_limit = 0.45 * self._config.mass * self._config.gravity * self._config.cage_radius
        moment = np.clip(moment, -moment_limit, moment_limit)

        yaw = float(target.yaw)
        from carolline_control.utils.so3 import desired_rotation_from_thrust_direction

        b3 = np.array([0.0, 0.0, 1.0])
        Rd = desired_rotation_from_thrust_direction(b3, yaw)
        return ControlCommand(
            thrust=thrust,
            moment_body=moment,
            desired_omega_body=np.zeros(3),
            desired_rotation=Rd,
            mode=mode,
        )
