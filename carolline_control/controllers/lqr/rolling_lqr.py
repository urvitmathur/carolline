"""LQR for ground rolling — regulates tangent velocity and body rates via contact torque."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carolline_control.controllers.ground_dynamics import GroundDynamics
from carolline_control.controllers.lqr.dare_solver import check_closed_loop_stable, discretize_euler, solve_dare
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState


@dataclass
class RollingLqrWeights:
    q_vel: np.ndarray
    q_rate: np.ndarray
    r_torque: np.ndarray
    dt: float = 0.004


def default_rolling_lqr_weights() -> RollingLqrWeights:
    return RollingLqrWeights(
        q_vel=np.array([10.0, 10.0]),
        q_rate=np.array([4.0, 4.0, 2.0]),
        r_torque=np.array([0.06, 0.06, 0.10]),
        dt=0.004,
    )


class RollingLqrController:
    """LQR torque controller on top of the paper ground allocator."""

    def __init__(self, config: ControllerConfig, weights: RollingLqrWeights | None = None) -> None:
        self._config = config
        self._weights = weights or default_rolling_lqr_weights()
        self._dynamics = GroundDynamics(config)
        self._K: np.ndarray | None = None
        self.last_allocation = None

    @property
    def gain_matrix(self) -> np.ndarray:
        assert self._K is not None
        return self._K

    def _build_gains(self, state: RobotState) -> None:
        w = self._weights
        jacobian, _, _ = self._dynamics.rolling_jacobian(state)
        J2 = jacobian[:2, :]
        Iinv = np.linalg.inv(self._dynamics.contact_inertia(state))

        n = 5
        m = 3
        Ac = np.zeros((n, n))
        Bc = np.zeros((n, m))
        Ac[0:2, 2:5] = J2
        Bc[2:5, 0:3] = Iinv

        Ad, Bd = discretize_euler(Ac, Bc, w.dt)
        Q = np.diag(np.concatenate([w.q_vel, w.q_rate]))
        R = np.diag(w.r_torque)
        K, _ = solve_dare(Ad, Bd, Q, R)
        if not check_closed_loop_stable(Ad, Bd, K):
            raise RuntimeError("Rolling LQR closed loop is unstable; tune Q/R in lqr_config.yaml")
        self._K = K

    def _tangent_velocity(self, state: RobotState) -> np.ndarray:
        _, _, normal_world = self._dynamics.contact_geometry(state)
        tangent = self._dynamics.tangent_basis(normal_world)
        v_world = np.asarray(state.velocity, dtype=float)
        normal = self._dynamics._unit(state.contact_normal_world, self._dynamics.E3)
        v_world = v_world - normal * float(np.dot(v_world, normal))
        return np.asarray(tangent.T @ v_world, dtype=float).reshape(2)

    def compute(
        self,
        state: RobotState,
        desired_velocity_xy: np.ndarray,
        *,
        hold: bool = False,
        yaw_rate: float = 0.0,
    ) -> ControlCommand:
        velocity = np.asarray(desired_velocity_xy, dtype=float).copy()
        speed = float(np.linalg.norm(velocity))
        active = hold or speed >= 1e-3 or abs(yaw_rate) >= 1e-3
        if not active:
            return ControlCommand(
                thrust=0.0,
                moment_body=np.zeros(3),
                desired_omega_body=np.zeros(3),
                desired_rotation=state.rotation.copy(),
                mode=ControlMode.ROLLING,
            )

        if self._K is None:
            self._build_gains(state)

        omega_des = self._dynamics.desired_omega(state, velocity)
        if abs(yaw_rate) >= 1e-3:
            _, _, normal_world = self._dynamics.contact_geometry(state)
            normal_body = state.rotation.T @ normal_world
            omega_des = omega_des + float(yaw_rate) * normal_body

        v_tan = self._tangent_velocity(state)
        _, _, normal_world = self._dynamics.contact_geometry(state)
        tangent = self._dynamics.tangent_basis(normal_world)
        v_des_world = np.array([velocity[0], velocity[1], 0.0], dtype=float)
        normal = self._dynamics._unit(state.contact_normal_world, self._dynamics.E3)
        v_des_world -= normal * float(np.dot(v_des_world, normal))
        v_des_tan = np.asarray(tangent.T @ v_des_world, dtype=float).reshape(2)

        x_err = np.concatenate([v_tan - v_des_tan, state.omega_body - omega_des])
        tau_fb = -self._K @ x_err

        inertia = self._dynamics.contact_inertia(state)
        omega = np.asarray(state.omega_body, dtype=float)
        gyro = np.cross(omega, inertia @ omega)
        _, q_body, _ = self._dynamics.contact_geometry(state)
        gravity_body = state.rotation.T @ np.array(
            [0.0, 0.0, -self._config.mass * self._config.gravity],
            dtype=float,
        )
        gravity_torque = np.cross(q_body, gravity_body)
        torque = gyro - gravity_torque + tau_fb
        return ControlCommand(
            thrust=0.0,
            moment_body=torque,
            desired_omega_body=omega_des,
            desired_rotation=state.rotation.copy(),
            mode=ControlMode.ROLLING,
        )

    def allocate(self, state: RobotState, command: ControlCommand):
        self.last_allocation = self._dynamics.allocate(state, command.moment_body)
        return self.last_allocation
