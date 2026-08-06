"""Linear MPC for ground rolling velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carolline_control.controllers.ground_dynamics import GroundDynamics
from carolline_control.controllers.mpc.finite_horizon import LinearMpcSolver, discretize_euler
from carolline_control.utils.types import ControlCommand, ControllerConfig, ControlMode, RobotState


@dataclass
class RollingMpcWeights:
    q_vel: np.ndarray
    q_rate: np.ndarray
    qf_scale: float
    r_torque: np.ndarray
    horizon: int = 12
    dt: float = 0.004
    torque_limit: float = 8.0


def default_rolling_mpc_weights() -> RollingMpcWeights:
    return RollingMpcWeights(
        q_vel=np.array([12.0, 12.0]),
        q_rate=np.array([5.0, 5.0, 2.5]),
        qf_scale=2.0,
        r_torque=np.array([0.08, 0.08, 0.12]),
        horizon=12,
        dt=0.004,
        torque_limit=8.0,
    )


class RollingMpcController:
    """Receding-horizon MPC on tangent velocity + body rates with torque bounds."""

    def __init__(self, config: ControllerConfig, weights: RollingMpcWeights | None = None) -> None:
        self._config = config
        self._weights = weights or default_rolling_mpc_weights()
        self._dynamics = GroundDynamics(config)
        self._Ad: np.ndarray | None = None
        self._Bd: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._Qf: np.ndarray | None = None
        self._solver: LinearMpcSolver | None = None
        self.last_allocation = None

    def _build_model(self, state: RobotState) -> None:
        w = self._weights
        jacobian, _, _ = self._dynamics.rolling_jacobian(state)
        J2 = jacobian[:2, :]
        Iinv = np.linalg.inv(self._dynamics.contact_inertia(state))
        Ac = np.zeros((5, 5))
        Bc = np.zeros((5, 3))
        Ac[0:2, 2:5] = J2
        Bc[2:5, 0:3] = Iinv
        self._Ad, self._Bd = discretize_euler(Ac, Bc, w.dt)
        self._Q = np.diag(np.concatenate([w.q_vel, w.q_rate]))
        self._R = np.diag(w.r_torque)
        self._Qf = w.qf_scale * self._Q
        lim = w.torque_limit
        self._solver = LinearMpcSolver(
            self._Ad,
            self._Bd,
            self._Q,
            self._R,
            self._Qf,
            w.horizon,
            u_min=np.array([-lim, -lim, -lim]),
            u_max=np.array([lim, lim, lim]),
        )

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

        if self._Ad is None:
            self._build_model(state)

        omega_des = self._dynamics.desired_omega(state, velocity)
        if abs(yaw_rate) >= 1e-3:
            _, _, normal_world = self._dynamics.contact_geometry(state)
            normal_body = state.rotation.T @ normal_world
            omega_des = omega_des + float(yaw_rate) * normal_body

        _, _, normal_world = self._dynamics.contact_geometry(state)
        tangent = self._dynamics.tangent_basis(normal_world)
        v_des_world = np.array([velocity[0], velocity[1], 0.0], dtype=float)
        normal = self._dynamics._unit(state.contact_normal_world, self._dynamics.E3)
        v_des_world -= normal * float(np.dot(v_des_world, normal))
        v_des_tan = np.asarray(tangent.T @ v_des_world, dtype=float).reshape(2)

        x = np.concatenate([self._tangent_velocity(state), state.omega_body])
        x_ref = np.concatenate([v_des_tan, omega_des])
        err = x - x_ref

        w = self._weights
        assert self._solver is not None
        du = self._solver.solve(err)

        inertia = self._dynamics.contact_inertia(state)
        omega = np.asarray(state.omega_body, dtype=float)
        gyro = np.cross(omega, inertia @ omega)
        _, q_body, _ = self._dynamics.contact_geometry(state)
        gravity_body = state.rotation.T @ np.array(
            [0.0, 0.0, -self._config.mass * self._config.gravity], dtype=float
        )
        gravity_torque = np.cross(q_body, gravity_body)
        torque = gyro - gravity_torque + du

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
