"""Paper-aligned contact-frame dynamics for CAROLLINE ground locomotion.

Implements the shared downstream cascade from the CAROLLINE paper:
  contact Jacobian and weighted inverse       Eqs. (3), (4), (12)-(14)
  angular-rate proportional control           Eq. (15)
  contact-point inertia and rolling torque     Eqs. (6), (16)
  contact torque to bidirectional thrust       Eqs. (10), (11), (17), (18)

The paper presents a flat-ground model.  Here the same equations are expressed
in the measured support-plane tangent basis, which reduces exactly to the paper
form when the support normal is world +Z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carolline_control.utils.types import ControllerConfig, MotorCommand, RobotState


@dataclass
class GroundAllocation:
    """Result of the paper's minimum-effort motor-thrust allocation."""

    motor: MotorCommand
    requested_torque: np.ndarray
    achieved_torque: np.ndarray
    allocation_scale: float
    saturated: bool
    q_matrix: np.ndarray


class GroundDynamics:
    """Contact-frame kinematics, dynamics, and motor allocation."""

    E3 = np.array([0.0, 0.0, 1.0])

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    @staticmethod
    def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.asarray(fallback, dtype=float).copy()
        return vector / norm

    def contact_geometry(
        self, state: RobotState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return contact-to-COM vector in world/body and support normal."""
        normal_world = self._unit(
            state.contact_normal_world,
            self.E3,
        )
        if state.contact_valid:
            q_world = state.position - state.contact_point_world
            # Contact aggregation can slightly shorten q; preserve measured
            # direction while enforcing the known spherical-cage radius.
            q_world = self._unit(q_world, normal_world) * self.config.cage_radius
        else:
            q_world = normal_world * self.config.cage_radius
        q_body = state.rotation.T @ q_world
        return q_world, q_body, normal_world

    @staticmethod
    def tangent_basis(normal_world: np.ndarray) -> np.ndarray:
        """Return a 3x2 orthonormal basis spanning the support tangent plane."""
        normal = GroundDynamics._unit(normal_world, GroundDynamics.E3)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, normal))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        t1 = seed - normal * float(np.dot(seed, normal))
        t1 = GroundDynamics._unit(t1, np.array([0.0, 1.0, 0.0]))
        t2 = np.cross(normal, t1)
        t2 = GroundDynamics._unit(t2, np.array([0.0, 1.0, 0.0]))
        return np.column_stack((t1, t2))

    def rolling_jacobian(
        self,
        state: RobotState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build Eq. (4) in the local support-plane tangent basis."""
        q_world, _, normal_world = self.contact_geometry(state)
        tangent = self.tangent_basis(normal_world)
        axes_world = state.rotation
        velocity_columns = np.column_stack(
            [np.cross(axes_world[:, i], q_world) for i in range(3)]
        )
        jacobian = tangent.T @ velocity_columns
        return jacobian, tangent, q_world

    def desired_omega(
        self,
        state: RobotState,
        desired_velocity_xy: np.ndarray,
    ) -> np.ndarray:
        """Weighted minimum-norm angular velocity, paper Eq. (14)."""
        jacobian, tangent, _ = self.rolling_jacobian(state)
        velocity_world = np.array(
            [desired_velocity_xy[0], desired_velocity_xy[1], 0.0],
            dtype=float,
        )
        normal = state.contact_normal_world
        normal = self._unit(normal, self.E3)
        velocity_world -= normal * float(np.dot(velocity_world, normal))
        velocity_tangent = tangent.T @ velocity_world

        weights = np.asarray(self.config.ground_omega_weights, dtype=float)
        weights = np.maximum(weights, 1e-6)
        weight_inv = np.diag(1.0 / weights)
        normal_matrix = jacobian @ weight_inv @ jacobian.T
        damping = float(self.config.ground_pseudoinverse_damping)
        solve = np.linalg.pinv(normal_matrix + damping * np.eye(2))
        return weight_inv @ jacobian.T @ solve @ velocity_tangent

    def contact_inertia(self, state: RobotState) -> np.ndarray:
        """Time-varying inertia about the instantaneous contact point, Eq. (6)."""
        _, q_body, _ = self.contact_geometry(state)
        q2 = float(np.dot(q_body, q_body))
        parallel_axis = self.config.mass * (q2 * np.eye(3) - np.outer(q_body, q_body))
        return np.asarray(self.config.inertia, dtype=float) + parallel_axis

    def tracking_torque(
        self,
        state: RobotState,
        desired_omega_body: np.ndarray,
        omega_kp: np.ndarray | None = None,
    ) -> np.ndarray:
        """Angular-rate control with ramp gravity compensation, Eqs. (15)-(16)."""
        omega_des = np.asarray(desired_omega_body, dtype=float)
        kp = np.asarray(
            self.config.ground_omega_kp if omega_kp is None else omega_kp,
            dtype=float,
        )
        omega_dot_u = kp * (omega_des - state.omega_body)
        inertia_contact = self.contact_inertia(state)
        # Rigid-body gyroscopic torque depends on the measured angular rate.
        # Using omega_des here creates a fictitious impulse at command steps.
        omega = np.asarray(state.omega_body, dtype=float)
        dynamic_torque = inertia_contact @ omega_dot_u + np.cross(
            omega, inertia_contact @ omega
        )
        _, q_body, _ = self.contact_geometry(state)
        gravity_body = state.rotation.T @ np.array(
            [0.0, 0.0, -self.config.mass * self.config.gravity],
            dtype=float,
        )
        gravity_torque = np.cross(q_body, gravity_body)
        return dynamic_torque - gravity_torque

    def thrust_to_contact_torque(self, state: RobotState) -> np.ndarray:
        """Generalized paper Q matrix (Eqs. 10-11) in the body frame."""
        _, q_body, _ = self.contact_geometry(state)
        matrix = np.zeros((3, 4), dtype=float)
        for motor_index in range(4):
            contact_to_rotor = q_body + self.config.rotor_positions[motor_index]
            force_torque = np.cross(contact_to_rotor, self.E3)
            drag_torque = self.config.yaw_drag_coeff[motor_index] * self.E3
            matrix[:, motor_index] = force_torque + drag_torque
        return matrix

    def allocate(self, state: RobotState, desired_torque: np.ndarray) -> GroundAllocation:
        """Minimum-effort Eq. (18), scaled only for actuator feasibility."""
        q_matrix = self.thrust_to_contact_torque(state)
        torque = np.asarray(desired_torque, dtype=float)
        damping = float(self.config.ground_pseudoinverse_damping)
        thrust = q_matrix.T @ np.linalg.pinv(
            q_matrix @ q_matrix.T + damping * np.eye(3)
        ) @ torque

        # Q has one motor-space null direction. Use it to limit outward normal
        # thrust without changing contact torque, preventing cage lift-off.
        _, _, normal_world = self.contact_geometry(state)
        body_z_world = state.rotation[:, 2]
        normal_per_collective = float(np.dot(body_z_world, normal_world))
        collective = float(np.sum(thrust))
        outward_normal_force = normal_per_collective * collective
        max_unload = (
            self.config.ground_max_unload_fraction
            * self.config.mass
            * self.config.gravity
            * max(float(normal_world[2]), 0.0)
        )
        if outward_normal_force > max_unload and abs(normal_per_collective) > 1e-6:
            _, _, vh = np.linalg.svd(q_matrix, full_matrices=True)
            null_direction = vh[-1]
            null_collective = float(np.sum(null_direction))
            if abs(null_collective) > 1e-8:
                target_collective = max_unload / normal_per_collective
                thrust += (
                    (target_collective - collective) / null_collective
                ) * null_direction

        scale = 1.0
        for value in thrust:
            if value > self.config.motor_max:
                scale = max(scale, float(value / self.config.motor_max))
            elif value < self.config.motor_min:
                scale = max(scale, float(value / self.config.motor_min))
        if scale > 1.0:
            thrust = thrust / scale

        thrust = np.clip(thrust, self.config.motor_min, self.config.motor_max)
        achieved = q_matrix @ thrust
        return GroundAllocation(
            motor=MotorCommand(thrusts=thrust),
            requested_torque=torque.copy(),
            achieved_torque=achieved,
            allocation_scale=scale,
            saturated=scale > 1.0 + 1e-9,
            q_matrix=q_matrix,
        )
