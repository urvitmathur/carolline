"""
State estimation from MuJoCo sensors and kinematics.

Step 1 of the CAROLLINE architecture: reads position, quaternion,
linear/angular velocity and constructs R, body/world vectors.
No Euler angles are used internally.
"""

from __future__ import annotations

import mujoco
import numpy as np

from carolline_control.utils.so3 import quat_to_rot
from carolline_control.utils.types import ControllerConfig, RobotState


class StateEstimator:
    """Simulator-backed state estimator."""

    def __init__(self, model: mujoco.MjModel, config: ControllerConfig) -> None:
        self._model = model
        self._config = config
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._cage_geom_ids = self._find_cage_geoms()

        self._sensor_quat = self._sensor_adr("body_quat", 4)
        self._sensor_gyro = self._sensor_adr("body_gyro", 3)
        self._sensor_acc = self._sensor_adr("body_linacc", 3)
        self._sensor_linvel = self._try_sensor_adr("body_vel", 3)
        self._sensor_angvel = self._try_sensor_adr("body_angvel", 3)

    def _find_cage_geoms(self) -> set[int]:
        """Find collision geoms rigidly belonging to the protective cage."""
        ids: set[int] = set()
        for name in ("cage_collision",):
            geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                ids.add(int(geom_id))
        if ids:
            return ids

        # Fallback for detailed rib-cage models: every geom below a body whose
        # name contains "cage" is considered part of the ground-contact shell.
        cage_bodies: set[int] = set()
        for body_id in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name and "cage" in name.lower():
                cage_bodies.add(body_id)
        for geom_id in range(self._model.ngeom):
            body_id = int(self._model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id in cage_bodies:
                    ids.add(geom_id)
                    break
                body_id = int(self._model.body_parentid[body_id])
        return ids

    def _support_contact(
        self,
        data: mujoco.MjData,
        position: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray, float]:
        """Return force-weighted support contact point and outward normal."""
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if (
                int(contact.geom1) not in self._cage_geom_ids
                and int(contact.geom2) not in self._cage_geom_ids
            ):
                continue

            point = np.asarray(contact.pos, dtype=float).copy()
            normal = np.asarray(contact.frame[:3], dtype=float).copy()
            to_center = position - point
            if float(np.dot(normal, to_center)) < 0.0:
                normal *= -1.0
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-9:
                continue
            normal /= normal_norm

            # Side impacts are useful collision information but are not a
            # support surface for rolling/takeoff mode decisions.
            if normal[2] < self._config.contact_min_normal_z:
                continue

            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self._model, data, contact_id, wrench)
            force = abs(float(wrench[0]))
            if force < self._config.contact_force_threshold:
                continue
            candidates.append((point, normal, force))

        if not candidates:
            return False, position - np.array([0.0, 0.0, self._config.cage_radius]), np.array(
                [0.0, 0.0, 1.0]
            ), 0.0

        weights = [max(item[2], 1e-6) for item in candidates]
        total_weight = sum(weights)
        total_force = sum(item[2] for item in candidates)
        point = sum(item[0] * weight for item, weight in zip(candidates, weights)) / total_weight
        normal = sum(item[1] * weight for item, weight in zip(candidates, weights))
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        return True, point, normal, total_force

    def _sensor_adr(self, name: str, size: int) -> slice:
        adr = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if adr < 0:
            raise ValueError(f"Required sensor '{name}' not found in MJCF model.")
        start = self._model.sensor_adr[adr]
        return slice(start, start + size)

    def _try_sensor_adr(self, name: str, size: int) -> slice | None:
        adr = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if adr < 0:
            return None
        start = self._model.sensor_adr[adr]
        return slice(start, start + size)

    def estimate(self, data: mujoco.MjData) -> RobotState:
        """Build RobotState from MuJoCo data buffers."""
        position = data.qpos[:3].copy()
        quaternion = data.qpos[3:7].copy()
        rotation = quat_to_rot(quaternion)

        if self._sensor_linvel is not None:
            velocity = data.sensordata[self._sensor_linvel].copy()
        else:
            velocity = data.qvel[:3].copy()

        if self._sensor_angvel is not None:
            omega_world = data.sensordata[self._sensor_angvel].copy()
        else:
            omega_world = data.qvel[3:6].copy()
        omega_body = rotation.T @ omega_world

        accel_body = data.sensordata[self._sensor_acc].copy()
        contact_valid, contact_point, contact_normal, contact_force = self._support_contact(
            data, position
        )
        # Treat the vehicle as grounded while the cage center remains near the
        # support plane, even if the contact solver briefly loses rib contacts.
        on_ground = contact_valid or position[2] <= self._config.cage_radius + 0.12
        if contact_valid:
            # Existing landing logic expects a center-height reference.
            ground_contact_z = float(position[2])
        else:
            ground_contact_z = self._config.cage_radius

        return RobotState(
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            rotation=rotation,
            omega_body=omega_body,
            omega_world=omega_world,
            accel_body=accel_body,
            on_ground=on_ground,
            ground_contact_z=ground_contact_z,
            time=float(data.time),
            contact_point_world=contact_point,
            contact_normal_world=contact_normal,
            contact_force=contact_force,
            contact_confidence=float(
                0.1
                if contact_valid and contact_force <= 0.0
                else 1.0
                - np.exp(
                    -contact_force
                    / max(0.25 * self._config.mass * self._config.gravity, 1e-6)
                )
            ),
            contact_valid=contact_valid,
        )

    def fill_inertial_params(self, config: ControllerConfig) -> None:
        """Read composite rigid-body mass/inertia from MuJoCo."""
        body_ids: list[int] = []
        for candidate in range(self._model.nbody):
            current = candidate
            articulated = False
            while current != self._body_id and current > 0:
                if int(self._model.body_jntnum[current]) > 0:
                    articulated = True
                    break
                current = int(self._model.body_parentid[current])
            if current == self._body_id and not articulated:
                body_ids.append(candidate)

        scratch = mujoco.MjData(self._model)
        if self._model.nq >= 7:
            scratch.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self._model, scratch)

        masses = np.asarray([self._model.body_mass[i] for i in body_ids], dtype=float)
        mass = float(np.sum(masses))
        if mass <= 0.0:
            mass = float(self._model.body_subtreemass[self._body_id])
        config.mass = mass

        centers = np.asarray([scratch.xipos[i] for i in body_ids], dtype=float)
        center = np.sum(centers * masses[:, None], axis=0) / max(mass, 1e-9)
        inertia_world = np.zeros((3, 3), dtype=float)
        for body_id, body_mass, body_center in zip(body_ids, masses, centers):
            inertial_rotation = scratch.ximat[body_id].reshape(3, 3)
            body_inertia = inertial_rotation @ np.diag(
                self._model.body_inertia[body_id]
            ) @ inertial_rotation.T
            offset = body_center - center
            inertia_world += body_inertia + body_mass * (
                float(np.dot(offset, offset)) * np.eye(3) - np.outer(offset, offset)
            )

        root_rotation = scratch.xmat[self._body_id].reshape(3, 3)
        inertia = root_rotation.T @ inertia_world @ root_rotation
        if np.allclose(inertia, 0.0):
            inertia = np.diag([0.043, 0.043, 0.043])
        config.inertia = 0.5 * (inertia + inertia.T)
