"""Rolling odometry and planar pose EKF for SLAM navigation."""

from __future__ import annotations

import numpy as np

from carolline_control.utils.so3 import quat_to_rot
from carolline_control.utils.types import RobotState


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    """Extract yaw [rad] from a unit quaternion (world Z rotation)."""
    quat = np.asarray(quaternion, dtype=float)
    rotation = quat_to_rot(quat)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([float(np.cos(half)), 0.0, 0.0, float(np.sin(half))], dtype=float)


class RollingOdometry:
    """Integrate planar motion from IMU velocity while rolling on the ground."""

    def __init__(self, *, process_noise: float = 0.02) -> None:
        self._process_noise = float(process_noise)
        self._last_time: float | None = None

    def predict(
        self,
        state: RobotState,
        *,
        x: float,
        y: float,
        yaw: float,
        covariance: np.ndarray,
    ) -> tuple[float, float, float, np.ndarray]:
        """Advance pose using body/world velocity when grounded."""
        t = float(state.time)
        if self._last_time is None:
            self._last_time = t
            return x, y, yaw, covariance

        dt = t - self._last_time
        self._last_time = t
        if dt <= 0.0:
            return x, y, yaw, covariance

        if state.on_ground and state.contact_valid:
            normal = state.contact_normal_world
            normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
            velocity = np.asarray(state.velocity, dtype=float)
            v_planar = velocity - normal * float(np.dot(velocity, normal))
            x += float(v_planar[0]) * dt
            y += float(v_planar[1]) * dt
        else:
            velocity = np.asarray(state.velocity, dtype=float)
            x += float(velocity[0]) * dt
            y += float(velocity[1]) * dt

        measured_yaw = yaw_from_quaternion(state.quaternion)
        yaw = measured_yaw

        q = self._process_noise
        motion = max(dt, 1e-4) * (abs(float(np.linalg.norm(state.velocity[:2]))) + 0.05)
        covariance = covariance + np.diag([q * motion, q * motion, q * 0.5 * motion])
        return x, y, yaw, covariance

    def reset(self, x: float, y: float, yaw: float) -> None:
        self._last_time = None
        self._seed = (float(x), float(y), float(yaw))


class PoseEKF:
    """Planar pose filter: state = [x, y, yaw]."""

    def __init__(
        self,
        *,
        initial_covariance: float = 0.05,
        process_noise: float = 0.02,
    ) -> None:
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.covariance = np.eye(3, dtype=float) * float(initial_covariance)
        self._odom = RollingOdometry(process_noise=process_noise)

    def reset(self, x: float, y: float, yaw: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.covariance = np.eye(3, dtype=float) * 0.05
        self._odom.reset(self.x, self.y, self.yaw)

    def predict(self, state: RobotState) -> None:
        self.x, self.y, self.yaw, self.covariance = self._odom.predict(
            state,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            covariance=self.covariance,
        )

    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray) -> None:
        """Kalman update with measurement [x, y, yaw]."""
        z = np.asarray(measurement[:3], dtype=float)
        r = np.asarray(measurement_cov, dtype=float)
        if r.shape != (3, 3):
            r = np.diag(np.asarray(r[:3], dtype=float))

        h = np.eye(3, dtype=float)
        y_innov = z - h @ np.array([self.x, self.y, self.yaw], dtype=float)
        y_innov[2] = _wrap_angle(y_innov[2])
        s = h @ self.covariance @ h.T + r
        k = self.covariance @ h.T @ np.linalg.inv(s)
        state_vec = np.array([self.x, self.y, self.yaw], dtype=float) + k @ y_innov
        self.x = float(state_vec[0])
        self.y = float(state_vec[1])
        self.yaw = _wrap_angle(float(state_vec[2]))
        self.covariance = (np.eye(3) - k @ h) @ self.covariance

    def position_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def position_3d(self, z: float) -> np.ndarray:
        return np.array([self.x, self.y, float(z)], dtype=float)

    def quaternion(self) -> np.ndarray:
        return quaternion_from_yaw(self.yaw)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
