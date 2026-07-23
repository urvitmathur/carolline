"""
SO(3) utilities for geometric control.

Implements the operators used throughout the CAROLLINE stack and
Lee et al. (2010) geometric quadrotor control:

- hat / vee maps between R^3 and so(3)
- rotation error e_R (Eq. 8, Lee et al.)
- trace-based attitude error Psi (Eq. 6, Lee et al.)
- matrix logarithm on SO(3)
"""

from __future__ import annotations

import numpy as np


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix hat(v) in so(3).

    For x, y in R^3: hat(x) @ y = x x y.
    """
    x, y, z = v
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def vee(skew: np.ndarray) -> np.ndarray:
    """Inverse of the hat map: vee(hat(v)) = v."""
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion [w, x, y, z] to rotation matrix R_wb."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to unit quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def attitude_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """SO(3) attitude error e_R (Lee et al., Eq. 8).

    Uses the matrix logarithm so the error remains well-defined near 180 deg,
    which is required for pre-takeoff recovery on the cage.
    """
    return rot_log(Rd.T @ R)


def attitude_error_function(R: np.ndarray, Rd: np.ndarray) -> float:
    """Trace attitude error Psi(R, Rd) (Lee et al., Eq. 6).

    Psi = 2 - tr(Rd^T R) in [0, 4].
    """
    return 2.0 - float(np.trace(Rd.T @ R))


def omega_error(omega: np.ndarray, R: np.ndarray, Rd: np.ndarray, omega_d: np.ndarray) -> np.ndarray:
    """Angular velocity error e_Omega (Lee et al., Eq. 9).

    e_Omega = Omega - R^T Rd Omega_d
    """
    return omega - R.T @ Rd @ omega_d


def rot_log(R_err: np.ndarray) -> np.ndarray:
    """Matrix logarithm on SO(3): R_err = exp(hat(theta)), returns theta in R^3."""
    cos_theta = (np.trace(R_err) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-4:
        # Near pi the usual skew/sin(theta) expression is singular. Recover
        # the axis from R + I using the largest diagonal component, which is
        # deterministic at exactly 180 degrees (paper Eq. 19 requirement).
        diag = np.maximum((np.diag(R_err) + 1.0) * 0.5, 0.0)
        index = int(np.argmax(diag))
        axis = np.zeros(3)
        axis[index] = np.sqrt(diag[index])
        if axis[index] > 1e-7:
            for other in range(3):
                if other != index:
                    axis[other] = (
                        R_err[index, other] + R_err[other, index]
                    ) / (4.0 * axis[index])
        if np.linalg.norm(axis) < 1e-7:
            eigenvalues, vectors = np.linalg.eig(R_err)
            eigen_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
            axis = np.real(vectors[:, eigen_index])
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)

        # Use the small residual skew to select a consistent sign when it is
        # observable. At exact pi either sign is an equally short solution.
        skew_hint = vee(R_err - R_err.T)
        if float(np.dot(axis, skew_hint)) < 0.0:
            axis *= -1.0
        return axis * theta
    skew = (R_err - R_err.T) / (2.0 * np.sin(theta))
    return vee(skew) * theta


def rot_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation mapping unit vector a to unit vector b."""
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    v = np.cross(a_n, b_n)
    c = float(np.dot(a_n, b_n))
    if np.linalg.norm(v) < 1e-8:
        if c > 0.0:
            return np.eye(3)
        # 180-degree rotation about any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a_n[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a_n * np.dot(a_n, axis)
        axis = axis / np.linalg.norm(axis)
        return np.eye(3) + 2.0 * hat(axis) @ hat(axis)
    vx = hat(v)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (np.linalg.norm(v) ** 2))


def desired_rotation_from_thrust_direction(b3_des: np.ndarray, yaw_des: float) -> np.ndarray:
    """Construct Rd in SO(3) from desired thrust axis b3 and yaw (Lee et al., Eq. 22)."""
    b3 = b3_des / np.linalg.norm(b3_des)
    b1_a = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    b1 = b1_a - b3 * np.dot(b3, b1_a)
    if np.linalg.norm(b1) < 1e-6:
        b1 = np.array([1.0, 0.0, 0.0]) - b3 * b3[0]
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(b3, b1)
    return np.column_stack((b1, b2, b3))


def body_z_world(R: np.ndarray) -> np.ndarray:
    """Body z-axis expressed in the world frame."""
    return R[:, 2]


def rot_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """ZYX yaw-pitch-roll [rad] for rotation matrix R (body -> world)."""
    R = np.asarray(R, dtype=float)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def rolling_omega_world(
    v_des_planar: np.ndarray,
    r_contact_to_center: np.ndarray,
    yaw_rate: float = 0.0,
) -> np.ndarray:
    """World angular velocity for no-slip rolling with desired planar COM velocity.

    For a sphere/cage with contact vector r (contact point -> center), kinematics give
    v_com = omega x r. The spin component about r is left free for yaw_rate.
    """
    r = np.asarray(r_contact_to_center, dtype=float)
    r2 = float(np.dot(r, r))
    if r2 < 1e-10:
        return np.zeros(3)

    v = np.asarray(v_des_planar, dtype=float).copy()
    v[2] = 0.0
    v_t = v - r * (float(np.dot(v, r)) / r2)
    omega = np.cross(r, v_t) / r2
    r_hat = r / np.sqrt(r2)
    omega += yaw_rate * r_hat
    return omega


def clamp_thrust_direction(b3: np.ndarray, min_z: float = 0.25) -> np.ndarray:
    """Keep desired thrust in the upper hemisphere to avoid flip maneuvers."""
    b3 = np.asarray(b3, dtype=float)
    norm = np.linalg.norm(b3)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    b3 = b3 / norm
    if b3[2] >= min_z:
        return b3
    tilt = np.array([b3[0], b3[1], min_z], dtype=float)
    return tilt / np.linalg.norm(tilt)
