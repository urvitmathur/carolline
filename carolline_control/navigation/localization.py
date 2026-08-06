"""Scan-to-map localization on an occupancy grid."""

from __future__ import annotations

import numpy as np

from carolline_control.navigation.mapping import OccupancyGridMapper
from carolline_control.navigation.odometry import PoseEKF, _wrap_angle


class ScanMatcher:
    """Coarse grid search to align rangefinder hits with the occupancy map."""

    def __init__(
        self,
        *,
        search_xy: float = 0.35,
        search_yaw_deg: float = 12.0,
        xy_step: float = 0.07,
        yaw_step_deg: float = 3.0,
        min_points: int = 2,
    ) -> None:
        self.search_xy = float(search_xy)
        self.search_yaw = float(np.radians(search_yaw_deg))
        self.xy_step = float(xy_step)
        self.yaw_step = float(np.radians(yaw_step_deg))
        self.min_points = int(min_points)

    def match(
        self,
        mapper: OccupancyGridMapper,
        scan_points_body: np.ndarray,
        pose: PoseEKF,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (measurement [x,y,yaw], covariance diag) or None if insufficient data."""
        if scan_points_body.shape[0] < self.min_points:
            return None

        best_score = -1.0
        best_pose = np.array([pose.x, pose.y, pose.yaw], dtype=float)
        prob = mapper.occupancy_probability()
        threshold = mapper.occupied_threshold()

        x_vals = np.arange(-self.search_xy, self.search_xy + 1e-9, self.xy_step)
        y_vals = np.arange(-self.search_xy, self.search_xy + 1e-9, self.xy_step)
        yaw_vals = np.arange(-self.search_yaw, self.search_yaw + 1e-9, self.yaw_step)

        cos_y = float(np.cos(pose.yaw))
        sin_y = float(np.sin(pose.yaw))
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
        world_points = (rot @ scan_points_body.T).T + np.array([pose.x, pose.y])

        for dx in x_vals:
            for dy in y_vals:
                for dyaw in yaw_vals:
                    trial_yaw = _wrap_angle(pose.yaw + dyaw)
                    c = float(np.cos(trial_yaw))
                    s = float(np.sin(trial_yaw))
                    r = np.array([[c, -s], [s, c]], dtype=float)
                    trial_points = (r @ scan_points_body.T).T + np.array(
                        [pose.x + dx, pose.y + dy]
                    )
                    score = 0.0
                    for pt in trial_points:
                        gx, gy = mapper.world_to_grid(float(pt[0]), float(pt[1]))
                        if not mapper.in_bounds(gx, gy):
                            continue
                        if prob[gy, gx] >= threshold:
                            score += 1.0
                    if score > best_score:
                        best_score = score
                        best_pose = np.array([pose.x + dx, pose.y + dy, trial_yaw], dtype=float)

        if best_score < self.min_points:
            return None

        confidence = min(1.0, best_score / max(scan_points_body.shape[0], 1))
        meas_cov = np.diag(
            [
                0.08 * (1.1 - confidence),
                0.08 * (1.1 - confidence),
                0.04 * (1.1 - confidence),
            ]
        )
        return best_pose, meas_cov

    @staticmethod
    def body_frame_points(
        scan_points_world: np.ndarray,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> np.ndarray:
        if scan_points_world.shape[0] == 0:
            return scan_points_world
        c = float(np.cos(pose_yaw))
        s = float(np.sin(pose_yaw))
        rot_inv = np.array([[c, s], [-s, c]], dtype=float)
        centered = scan_points_world - np.array([pose_x, pose_y])
        return (rot_inv @ centered.T).T
