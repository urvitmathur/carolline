"""Tests for SLAM navigation modules."""

from __future__ import annotations

import numpy as np

from carolline_control.navigation.global_planner import GlobalPlanner
from carolline_control.navigation.mapping import OccupancyGridMapper
from carolline_control.navigation.odometry import PoseEKF, quaternion_from_yaw, yaw_from_quaternion


def test_pose_ekf_reset_and_quaternion():
    ekf = PoseEKF()
    ekf.reset(1.0, 2.0, 0.5)
    assert abs(ekf.x - 1.0) < 1e-9
    assert abs(ekf.y - 2.0) < 1e-9
    quat = quaternion_from_yaw(0.5)
    assert abs(yaw_from_quaternion(quat) - 0.5) < 1e-6


def test_occupancy_grid_mark_occupied():
    mapper = OccupancyGridMapper(origin_xy=np.array([-5.0, -5.0]), width_m=10.0, height_m=10.0, resolution=0.5)
    gx, gy = mapper.world_to_grid(0.0, 0.0)
    mapper._update_cell(gx, gy, mapper._log_hit)
    mapper.invalidate_inflation_cache()
    assert mapper.is_occupied(gx, gy, use_inflated=False)


def test_global_planner_finds_path():
    mapper = OccupancyGridMapper(
        origin_xy=np.array([-5.0, -5.0]),
        width_m=10.0,
        height_m=10.0,
        resolution=0.5,
        inflation_radius=0.2,
    )
    wall_gx = mapper.nx // 2
    gap_half = 4
    for gy in range(mapper.ny):
        if abs(gy - mapper.ny // 2) <= gap_half:
            continue
        mapper._update_cell(wall_gx, gy, mapper._log_hit)
    mapper.invalidate_inflation_cache()
    planner = GlobalPlanner(mapper)
    path = planner.plan(np.array([-4.0, 0.0]), np.array([4.0, 0.0]))
    assert len(path) >= 2
    assert float(np.linalg.norm(path[-1] - np.array([4.0, 0.0]))) < 1.5


def test_global_planner_open_field():
    mapper = OccupancyGridMapper(origin_xy=np.array([-5.0, -5.0]), width_m=10.0, height_m=10.0, resolution=0.5)
    for gx in range(mapper.nx):
        for gy in range(mapper.ny):
            mapper._update_cell(gx, gy, mapper._log_miss)
    mapper.invalidate_inflation_cache()
    planner = GlobalPlanner(mapper)
    path = planner.plan(np.array([-4.0, 0.0]), np.array([4.0, 0.0]))
    assert len(path) >= 1
    assert float(np.linalg.norm(path[-1] - np.array([4.0, 0.0]))) < 0.5
