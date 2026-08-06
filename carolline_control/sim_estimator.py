"""Shared helpers for oracle vs sensor-only vs SLAM state estimation in simulations."""

from __future__ import annotations

import argparse

import mujoco

from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.types import ControllerConfig


def add_sensor_only_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--sensor-only`` on an argparse parser."""
    parser.add_argument(
        "--sensor-only",
        action="store_true",
        help=(
            "Estimate pose from IMU sensors only (body_quat, body_gyro, body_vel, "
            "body_linacc) with dead-reckoned position; do not read qpos during control"
        ),
    )


def add_slam_nav_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--slam-nav`` on an argparse parser."""
    parser.add_argument(
        "--slam-nav",
        action="store_true",
        help=(
            "Use SLAM odometry pose (mapping + scan matching) instead of oracle qpos "
            "for navigation and control"
        ),
    )


def build_estimator(
    model: mujoco.MjModel,
    config: ControllerConfig,
    *,
    sensor_only: bool = False,
    slam_odom: bool = False,
) -> StateEstimator:
    """Construct a StateEstimator and populate mass/inertia from the model."""
    estimator = StateEstimator(
        model,
        config,
        sensor_only=sensor_only,
        slam_odom=slam_odom,
    )
    estimator.fill_inertial_params(config)
    return estimator


def seed_estimator_from_sim(estimator: StateEstimator, data: mujoco.MjData) -> None:
    """Seed dead-reckoning position after the sim pose is set (known spawn point)."""
    if estimator.sensor_only:
        estimator.reset(data.qpos[:3].copy())
    if estimator.slam_odom:
        estimator.reset_slam_pose(data.qpos[:3].copy())


def print_estimator_mode(*, sensor_only: bool = False, slam_odom: bool = False) -> None:
    if slam_odom:
        print(
            "State estimator: SLAM ODOM (IMU + planar SLAM pose; qpos used only for eval error)"
        )
    elif sensor_only:
        print(
            "State estimator: SENSOR-ONLY (body_quat, body_gyro, body_vel, "
            "integrated position from known spawn)"
        )
    else:
        print("State estimator: ORACLE (qpos pose + IMU velocity/acceleration)")
