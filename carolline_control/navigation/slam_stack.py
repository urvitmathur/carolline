"""Integrated SLAM stack: odometry, mapping, localization."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

from carolline_control.navigation.global_planner import GlobalPlanner
from carolline_control.navigation.localization import ScanMatcher
from carolline_control.navigation.mapping import OccupancyGridMapper
from carolline_control.navigation.odometry import PoseEKF, yaw_from_quaternion
from carolline_control.navigation.perception import RangePerception
from carolline_control.utils.so3 import quat_to_rot
from carolline_control.utils.types import RobotState

CAROLLINE_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
DEFAULT_SLAM_CONFIG = CAROLLINE_ROOT / "navigation" / "slam_config.yaml"


@dataclass
class SlamConfig:
    map_width_m: float = 36.0
    map_height_m: float = 36.0
    map_resolution: float = 0.05
    map_center_xy: tuple[float, float] | None = None
    inflation_radius: float = 0.42
    soft_cost_radius: float = 0.95
    cost_weight: float = 2.5
    unknown_cost: float = 1.5
    log_odds_hit: float = 0.85
    log_odds_miss: float = -0.4
    free_log_odds: float = -0.25
    occupied_log_odds: float = 0.85
    min_hit_range: float = 0.22
    search_xy: float = 0.35
    search_yaw_deg: float = 12.0
    xy_step: float = 0.07
    yaw_step_deg: float = 3.0
    update_interval_steps: int = 8
    process_noise: float = 0.02
    initial_covariance: float = 0.05
    replan_interval_s: float = 2.5
    min_waypoint_spacing: float = 0.8


def load_slam_config(path=None) -> tuple[SlamConfig, dict]:
    cfg_path = path or DEFAULT_SLAM_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    m = raw.get("map", {})
    loc = raw.get("localization", {})
    odom = raw.get("odometry", {})
    plan = raw.get("planning", {})
    config = SlamConfig(
        map_width_m=float(m.get("width_m", 36.0)),
        map_height_m=float(m.get("height_m", 36.0)),
        map_resolution=float(m.get("resolution", 0.05)),
        map_center_xy=tuple(m["center_xy"]) if m.get("center_xy") is not None else None,
        inflation_radius=float(m.get("inflation_radius", 0.42)),
        soft_cost_radius=float(m.get("soft_cost_radius", m.get("inflation_radius", 0.42) * 2.2)),
        cost_weight=float(m.get("cost_weight", 2.5)),
        unknown_cost=float(m.get("unknown_cost", 1.5)),
        log_odds_hit=float(m.get("log_odds_hit", 0.85)),
        log_odds_miss=float(m.get("log_odds_miss", -0.4)),
        free_log_odds=float(m.get("free_log_odds", -0.25)),
        occupied_log_odds=float(m.get("occupied_log_odds", 0.85)),
        min_hit_range=float(m.get("min_hit_range", 0.22)),
        search_xy=float(loc.get("search_xy", 0.35)),
        search_yaw_deg=float(loc.get("search_yaw_deg", 12.0)),
        xy_step=float(loc.get("xy_step", 0.07)),
        yaw_step_deg=float(loc.get("yaw_step_deg", 3.0)),
        update_interval_steps=int(loc.get("update_interval_steps", 8)),
        process_noise=float(odom.get("process_noise", 0.02)),
        initial_covariance=float(odom.get("initial_covariance", 0.05)),
        replan_interval_s=float(plan.get("replan_interval_s", 2.5)),
        min_waypoint_spacing=float(plan.get("min_waypoint_spacing", 0.8)),
    )
    return config, raw


class SlamNavigator:
    """Orchestrate mapping, localization, and global planning."""

    def __init__(
        self,
        model: mujoco.MjModel,
        perception: RangePerception,
        slam_config: SlamConfig | None = None,
        *,
        map_origin_xy: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.perception = perception
        self.config = slam_config or SlamConfig()
        origin = np.asarray(map_origin_xy if map_origin_xy is not None else [-18.0, -18.0], dtype=float)
        self.mapper = OccupancyGridMapper(
            origin_xy=origin,
            width_m=self.config.map_width_m,
            height_m=self.config.map_height_m,
            resolution=self.config.map_resolution,
            log_odds_hit=self.config.log_odds_hit,
            log_odds_miss=self.config.log_odds_miss,
            inflation_radius=self.config.inflation_radius,
            soft_cost_radius=self.config.soft_cost_radius,
            cost_weight=self.config.cost_weight,
            unknown_cost=self.config.unknown_cost,
            free_log_odds=self.config.free_log_odds,
            occupied_log_odds=self.config.occupied_log_odds,
            min_hit_range=self.config.min_hit_range,
        )
        self.pose = PoseEKF(
            process_noise=self.config.process_noise,
            initial_covariance=self.config.initial_covariance,
        )
        self.matcher = ScanMatcher(
            search_xy=self.config.search_xy,
            search_yaw_deg=self.config.search_yaw_deg,
            xy_step=self.config.xy_step,
            yaw_step_deg=self.config.yaw_step_deg,
        )
        self.planner = GlobalPlanner(self.mapper)
        self._step = 0
        self._goal_xy: np.ndarray | None = None
        self._last_plan_time = -1.0
        self._planned_waypoints: list[np.ndarray] = []
        self.last_oracle_position_error = 0.0
        self._map_center_xy = (
            np.asarray(self.config.map_center_xy, dtype=float)
            if self.config.map_center_xy is not None
            else None
        )

    def _map_origin_for_center(self, center_xy: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(center_xy[0]) - 0.5 * self.config.map_width_m,
                float(center_xy[1]) - 0.5 * self.config.map_height_m,
            ],
            dtype=float,
        )

    def reset(self, spawn_xy: np.ndarray, spawn_z: float, quaternion: np.ndarray) -> None:
        yaw = yaw_from_quaternion(quaternion)
        self.pose.reset(float(spawn_xy[0]), float(spawn_xy[1]), yaw)
        if self._map_center_xy is not None:
            origin = self._map_origin_for_center(self._map_center_xy)
        else:
            origin = self._map_origin_for_center(np.asarray(spawn_xy[:2], dtype=float))
        self.mapper.reset(origin_xy=origin)
        self._step = 0
        self._last_plan_time = -1.0
        self._planned_waypoints = []
        _ = spawn_z

    def set_goal(self, goal_xy: np.ndarray) -> None:
        self._goal_xy = np.asarray(goal_xy[:2], dtype=float).copy()

    def plan_to_goal(
        self,
        start_xy: np.ndarray | None = None,
        *,
        conservative: bool = False,
    ) -> list[np.ndarray]:
        if self._goal_xy is None:
            return []
        start = self.pose.position_xy() if start_xy is None else np.asarray(start_xy[:2], dtype=float)
        self._planned_waypoints = self.planner.plan(
            start,
            self._goal_xy,
            min_waypoint_spacing=self.config.min_waypoint_spacing,
            conservative=conservative,
        )
        return [w.copy() for w in self._planned_waypoints]

    @property
    def waypoints(self) -> list[np.ndarray]:
        return [w.copy() for w in self._planned_waypoints]

    def should_replan(self, t: float, *, force: bool = False) -> bool:
        if force or not self._planned_waypoints:
            return True
        return (t - self._last_plan_time) >= self.config.replan_interval_s

    def mark_replanned(self, t: float) -> None:
        self._last_plan_time = float(t)

    def update(
        self,
        data: mujoco.MjData,
        sensor_state: RobotState,
        *,
        oracle_position: np.ndarray | None = None,
        mapping_state: RobotState | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance SLAM and return (position_xyz, quaternion) for control."""
        self.pose.predict(sensor_state)
        self._step += 1

        site_ids = self.perception._site_ids
        site_is_up = self.perception._sensor_is_up
        ranges = [
            self.perception._read_range(data, sid) for sid in self.perception._sensor_ids
        ]
        map_state = mapping_state if mapping_state is not None else self._pose_as_robot_state(sensor_state)
        self.mapper.update_from_rays(
            map_state,
            site_ids=site_ids,
            site_is_up=site_is_up,
            ranges=ranges,
            model=self.model,
            data=data,
            max_range=self.perception.max_range,
        )

        if self._step % max(1, self.config.update_interval_steps) == 0:
            scan_world = self.mapper.extract_scan_points_world(
                self._pose_as_robot_state(sensor_state),
                site_ids=site_ids,
                site_is_up=site_is_up,
                ranges=ranges,
                model=self.model,
                data=data,
                max_range=self.perception.max_range,
            )
            scan_body = ScanMatcher.body_frame_points(
                scan_world, self.pose.x, self.pose.y, self.pose.yaw
            )
            match = self.matcher.match(self.mapper, scan_body, self.pose)
            if match is not None:
                measurement, meas_cov = match
                self.pose.update(measurement, meas_cov)

        position = self.pose.position_3d(float(sensor_state.position[2]))
        quaternion = self.pose.quaternion()
        if oracle_position is not None:
            self.last_oracle_position_error = float(
                np.linalg.norm(position[:2] - np.asarray(oracle_position[:2], dtype=float))
            )
        return position, quaternion

    def _pose_as_robot_state(self, sensor_state: RobotState) -> RobotState:
        position = self.pose.position_3d(float(sensor_state.position[2]))
        quaternion = self.pose.quaternion()
        rotation = quat_to_rot(quaternion)
        return RobotState(
            position=position,
            velocity=sensor_state.velocity,
            quaternion=quaternion,
            rotation=rotation,
            omega_body=sensor_state.omega_body,
            omega_world=sensor_state.omega_world,
            accel_body=sensor_state.accel_body,
            on_ground=sensor_state.on_ground,
            ground_contact_z=sensor_state.ground_contact_z,
            time=sensor_state.time,
            contact_point_world=sensor_state.contact_point_world,
            contact_normal_world=sensor_state.contact_normal_world,
            contact_force=sensor_state.contact_force,
            contact_confidence=sensor_state.contact_confidence,
            contact_valid=sensor_state.contact_valid,
        )
