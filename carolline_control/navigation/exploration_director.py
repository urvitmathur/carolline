"""Mission director for SLAM exploration in unknown environments."""

from __future__ import annotations

import enum

import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.navigation.perception import ObstacleScan, filter_scan_for_exploration
from carolline_control.navigation.slam_stack import SlamNavigator
from carolline_control.navigation.waypoint_follower import WaypointFollower
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode


class ExplorationPhase(enum.Enum):
    MAP_AND_ROLL = "MAP_AND_ROLL"
    REPLAN = "REPLAN"
    BLOCKED_RECOVER = "BLOCKED_RECOVER"
    TAKEOFF_OVER = "TAKEOFF_OVER"
    FLY_OVER = "FLY_OVER"
    LAND_BEYOND = "LAND_BEYOND"
    GOAL_REACHED = "GOAL_REACHED"


class ExplorationDirector:
    """Roll toward SLAM-planned waypoints; fly over obstacles when blocked."""

    def __init__(
        self,
        controller: CarollineController,
        follower: WaypointFollower,
        navigator: SlamNavigator,
        *,
        goal_xy: np.ndarray,
        terrain_z_at,
        fly_clearance: float = 0.90,
        blocked_confirm_time: float = 0.35,
        hover_height: float = 1.05,
        cage_radius: float = 0.40,
        max_obstacle_height: float = 1.0,
        max_flight_height: float = 1.85,
    ) -> None:
        self.ctrl = controller
        self.follower = follower
        self.navigator = navigator
        self.goal_xy = np.asarray(goal_xy[:2], dtype=float)
        self.terrain_z_at = terrain_z_at
        self.fly_clearance = float(fly_clearance)
        self.blocked_confirm_time = float(blocked_confirm_time)
        self.hover_height = float(hover_height)
        self.cage_radius = float(cage_radius)
        self.max_obstacle_height = float(max_obstacle_height)
        self.max_flight_height = float(max_flight_height)
        self.phase = ExplorationPhase.MAP_AND_ROLL
        self._phase_t0 = 0.0
        self._blocked_timer = 0.0
        self._land_hold = 0.0
        self._fly_land_xy: np.ndarray | None = None
        self._pending_fly_wps: list[list[float]] | None = None
        self._last_roll_dist: float | None = None
        self._last_progress_t = 0.0
        self._cfg = controller.config
        self._arena_half_x = 14.0
        self._arena_half_y = 10.0

    def begin(self, t: float) -> None:
        waypoints = self.navigator.plan_to_goal()
        self.follower.set_waypoints(waypoints if waypoints else [self.goal_xy.copy()])
        self.navigator.mark_replanned(t)
        self._enter(ExplorationPhase.MAP_AND_ROLL, t, "Begin SLAM rolling toward goal")

    def current_target_xy(self) -> np.ndarray:
        if self._fly_land_xy is not None and self.phase in (
            ExplorationPhase.FLY_OVER,
            ExplorationPhase.LAND_BEYOND,
        ):
            return self._fly_land_xy.copy()
        return self.follower.current_target_xy()

    def _enter(self, phase: ExplorationPhase, t: float, msg: str) -> None:
        self.phase = phase
        self._phase_t0 = t
        self._land_hold = 0.0
        if phase != ExplorationPhase.MAP_AND_ROLL:
            self._blocked_timer = 0.0
        print(f"t={t:6.2f}s  SLAM -> {phase.value}  |  {msg}")

    def _set_roll_target(self, xy: np.ndarray) -> None:
        self._cfg.roll_target = np.asarray(xy[:2], dtype=float).copy()

    def _set_ground_context(self, xy: np.ndarray) -> None:
        ground_z = float(self.terrain_z_at(float(xy[0]), float(xy[1])))
        self._cfg.landing_height = ground_z + self.cage_radius
        self._cfg.spawn_xy = np.asarray(xy[:2], dtype=float).copy()

    def _set_flight_waypoints(self, waypoints: list[list[float]], hover_z: float) -> None:
        self._cfg.hover_height = hover_z
        wps = [np.array(w, dtype=float) for w in waypoints]
        self.ctrl.planner._mission_waypoints = [w.copy() for w in wps]
        self.ctrl.planner._waypoints = [w.copy() for w in wps]
        self.ctrl.planner.reset()

    def mission_scan(self, scan: ObstacleScan, state) -> ObstacleScan:
        return filter_scan_for_exploration(
            scan,
            state,
            arena_half_x=self._arena_half_x,
            arena_half_y=self._arena_half_y,
        )

    def _rolling_stuck(self, state, t: float, target_xy: np.ndarray) -> bool:
        dist = float(np.linalg.norm(state.position[:2] - target_xy))
        if dist < 0.7:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False
        if self._last_roll_dist is None:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False
        if self._last_roll_dist - dist >= 0.15:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False
        slow = float(np.linalg.norm(state.velocity[:2])) < 0.25
        return slow and (t - self._last_progress_t) >= 2.5

    def _near_xy(self, state, xy: np.ndarray, tol: float = 0.55) -> bool:
        return float(np.linalg.norm(state.position[:2] - xy)) < tol

    def _ground_settled(self, state) -> bool:
        return (
            state.on_ground
            and float(np.linalg.norm(state.velocity)) < 0.12
            and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold
        )

    def _goal_reached(self, state) -> bool:
        return float(np.linalg.norm(state.position[:2] - self.goal_xy)) < 1.2

    def _replan_if_needed(self, state, t: float, *, force: bool = False) -> None:
        if not force and not self.navigator.should_replan(t):
            return
        if not force and len(self.follower.waypoints) > 1:
            return
        waypoints = self.navigator.plan_to_goal(state.position[:2])
        if waypoints:
            self.follower.set_waypoints(waypoints)
            self.navigator.mark_replanned(t)
            self._blocked_timer = 0.0
            if force and len(waypoints) > 1:
                print(f"t={t:6.2f}s  SLAM replanned {len(waypoints)} waypoints")

    def _cruise_altitude(self, state, scan: ObstacleScan) -> float:
        """Ground-relative cruise height — capped, never uses open-sky range."""
        ground = float(self.terrain_z_at(float(state.position[0]), float(state.position[1])))
        # Obstacle height hint from forward range (clamped); ignore upward max-range (open sky).
        forward_hint = float(np.clip(scan.forward_min_m, 0.2, 3.5))
        obstacle_top = min(self.max_obstacle_height, max(0.45, forward_hint * 0.55))
        clearance = self.fly_clearance
        if scan.upward_clear_m < self.ctrl.config.cage_radius + 2.5:
            clearance = max(clearance, scan.upward_clear_m * 0.25)
        cruise_z = ground + self.cage_radius + obstacle_top + clearance
        return float(min(cruise_z, self.max_flight_height))

    def _fly_over_waypoints(self, state, scan: ObstacleScan) -> list[list[float]]:
        pos = state.position.copy()
        forward = state.rotation[:, 0]
        forward[2] = 0.0
        norm = float(np.linalg.norm(forward[:2]))
        if norm < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward[:2] /= norm

        ground = float(self.terrain_z_at(float(pos[0]), float(pos[1])))
        ground_center = ground + self.cage_radius
        cruise_z = self._cruise_altitude(state, scan)
        hop_pre = 0.55
        hop_over = 1.6
        hop_post = 2.8

        pre = pos + forward * hop_pre
        pre[2] = min(cruise_z - 0.15, ground_center + 0.35)
        over = pos + forward * hop_over
        over[2] = cruise_z
        post = pos + forward * hop_post
        post[2] = ground_center + 0.12
        return [pre.tolist(), over.tolist(), post.tolist()]

    def _prepare_fly_over(self, state, scan: ObstacleScan) -> None:
        wps = self._fly_over_waypoints(state, scan)
        hover_z = min(max(float(w[2]) for w in wps), self.max_flight_height)
        self._pending_fly_wps = wps
        self._cfg.hover_height = hover_z
        ground = float(self.terrain_z_at(float(state.position[0]), float(state.position[1])))
        self._cfg.takeoff_height = min(hover_z - 0.35, ground + self.cage_radius + 0.25)
        self._fly_land_xy = np.array(wps[-1][:2], dtype=float).copy()

    def _launch_fly_over(self, state, t: float) -> None:
        if self._pending_fly_wps is None:
            return
        wps = self._pending_fly_wps
        hover_z = min(max(float(w[2]) for w in wps), self.max_flight_height)
        self._set_flight_waypoints(wps, hover_z=hover_z)
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        self.ctrl.mode_manager.request_flight()
        self.follower.enabled = False
        self._pending_fly_wps = None
        self._enter(ExplorationPhase.FLY_OVER, t, f"Fly over obstacle at z={hover_z:.2f} m")

    def update(
        self,
        state,
        mode: ControlMode,
        t: float,
        dt: float,
        scan: ObstacleScan,
    ) -> bool:
        mm = self.ctrl.mode_manager
        mission_scan = self.mission_scan(scan, state)

        if self.phase == ExplorationPhase.MAP_AND_ROLL:
            if mode != ControlMode.ROLLING:
                mm.request_roll()
            self._set_ground_context(state.position[:2])
            self._set_roll_target(self.follower.current_target_xy())
            self.follower.enabled = True
            self.follower.respect_blocked_scan = True

            if self._goal_reached(state) and float(np.linalg.norm(state.velocity[:2])) < 0.35:
                mm.request_idle()
                self._enter(ExplorationPhase.GOAL_REACHED, t, "Goal reached via SLAM navigation")
                return False

            if self.follower.near_current_target(state):
                if self.follower.advance():
                    self._set_roll_target(self.follower.current_target_xy())
                elif self._goal_reached(state):
                    mm.request_idle()
                    self._enter(ExplorationPhase.GOAL_REACHED, t, "Goal reached")
                    return False

            target_xy = self.follower.current_target_xy()
            if self._rolling_stuck(state, t, target_xy) and mission_scan.flyable:
                self._prepare_fly_over(state, mission_scan)
                mm.request_takeoff()
                self._enter(ExplorationPhase.BLOCKED_RECOVER, t, "Stuck — preparing fly-over")

            if mission_scan.blocked:
                self._blocked_timer += dt
            else:
                self._blocked_timer = 0.0

            if self._blocked_timer >= self.blocked_confirm_time:
                if mission_scan.flyable:
                    self._prepare_fly_over(state, mission_scan)
                    mm.request_takeoff()
                    self._enter(ExplorationPhase.BLOCKED_RECOVER, t, "Blocked — preparing fly-over")
                else:
                    self._replan_if_needed(state, t, force=True)

            if (
                self.navigator.should_replan(t)
                and len(self.follower.waypoints) <= 1
                and not mission_scan.blocked
            ):
                self._replan_if_needed(state, t)

        elif self.phase == ExplorationPhase.BLOCKED_RECOVER:
            self.follower.enabled = False
            hover_z = float(self._cfg.hover_height)
            alt_ok = abs(float(state.position[2]) - hover_z) < self._cfg.hover_altitude_tolerance + 0.12
            if mode in (ControlMode.HOVER, ControlMode.FLIGHT) and alt_ok:
                self._launch_fly_over(state, t)
            elif mode == ControlMode.TAKEOFF and float(state.position[2]) >= hover_z - 0.15:
                self._launch_fly_over(state, t)
            elif t - self._phase_t0 > 10.0 and float(state.position[2]) > self.cage_radius + 0.5:
                self._launch_fly_over(state, t)

        elif self.phase == ExplorationPhase.FLY_OVER:
            self.follower.enabled = False
            if self.ctrl.planner.mission_complete:
                mm.request_landing()
                self._enter(ExplorationPhase.LAND_BEYOND, t, "Descending past obstacle")

        elif self.phase == ExplorationPhase.LAND_BEYOND:
            self.follower.enabled = False
            if mode == ControlMode.ROLLING and self._ground_settled(state):
                self._land_hold += dt
                if self._land_hold >= 0.35:
                    self._fly_land_xy = None
                    self._replan_if_needed(state, t, force=True)
                    self._enter(ExplorationPhase.MAP_AND_ROLL, t, "Resumed rolling after fly-over")
            elif t - self._phase_t0 > 12.0 and self._ground_settled(state):
                self._fly_land_xy = None
                mm.request_roll()
                self._replan_if_needed(state, t, force=True)
                self._enter(ExplorationPhase.MAP_AND_ROLL, t, "Resume rolling (timeout landing)")

        elif self.phase == ExplorationPhase.GOAL_REACHED:
            return False

        return True
