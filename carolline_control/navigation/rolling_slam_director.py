"""Rolling-only SLAM mission director (no flight)."""

from __future__ import annotations

import enum

import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.navigation.perception import ObstacleScan
from carolline_control.navigation.slam_stack import SlamNavigator
from carolline_control.navigation.waypoint_follower import WaypointFollower
from carolline_control.utils.types import ControlMode


class RollingSlamPhase(enum.Enum):
    MAP_AND_ROLL = "MAP_AND_ROLL"
    REPLAN = "REPLAN"
    GOAL_REACHED = "GOAL_REACHED"


class RollingSlamDirector:
    """Map with rangefinders, A* on the SLAM grid, lookahead path follow."""

    def __init__(
        self,
        controller: CarollineController,
        follower: WaypointFollower,
        navigator: SlamNavigator,
        *,
        goal_xy: np.ndarray,
        goal_radius: float = 0.40,
        stuck_time: float = 3.0,
        cage_radius: float = 0.40,
        replan_interval_s: float = 3.0,
    ) -> None:
        self.ctrl = controller
        self.follower = follower
        self.navigator = navigator
        self.goal_xy = np.asarray(goal_xy[:2], dtype=float)
        self.goal_radius = float(goal_radius)
        self.stuck_time = float(stuck_time)
        self.cage_radius = float(cage_radius)
        self.replan_interval_s = float(replan_interval_s)
        self.phase = RollingSlamPhase.MAP_AND_ROLL
        self._last_progress_t = 0.0
        self._last_progress_xy: np.ndarray | None = None
        self._last_replan_t = -10.0
        self._cfg = controller.config
        self._planned_route: list[np.ndarray] = []

    def _build_route(self, start_xy: np.ndarray) -> list[np.ndarray]:
        """A* on the live (seeded + LIDAR) occupancy grid."""
        start = np.asarray(start_xy[:2], dtype=float)
        waypoints = self.navigator.plan_to_goal(start)
        if not waypoints:
            return [self.goal_xy.copy()]
        # Ensure the path ends exactly on the goal marker.
        if float(np.linalg.norm(waypoints[-1] - self.goal_xy)) > 0.15:
            waypoints.append(self.goal_xy.copy())
        return waypoints

    def begin(self, t: float) -> None:
        start = np.array([self.navigator.pose.x, self.navigator.pose.y], dtype=float)
        waypoints = self._build_route(start)
        self._planned_route = [w.copy() for w in waypoints]
        self.follower.set_waypoints(waypoints)
        self.follower.index = 0
        self.navigator.mark_replanned(t)
        self._last_replan_t = t
        self._last_progress_t = t
        self._last_progress_xy = None
        print(
            f"t={t:6.2f}s  MAZE -> {RollingSlamPhase.MAP_AND_ROLL.value}  |  "
            f"SLAM A* plan ({len(waypoints)} wps), lookahead follow"
        )

    def current_target_xy(self) -> np.ndarray:
        return self.follower.current_target_xy()

    def planned_route_xy(self) -> np.ndarray | None:
        if self._planned_route:
            return np.asarray(self._planned_route, dtype=float)
        if self.follower.waypoints:
            return np.asarray(self.follower.waypoints, dtype=float)
        return None

    def _goal_reached(self, state) -> bool:
        return float(np.linalg.norm(state.position[:2] - self.goal_xy)) < self.goal_radius

    def _note_progress(self, state, t: float) -> None:
        pos = state.position[:2]
        if self._last_progress_xy is None:
            self._last_progress_xy = pos.copy()
            self._last_progress_t = t
            return
        if float(np.linalg.norm(pos - self._last_progress_xy)) >= 0.20:
            self._last_progress_xy = pos.copy()
            self._last_progress_t = t

    def _is_stuck(self, state, t: float) -> bool:
        slow = float(np.linalg.norm(state.velocity[:2])) < 0.18
        return slow and (t - self._last_progress_t) >= self.stuck_time

    def _replan(self, state, t: float, *, reason: str = "") -> None:
        if t - self._last_replan_t < 1.5:
            return
        waypoints = self._build_route(state.position[:2])
        if not waypoints:
            return
        self.follower.set_waypoints(waypoints)
        self.follower.snap_to_nearest(state, prefer_ahead=True, goal_xy=self.goal_xy)
        self._planned_route = [w.copy() for w in waypoints]
        self.navigator.mark_replanned(t)
        self._last_replan_t = t
        self._last_progress_t = t
        self._last_progress_xy = state.position[:2].copy()
        print(
            f"t={t:6.2f}s  MAZE -> REPLAN  |  {reason} -> {len(waypoints)} wps "
            f"(lookahead A*)"
        )

    def update(
        self,
        state,
        mode: ControlMode,
        t: float,
        dt: float,
        scan: ObstacleScan,
    ) -> bool:
        _ = dt
        _ = scan
        mm = self.ctrl.mode_manager
        if mode != ControlMode.ROLLING:
            mm.request_roll()

        # Steer toward lookahead carrot, not discrete waypoint only.
        self._cfg.roll_target = self.follower.current_target_xy()
        self.follower.enabled = True
        self.follower.respect_blocked_scan = False
        self._note_progress(state, t)

        dist_goal = float(np.linalg.norm(state.position[:2] - self.goal_xy))
        if dist_goal < 1.0:
            self._cfg.roll_target = self.goal_xy.copy()

        if self._goal_reached(state) and float(np.linalg.norm(state.velocity[:2])) < 0.70:
            mm.request_idle()
            self.phase = RollingSlamPhase.GOAL_REACHED
            print(
                f"t={t:6.2f}s  MAZE -> GOAL_REACHED  |  "
                f"Goal reached via SLAM A* (dist={dist_goal:.2f} m)"
            )
            return False

        # Advance discrete index when near the current polyline node.
        if self.follower.near_current_target(state):
            self.follower.advance()

        if self._is_stuck(state, t):
            self._replan(state, t, reason="stuck")
        elif t - self._last_replan_t >= self.replan_interval_s:
            # Periodic replan as the SLAM map fills in.
            self._replan(state, t, reason="map-update")

        return True
