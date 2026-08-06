"""Mission director for autonomous roll-fly-roll course navigation."""

from __future__ import annotations

import enum

import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.navigation.course_layout import CourseLayout, WallSpec
from carolline_control.navigation.perception import ObstacleScan, filter_scan_for_mission
from carolline_control.navigation.rolling_follower import TerrainRollingFollower
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode


class CoursePhase(enum.Enum):
    ROLL_TO_CHECKPOINT = "ROLL_TO_CHECKPOINT"
    ROLL_RECOVER = "ROLL_RECOVER"
    BLOCKED_RECOVER = "BLOCKED_RECOVER"
    TAKEOFF_OVER = "TAKEOFF_OVER"
    FLY_OVER = "FLY_OVER"
    LAND_BEYOND = "LAND_BEYOND"
    FLY_TERRAIN = "FLY_TERRAIN"
    LAND_TERRAIN = "LAND_TERRAIN"
    DONE = "DONE"


class HybridCourseDirector:
    """Sequence rolling, wall detection, fly-over hops, and resume rolling."""

    def __init__(
        self,
        controller: CarollineController,
        layout: CourseLayout,
        follower: TerrainRollingFollower,
        terrain_z_at,
    ) -> None:
        self.ctrl = controller
        self.layout = layout
        self.follower = follower
        self.terrain_z_at = terrain_z_at
        self.phase = CoursePhase.ROLL_TO_CHECKPOINT
        self._phase_t0 = 0.0
        self._blocked_timer = 0.0
        self._land_hold = 0.0
        self._active_wall: WallSpec | None = None
        self._active_leg = 0
        self._terrain_land_xy: np.ndarray | None = None
        self._cleared_legs: set[int] = set()
        self._last_roll_dist: float | None = None
        self._last_progress_t = 0.0
        self._recovery_cooldown_until = 0.0
        self._recoveries_by_checkpoint: dict[int, int] = {}
        self._max_recoveries_per_checkpoint = self.layout.terrain_fly_after_recoveries + 2
        self._cfg = controller.config
        self._landing_height_nominal = float(self._cfg.landing_height)
        self._hover_height_nominal = float(self._cfg.hover_height)
        self._spawn_nominal = self._cfg.spawn_xy.copy()

    def begin(self, t: float) -> None:
        self.follower.set_checkpoint_index(0)
        self._set_roll_target(self.follower.current_target_xy())
        self._last_progress_t = t
        self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Begin rolling toward first checkpoint")

    def mission_scan(self, scan: ObstacleScan, state) -> ObstacleScan:
        return filter_scan_for_mission(
            scan,
            state,
            self.layout,
            checkpoint_index=self.follower.checkpoint_index,
            cleared_legs=self._cleared_legs,
        )

    def _reset_roll_progress(self, state, t: float) -> None:
        target = self.follower.current_target_xy()
        self._last_roll_dist = float(np.linalg.norm(state.position[:2] - target))
        self._last_progress_t = t

    def _rolling_stuck(self, state, t: float, target_xy: np.ndarray) -> bool:
        if t < self._recovery_cooldown_until:
            return False

        dist = float(np.linalg.norm(state.position[:2] - target_xy))
        if dist < self.layout.patrol_arrival_radius + 0.20:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False
        if self._last_roll_dist is None:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False
        if self._last_roll_dist - dist >= 0.12:
            self._last_roll_dist = dist
            self._last_progress_t = t
            return False

        cmd = self.follower.velocity_command_xy(state, None)
        if float(np.linalg.norm(cmd)) < 0.08:
            slow = float(np.linalg.norm(state.velocity[:2])) < 0.30
            return slow and (t - self._last_progress_t) >= 1.5
        slow = float(np.linalg.norm(state.velocity[:2])) < 0.35
        return slow and (t - self._last_progress_t) >= 2.0

    def _should_fly_terrain(self, state, target_xy: np.ndarray) -> bool:
        cp = self.follower.checkpoint_index
        if cp < self.layout.terrain_fly_min_checkpoint:
            return False
        wall = self._wall_for_current_leg()
        if wall is not None and wall.leg_index not in self._cleared_legs:
            return False
        dist = float(np.linalg.norm(state.position[:2] - target_xy))
        return dist > 1.2

    def _in_post_wall_ditch(self, state) -> bool:
        """Rough bounds of the V-ditch immediately after wall 2."""
        if self.follower.checkpoint_index < self.layout.terrain_fly_min_checkpoint:
            return False
        x = float(state.position[0])
        y = float(state.position[1])
        return 1.0 <= x <= 4.8 and -0.8 <= y <= 1.5

    def _start_terrain_hop(
        self,
        state,
        t: float,
        land_xy: np.ndarray,
        *,
        immediate_flight: bool = False,
    ) -> None:
        self._terrain_land_xy = np.asarray(land_xy[:2], dtype=float).copy()
        self._active_wall = None
        lift, cruise, approach = self.layout.terrain_hop_waypoints(
            state.position[:2],
            self._terrain_land_xy,
            self.terrain_z_at,
            current_z=float(state.position[2]),
        )
        hover_z = max(float(lift[2]), float(cruise[2]), float(approach[2]))
        self._set_flight_waypoints(
            [lift.tolist(), cruise.tolist(), approach.tolist()],
            hover_z=hover_z,
        )
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        self.follower.enabled = False
        launch_now = immediate_flight or self._in_post_wall_ditch(state)
        if launch_now:
            self.ctrl.mode_manager.request_flight()
            self._enter(
                CoursePhase.FLY_TERRAIN,
                t,
                f"Direct flight over ditch toward ({self._terrain_land_xy[0]:.1f},{self._terrain_land_xy[1]:.1f})",
            )
        else:
            self.ctrl.mode_manager.request_takeoff()
            self._enter(
                CoursePhase.TAKEOFF_OVER,
                t,
                f"Flying over rough terrain toward ({self._terrain_land_xy[0]:.1f},{self._terrain_land_xy[1]:.1f})",
            )

    def _begin_terrain_flight(self, state, t: float) -> None:
        self.ctrl.mode_manager.request_flight()
        self._enter(CoursePhase.FLY_TERRAIN, t, "Air transit over ditch/hills")

    def _skip_ahead_on_stall(self, state) -> bool:
        """Advance target when close but terrain keeps stalling."""
        cp = self.follower.checkpoint_index
        target = self.follower.current_target_xy()
        dist = float(np.linalg.norm(state.position[:2] - target))
        if cp >= len(self.layout.checkpoints):
            return False
        if cp >= len(self.layout.checkpoints) - 1 and dist < 3.0:
            self.follower.set_checkpoint_index(len(self.layout.checkpoints))
            self._set_roll_target(self.layout.goal_xy)
            return True
        if dist < 2.0:
            self.follower.advance_checkpoint()
            self._set_roll_target(self.follower.current_target_xy())
            return True
        return False

    def _enter(self, phase: CoursePhase, t: float, msg: str) -> None:
        self.phase = phase
        self._phase_t0 = t
        self._land_hold = 0.0
        if phase == CoursePhase.ROLL_TO_CHECKPOINT:
            self._last_roll_dist = None
        if phase != CoursePhase.ROLL_TO_CHECKPOINT:
            self._blocked_timer = 0.0
        print(f"t={t:6.2f}s  PHASE -> {phase.value}  |  {msg}")

    def _set_roll_target(self, xy: np.ndarray) -> None:
        self._cfg.roll_target = np.asarray(xy[:2], dtype=float).copy()

    def _set_ground_context(self, xy: np.ndarray) -> None:
        ground_z = float(self.terrain_z_at(float(xy[0]), float(xy[1])))
        self._cfg.landing_height = ground_z + self.layout.cage_radius
        self._cfg.spawn_xy = np.asarray(xy[:2], dtype=float).copy()

    def _set_flight_waypoints(self, waypoints: list[list[float]], hover_z: float) -> None:
        self._cfg.hover_height = hover_z
        wps = [np.array(w, dtype=float) for w in waypoints]
        self.ctrl.planner._mission_waypoints = [w.copy() for w in wps]
        self.ctrl.planner._waypoints = [w.copy() for w in wps]
        self.ctrl.planner.reset()

    def _near_xy(self, state, xy: np.ndarray, tol: float | None = None) -> bool:
        tol = self.layout.patrol_arrival_radius if tol is None else tol
        return float(np.linalg.norm(state.position[:2] - xy)) < tol

    def _ground_settled(self, state) -> bool:
        return (
            state.on_ground
            and float(np.linalg.norm(state.velocity)) < 0.10
            and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold
        )

    def _goal_reached(self, state) -> bool:
        if self.follower.at_final_goal(state):
            return True
        if self.follower.checkpoint_index >= len(self.layout.checkpoints):
            return self.layout.near_goal(state.position[:2])
        return False

    def current_target_xy(self) -> np.ndarray:
        if self._terrain_land_xy is not None and self.phase in (
            CoursePhase.FLY_TERRAIN,
            CoursePhase.LAND_TERRAIN,
        ):
            return self._terrain_land_xy.copy()
        if self.phase in (CoursePhase.FLY_OVER, CoursePhase.LAND_BEYOND) and self._active_wall is not None:
            return self.layout.landing_xy_after_wall(self._active_wall, self._active_leg)
        return self.follower.current_target_xy()

    def _wall_for_current_leg(self) -> WallSpec | None:
        return self.layout.wall_for_leg(self.follower.current_leg_index)

    def _start_fly_over(self, state, t: float, wall: WallSpec) -> None:
        self._terrain_land_xy = None
        leg = self.follower.current_leg_index
        pre, over, post = self.layout.fly_over_waypoints(wall, self.terrain_z_at, leg_index=leg)
        hover_z = max(float(pre[2]), float(over[2]), float(post[2]))
        self._set_flight_waypoints(
            [pre.tolist(), over.tolist(), post.tolist()],
            hover_z=hover_z,
        )
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        self.ctrl.mode_manager.request_flight()
        self._enter(CoursePhase.FLY_OVER, t, f"Fly over wall at ({wall.x:.1f},{wall.y:.1f})")

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
        self.follower.enabled = self.phase == CoursePhase.ROLL_TO_CHECKPOINT

        if self.phase == CoursePhase.ROLL_TO_CHECKPOINT:
            self._set_ground_context(state.position[:2])
            self._set_roll_target(self.follower.current_target_xy())
            leg = self.follower.current_leg_index
            wall = self._wall_for_current_leg()
            self.follower.respect_blocked_scan = (
                wall is not None and leg not in self._cleared_legs
            )

            if self._goal_reached(state) and float(np.linalg.norm(state.velocity[:2])) < 0.35:
                mm.request_idle()
                self._enter(CoursePhase.DONE, t, "Goal reached")
                return False

            target_xy = self.follower.current_target_xy()
            if self._in_post_wall_ditch(state) and self._should_fly_terrain(state, target_xy):
                self._start_terrain_hop(state, t, target_xy, immediate_flight=True)
                return self.phase != CoursePhase.DONE
            if self._rolling_stuck(state, t, target_xy):
                cp = self.follower.checkpoint_index
                recoveries = self._recoveries_by_checkpoint.get(cp, 0)
                fly_threshold = self.layout.terrain_fly_after_recoveries
                if recoveries >= fly_threshold and self._should_fly_terrain(state, target_xy):
                    self._start_terrain_hop(state, t, target_xy)
                    return self.phase != CoursePhase.DONE
                if recoveries < fly_threshold:
                    self._recoveries_by_checkpoint[cp] = recoveries + 1
                    self.follower.enabled = False
                    mm.request_ground_recovery(resume_rolling=True)
                    self._enter(
                        CoursePhase.ROLL_RECOVER,
                        t,
                        f"Terrain stall on cp {cp} - reorient (attempt {recoveries + 1})",
                    )
                    return self.phase != CoursePhase.DONE
                if self._should_fly_terrain(state, target_xy):
                    self._start_terrain_hop(state, t, target_xy)
                    return self.phase != CoursePhase.DONE
                if self._skip_ahead_on_stall(state):
                    self._last_roll_dist = None
                else:
                    self._last_progress_t = t

            if (
                self.follower.near_current_target(state)
                and float(np.linalg.norm(state.velocity[:2])) < 0.20
            ):
                if self.follower.checkpoint_index >= len(self.layout.checkpoints):
                    if self._goal_reached(state):
                        mm.request_idle()
                        self._enter(CoursePhase.DONE, t, "Goal reached")
                        return False
                else:
                    self.follower.advance_checkpoint()
                    self._set_roll_target(self.follower.current_target_xy())
                    self._last_roll_dist = None

            if (
                wall is not None
                and leg not in self._cleared_legs
                and self._blocked_timer >= self.layout.blocked_confirm_time
            ):
                if mission_scan.flyable or wall.height > self.layout.max_roll_height:
                    self._active_wall = wall
                    self._active_leg = leg
                    self.follower.enabled = False
                    mm.request_takeoff()
                    self._enter(CoursePhase.BLOCKED_RECOVER, t, "Wall detected - recover for fly-over")
            elif wall is not None and leg not in self._cleared_legs:
                if mission_scan.blocked or float(np.linalg.norm(state.velocity[:2])) < 0.12:
                    self._blocked_timer += dt
                else:
                    self._blocked_timer = max(0.0, self._blocked_timer - dt)
            else:
                self._blocked_timer = max(0.0, self._blocked_timer - dt)

        elif self.phase == CoursePhase.ROLL_RECOVER:
            bz = float(body_z_world(state.rotation)[2])
            cp = self.follower.checkpoint_index
            if cp >= self.layout.terrain_fly_min_checkpoint and (t - self._phase_t0) >= 2.0:
                target = self.follower.current_target_xy()
                if self._should_fly_terrain(state, target):
                    self._start_terrain_hop(state, t, target)
                    return self.phase != CoursePhase.DONE
            if mode == ControlMode.PRETAKEOFF and bz >= self._cfg.upright_cos_threshold * 0.88:
                mm.mode = ControlMode.UPRIGHT
                mm.upright_timer = 0.12
            settled = self._ground_settled(state) or (
                mode == ControlMode.ROLLING
                and state.on_ground
                and bz >= self._cfg.upright_cos_threshold * 0.85
                and float(np.linalg.norm(state.velocity[:2])) < 0.50
            )
            if settled:
                self._reset_roll_progress(state, t)
                self._recovery_cooldown_until = t + 3.0
                self.follower.enabled = True
                self.ctrl.rolling.reset()
                mm.request_roll()
                self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Resume rolling after terrain recovery")
            elif mode not in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT, ControlMode.ROLLING):
                mm.request_ground_recovery(resume_rolling=True)

        elif self.phase == CoursePhase.BLOCKED_RECOVER:
            bz = float(body_z_world(state.rotation)[2])
            if mode == ControlMode.ROLLING:
                mm.request_takeoff()
            elif mode in (ControlMode.UPRIGHT, ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                self._enter(CoursePhase.TAKEOFF_OVER, t, "Upright - starting fly-over takeoff")
            elif mode == ControlMode.PRETAKEOFF and bz >= self._cfg.upright_cos_threshold:
                mm.mode = ControlMode.UPRIGHT
                mm.upright_timer = float(self._cfg.upright_settle_time)
                self._enter(CoursePhase.TAKEOFF_OVER, t, "Recovered upright at wall")

        elif self.phase == CoursePhase.TAKEOFF_OVER:
            if self._terrain_land_xy is not None and self._active_wall is None:
                bz = float(body_z_world(state.rotation)[2])
                elapsed = t - self._phase_t0
                if mode == ControlMode.FLIGHT:
                    self._enter(CoursePhase.FLY_TERRAIN, t, "Air transit over ditch/hills")
                elif mode in (ControlMode.UPRIGHT, ControlMode.TAKEOFF, ControlMode.HOVER):
                    self._begin_terrain_flight(state, t)
                elif mode == ControlMode.ROLLING:
                    mm.request_takeoff()
                elif mode == ControlMode.PRETAKEOFF and bz >= self._cfg.upright_cos_threshold * 0.80:
                    mm.mode = ControlMode.UPRIGHT
                    mm.upright_timer = 0.08
                elif elapsed >= 2.5 or (elapsed >= 1.0 and bz >= 0.72):
                    self._begin_terrain_flight(state, t)
            elif self._active_wall is not None:
                if mode in (ControlMode.UPRIGHT, ControlMode.TAKEOFF, ControlMode.HOVER):
                    self._start_fly_over(state, t, self._active_wall)
                elif mode == ControlMode.FLIGHT:
                    pass
                elif mode == ControlMode.ROLLING:
                    mm.request_takeoff()
                elif mode == ControlMode.PRETAKEOFF and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold:
                    self._start_fly_over(state, t, self._active_wall)

        elif self.phase == CoursePhase.FLY_OVER:
            if self._active_wall is None:
                self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Resume rolling")
            else:
                land_xy = self.layout.landing_xy_after_wall(self._active_wall, self._active_leg)
                if mode == ControlMode.LANDING and not self._near_xy(state, land_xy, tol=0.70):
                    mm.request_flight()
                dist_xy = float(np.linalg.norm(state.position[:2] - land_xy))
                if dist_xy < 0.85 and state.position[2] > self._cfg.landing_height + 0.15:
                    mm.request_landing()
                    self._enter(CoursePhase.LAND_BEYOND, t, "Landing beyond wall")
                elif (
                    self.ctrl.planner.mission_complete
                    and dist_xy < 1.2
                    and state.position[2] > self._cfg.landing_height + 0.10
                ):
                    mm.request_landing()
                    self._enter(CoursePhase.LAND_BEYOND, t, "Landing beyond wall")

        elif self.phase == CoursePhase.FLY_TERRAIN:
            if self._terrain_land_xy is None:
                self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Resume rolling")
            else:
                land_xy = self._terrain_land_xy
                if mode == ControlMode.LANDING and not self._near_xy(state, land_xy, tol=0.85):
                    mm.request_flight()
                dist_xy = float(np.linalg.norm(state.position[:2] - land_xy))
                if dist_xy < 1.0 and state.position[2] > self._cfg.landing_height + 0.12:
                    mm.request_landing()
                    self._enter(CoursePhase.LAND_TERRAIN, t, "Landing after terrain flight")
                elif (
                    self.ctrl.planner.mission_complete
                    and dist_xy < 1.4
                    and state.position[2] > self._cfg.landing_height + 0.08
                ):
                    mm.request_landing()
                    self._enter(CoursePhase.LAND_TERRAIN, t, "Landing after terrain flight")

        elif self.phase == CoursePhase.LAND_TERRAIN:
            if self._terrain_land_xy is not None:
                self._set_ground_context(self._terrain_land_xy)
            if mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.TAKEOFF):
                mm.request_landing()
            if state.on_ground and mode == ControlMode.LANDING:
                bz = float(body_z_world(state.rotation)[2])
                if bz < self._cfg.upright_cos_threshold:
                    mm.request_takeoff()
            settled = self._ground_settled(state) or (
                state.on_ground
                and float(np.linalg.norm(state.velocity)) < 0.18
                and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold * 0.85
            )
            if settled:
                self._land_hold += dt
                if self._land_hold >= 0.7:
                    land_xy = self._terrain_land_xy
                    self._terrain_land_xy = None
                    self._recoveries_by_checkpoint[self.follower.checkpoint_index] = 0
                    if land_xy is not None and float(np.linalg.norm(state.position[:2] - land_xy)) < 1.5:
                        if self.follower.checkpoint_index < len(self.layout.checkpoints):
                            self.follower.advance_checkpoint()
                    self._set_roll_target(self.follower.current_target_xy())
                    self._last_roll_dist = None
                    self.ctrl.rolling.reset()
                    mm.request_roll()
                    self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Resumed rolling after terrain flight")
            else:
                self._land_hold = max(0.0, self._land_hold - dt)

        elif self.phase == CoursePhase.LAND_BEYOND:
            if self._active_wall is not None:
                land_xy = self.layout.landing_xy_after_wall(self._active_wall, self._active_leg)
                self._set_ground_context(land_xy)
            if mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.TAKEOFF):
                mm.request_landing()
            if state.on_ground and mode == ControlMode.LANDING:
                bz = float(body_z_world(state.rotation)[2])
                if bz < self._cfg.upright_cos_threshold:
                    mm.request_takeoff()
            settled = self._ground_settled(state) or (
                state.on_ground
                and float(np.linalg.norm(state.velocity)) < 0.15
                and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold * 0.85
            )
            if settled:
                self._land_hold += dt
                if self._land_hold >= 0.8:
                    cleared_leg = self._active_leg
                    if self._active_wall is not None and cleared_leg is not None:
                        self._cleared_legs.add(cleared_leg)
                    self._active_wall = None
                    if cleared_leg + 1 < len(self.layout.checkpoints):
                        self.follower.set_checkpoint_index(cleared_leg + 1)
                    fly_target = self.follower.current_target_xy()
                    self._last_roll_dist = None
                    if (
                        cleared_leg == self.layout.auto_fly_after_wall_leg
                        and self.follower.checkpoint_index >= self.layout.terrain_fly_min_checkpoint
                    ):
                        self._start_terrain_hop(
                            state,
                            t,
                            fly_target,
                            immediate_flight=True,
                        )
                    else:
                        self._set_roll_target(fly_target)
                        self.ctrl.rolling.reset()
                        mm.request_roll()
                        self._enter(CoursePhase.ROLL_TO_CHECKPOINT, t, "Resumed rolling after fly-over")
            else:
                self._land_hold = max(0.0, self._land_hold - dt)

        elif self.phase == CoursePhase.DONE:
            return False

        return self.phase != CoursePhase.DONE
