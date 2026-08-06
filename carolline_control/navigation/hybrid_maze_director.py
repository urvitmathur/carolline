"""Obstacle-driven hybrid roll/fly maze mission director."""

from __future__ import annotations

import enum

import numpy as np

from carolline_control.carolline_controller import CarollineController
from carolline_control.navigation.hybrid_maze_layout import HybridMazeLayout
from carolline_control.navigation.perception import ObstacleScan, filter_scan_for_hybrid_maze
from carolline_control.utils.so3 import body_z_world, rot_to_euler_zyx
from carolline_control.utils.types import ControlMode

TakeoffReason = str  # "WALL" | "GAP" | "CLIMB"


class HybridMazePhase(enum.Enum):
    ROLL = "ROLL"
    RECOVER = "RECOVER"
    FLY = "FLY"
    LAND = "LAND"
    DONE = "DONE"


class HybridMazeDirector:
    """Roll toward nav goals; takeoff/fly/land when rangefinders classify obstacles."""

    def __init__(
        self,
        controller: CarollineController,
        layout: HybridMazeLayout,
        *,
        blocked_confirm_time: float = 0.35,
    ) -> None:
        self.ctrl = controller
        self.layout = layout
        self.blocked_confirm_time = float(blocked_confirm_time)
        self.phase = HybridMazePhase.ROLL
        self._phase_t0 = 0.0
        self._hover_ready_t: float | None = None
        self._land_hold = 0.0
        self._cfg = controller.config
        self._landing_height_nominal = float(self._cfg.landing_height)
        self._hover_height_nominal = float(self._cfg.hover_height)
        self._spawn_nominal = self._cfg.spawn_xy.copy()
        self._active_land_xy = layout.spawn_xy.copy()
        self._active_cruise_z = self._hover_height_nominal
        self._roll_target_key = "deadend"
        self._deadend_mapped = False
        self._cleared: set[str] = set()
        self._confirm_timers: dict[str, float] = {"WALL": 0.0, "GAP": 0.0, "CLIMB": 0.0}
        self._takeoff_reason: TakeoffReason | None = None
        self._flight_waypoints: list[np.ndarray] = []
        self._land_on_platform = False
        self._last_takeoff_reason: TakeoffReason | None = None
        self._climb_stall_t = 0.0

    def begin(self, t: float) -> None:
        self._active_land_xy = self.layout.spawn_xy.copy()
        self._set_ground_context(on_platform=False)
        self._set_roll_target(self._roll_target_xy())
        self._enter(HybridMazePhase.ROLL, t, "Roll toward dead-end (map spur)")

    def _enter(self, phase: HybridMazePhase, t: float, msg: str) -> None:
        self.phase = phase
        self._phase_t0 = t
        self._hover_ready_t = None
        self._land_hold = 0.0
        print(f"t={t:6.2f}s  PHASE -> {phase.value}  |  {msg}")

    def current_target_xy(self) -> np.ndarray:
        if self.phase == HybridMazePhase.DONE:
            return self.layout.goal_xy.copy()
        if self.phase in (HybridMazePhase.FLY, HybridMazePhase.LAND) and self._flight_waypoints:
            return self._flight_waypoints[-1][:2].copy()
        return self._roll_target_xy().copy()

    def _set_roll_target(self, xy: np.ndarray) -> None:
        self._cfg.roll_target = np.asarray(xy[:2], dtype=float).copy()

    def _set_ground_context(self, *, on_platform: bool) -> None:
        if on_platform:
            com_z = self.layout.platform.com_z(self.layout.cage_radius)
            self._cfg.landing_height = com_z
            self._cfg.spawn_xy = self.layout.checkpoints["platform_land"].copy()
        else:
            self._cfg.landing_height = self._landing_height_nominal
            self._cfg.spawn_xy = self._active_land_xy.copy()

    def _roll_target_xy(self) -> np.ndarray:
        """Roll routing hints; flight remains perception-driven."""
        cps = self.layout.checkpoints
        if not self._deadend_mapped:
            return cps["deadend"]
        if "WALL" not in self._cleared:
            return cps["hop_approach"]
        if "GAP" not in self._cleared:
            return cps["gap_approach"]
        if "CLIMB" not in self._cleared:
            return cps["shaft_base"]
        return self.layout.goal_xy

    def _set_flight_waypoints(self, waypoints: list[np.ndarray], hover_z: float) -> None:
        self._cfg.hover_height = float(hover_z)
        self._active_cruise_z = float(hover_z)
        wps = [np.asarray(w, dtype=float) for w in waypoints]
        self.ctrl.planner._mission_waypoints = [w.copy() for w in wps]
        self.ctrl.planner._waypoints = [w.copy() for w in wps]
        self.ctrl.planner.reset()

    def _near_xy(self, state, xy: np.ndarray, tol: float | None = None) -> bool:
        tol = self.layout.arrival_radius if tol is None else tol
        return float(np.linalg.norm(state.position[:2] - np.asarray(xy[:2], dtype=float))) < tol

    def _slow(self, state, limit: float = 0.12) -> bool:
        return float(np.linalg.norm(state.velocity[:2])) < limit

    def _ground_settled(self, state) -> bool:
        z_ok = float(state.position[2]) < self.layout.cage_radius + 0.25
        slow = float(np.linalg.norm(state.velocity)) < 0.20
        upright = float(body_z_world(state.rotation)[2]) >= min(
            0.75, float(self._cfg.upright_cos_threshold)
        )
        return z_ok and slow and (bool(state.on_ground) or upright)

    def _platform_settled(self, state) -> bool:
        com_z = self.layout.platform.com_z(self.layout.cage_radius)
        return (
            self.layout.platform.contains_xy(state.position[:2], margin=0.20)
            and abs(float(state.position[2]) - com_z) < 0.20
            and float(np.linalg.norm(state.velocity)) < 0.20
            and float(body_z_world(state.rotation)[2])
            >= min(0.75, float(self._cfg.upright_cos_threshold))
        )

    def _bearing(self, state, scan: ObstacleScan) -> float:
        yaw = float(rot_to_euler_zyx(state.rotation)[2])
        return yaw + float(np.radians(scan.forward_argmin_angle))

    def _mission_scan(self, scan: ObstacleScan, state) -> ObstacleScan:
        on_platform = self.layout.platform.contains_xy(state.position[:2], margin=0.15)
        return filter_scan_for_hybrid_maze(
            scan,
            state,
            self.layout,
            cleared=self._cleared,
            on_platform=on_platform,
        )

    def _build_waypoints(
        self,
        reason: TakeoffReason,
        state,
        scan: ObstacleScan,
        hover: float,
    ) -> tuple[list[np.ndarray], bool]:
        pose = state.position[:2]
        bearing = self._bearing(state, scan)
        layout = self.layout
        if reason == "WALL":
            wps = layout.wall_hop_waypoints(
                pose, bearing, scan.forward_min_m, hover
            )
            return wps, False
        if reason == "GAP":
            wps = layout.gap_cross_waypoints(pose, bearing, hover)
            return wps, False
        land = layout.checkpoints["platform_land"]
        wps = layout.climb_waypoints(pose, land, hover)
        return wps, True

    def _start_takeoff(
        self,
        reason: TakeoffReason,
        state,
        scan: ObstacleScan,
        t: float,
        hover: float,
    ) -> None:
        self._takeoff_reason = reason
        self._last_takeoff_reason = reason
        self._flight_waypoints, self._land_on_platform = self._build_waypoints(
            reason, state, scan, hover
        )
        land_xy = self._flight_waypoints[-1][:2]
        self._active_land_xy = np.asarray(land_xy, dtype=float).copy()
        px, py = float(state.position[0]), float(state.position[1])
        print(
            f"t={t:6.2f}s  TAKEOFF reason={reason}  "
            f"pos=({px:.2f},{py:.2f})  fwd={scan.forward_min_m:.2f}  "
            f"up={scan.upward_clear_m:.2f}  down={scan.down_min_m:.2f}"
        )
        self.ctrl.mode_manager.request_takeoff()
        self._enter(HybridMazePhase.RECOVER, t, f"Recover for {reason} takeoff")

    def _accumulate_confirm(
        self, mission_scan: ObstacleScan, state, dt: float
    ) -> TakeoffReason | None:
        xy = state.position[:2]
        layout = self.layout
        candidates: list[tuple[str, bool]] = []
        if layout.near_low_wall_xy(xy):
            candidates.append(("WALL", mission_scan.wall_block))
        if layout.near_gap_xy(xy):
            candidates.append(("GAP", mission_scan.floor_gap))
        if layout.near_climb_xy(xy):
            climb_hit = mission_scan.climb_face or mission_scan.blocked or mission_scan.flyable
            candidates.append(("CLIMB", climb_hit))

        active = {key for key, _ in candidates}
        for key in ("WALL", "GAP", "CLIMB"):
            if key not in active:
                self._confirm_timers[key] = max(0.0, self._confirm_timers[key] - 0.5 * dt)

        for key, flag in candidates:
            if flag:
                self._confirm_timers[key] += dt
            else:
                self._confirm_timers[key] = max(0.0, self._confirm_timers[key] - 0.5 * dt)

        for key, flag in candidates:
            if flag and self._confirm_timers[key] >= self.blocked_confirm_time:
                return key
        return None

    def _maybe_launch_flight(
        self,
        state,
        mode: ControlMode,
        t: float,
        *,
        hover_z: float,
    ) -> None:
        mm = self.ctrl.mode_manager
        if mode == ControlMode.HOVER:
            if self._hover_ready_t is None:
                self._hover_ready_t = t
            elif t - self._hover_ready_t >= self.layout.hover_settle_s:
                self._set_flight_waypoints(self._flight_waypoints, hover_z)
                self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
                mm.request_flight()
                reason = self._takeoff_reason or "?"
                self._enter(HybridMazePhase.FLY, t, f"Fly segment ({reason})")
        elif mode == ControlMode.FLIGHT:
            self._set_flight_waypoints(self._flight_waypoints, hover_z)
            self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
            mm.request_flight()
            reason = self._takeoff_reason or "?"
            self._enter(HybridMazePhase.FLY, t, f"Fly segment ({reason})")

    def _fly_and_land(
        self,
        state,
        mode: ControlMode,
        t: float,
    ) -> None:
        mm = self.ctrl.mode_manager
        land_xy = self._active_land_xy
        cruise = self._active_cruise_z
        reason = self._takeoff_reason or ""

        if reason == "CLIMB":
            pad_approach_z = self.layout.platform.com_z(self.layout.cage_radius) + 0.35
            if mode == ControlMode.LANDING and not self._near_xy(
                state, land_xy, tol=self.layout.fly_xy_tol + 0.15
            ):
                mm.request_flight()
            near_pad = self._near_xy(state, land_xy, tol=self.layout.fly_xy_tol)
            high_enough = float(state.position[2]) > self.layout.platform.top_z + 0.25
            if near_pad and high_enough and abs(float(state.position[2]) - pad_approach_z) < 0.55:
                self._set_ground_context(on_platform=True)
                self._cfg.spawn_xy = land_xy.copy()
                mm.request_landing()
                self._enter(HybridMazePhase.LAND, t, "Land on raised platform")
            return

        if mode == ControlMode.LANDING and not (
            self._near_xy(state, land_xy, tol=self.layout.fly_xy_tol + 0.1)
            and abs(float(state.position[2]) - cruise) < self.layout.fly_z_tol + 0.15
        ):
            mm.request_flight()
        if self._near_xy(state, land_xy, tol=self.layout.fly_xy_tol) and abs(
            float(state.position[2]) - cruise
        ) < self.layout.fly_z_tol:
            self._set_ground_context(on_platform=self._land_on_platform)
            self._cfg.spawn_xy = land_xy.copy()
            mm.request_landing()
            self._enter(HybridMazePhase.LAND, t, f"Land after {reason}")

    def _ensure_rolling(self, state, mode: ControlMode, t: float) -> None:
        """Force a return to ROLLING after land / stall (matches course director recovery)."""
        mm = self.ctrl.mode_manager
        if mode == ControlMode.ROLLING:
            return
        mm.hold_upright_after_recovery = False
        bz = float(body_z_world(state.rotation)[2])
        upright_ok = bz >= float(self._cfg.upright_cos_threshold) * 0.88
        if self.phase == HybridMazePhase.ROLL and mode in (
            ControlMode.PRETAKEOFF,
            ControlMode.UPRIGHT,
        ):
            mm.mode = ControlMode.ROLLING
            mm.request_rolling = False
            mm.resume_rolling_after_recovery = False
            self.ctrl.rolling.reset()
            return
        if mode == ControlMode.IDLE:
            mm.request_roll()
            return
        if mode == ControlMode.PRETAKEOFF and upright_ok:
            mm.mode = ControlMode.UPRIGHT
            mm.upright_timer = 0.12
            mm.resume_rolling_after_recovery = True
            mm.request_rolling = True
            mm.request_roll()
            return
        if mode == ControlMode.UPRIGHT:
            mm.resume_rolling_after_recovery = True
            mm.request_roll()
            return
        if (
            (t - self._phase_t0) >= 2.5
            and bool(state.on_ground)
            and mode
            not in (
                ControlMode.TAKEOFF,
                ControlMode.HOVER,
                ControlMode.FLIGHT,
                ControlMode.LANDING,
            )
        ):
            mm.mode = ControlMode.ROLLING
            mm.request_rolling = False
            mm.resume_rolling_after_recovery = False
            self.ctrl.rolling.reset()
            print(f"t={t:6.2f}s  MODE force -> ROLLING (stuck recovery)")

    def _finish_land_to_roll(
        self,
        state,
        mode: ControlMode,
        t: float,
        dt: float,
    ) -> None:
        mm = self.ctrl.mode_manager
        on_platform = self._land_on_platform
        self._set_ground_context(on_platform=on_platform)
        self._cfg.spawn_xy = self._active_land_xy.copy()
        settled_fn = self._platform_settled if on_platform else self._ground_settled
        near_pad = self._near_xy(state, self._active_land_xy, tol=0.55)
        if settled_fn(state) or mode in (ControlMode.IDLE, ControlMode.ROLLING):
            self._land_hold += dt
        elif near_pad and mode in (ControlMode.PRETAKEOFF, ControlMode.LANDING) and self._slow(
            state, 0.25
        ):
            self._land_hold += dt
        else:
            self._land_hold = max(0.0, self._land_hold - 0.25 * dt)

        timed_out = (t - self._phase_t0) >= 4.0 and near_pad and self._slow(state, 0.30)
        if self._land_hold >= self.layout.land_hold_s or timed_out:
            if self._takeoff_reason:
                self._cleared.add(self._takeoff_reason)
                self._confirm_timers[self._takeoff_reason] = 0.0
            self._takeoff_reason = None
            self._flight_waypoints = []
            if on_platform:
                self._roll_target_key = "goal"
                next_xy = self.layout.goal_xy
                msg = "Roll on platform to target"
            else:
                next_xy = self._roll_target_xy()
                self._roll_target_key = "goal"
                msg = "Resume roll toward next segment"
            self._set_roll_target(next_xy)
            self.ctrl.rolling.reset()
            mm.hold_upright_after_recovery = False
            mm.resume_rolling_after_recovery = True
            mm.request_roll()
            self._ensure_rolling(state, mode, t)
            self._enter(HybridMazePhase.ROLL, t, msg)

    def update(
        self,
        state,
        mode: ControlMode,
        t: float,
        dt: float,
        scan: ObstacleScan,
    ) -> bool:
        layout = self.layout
        hover = max(self._hover_height_nominal, layout.hover_height_nominal)
        mm = self.ctrl.mode_manager
        mission_scan = self._mission_scan(scan, state)

        if self.phase == HybridMazePhase.ROLL:
            on_platform = layout.platform.contains_xy(state.position[:2], margin=0.15)
            self._set_ground_context(on_platform=on_platform)
            target = self._roll_target_xy()
            self._set_roll_target(target)
            if self._deadend_mapped and self._roll_target_key != "goal":
                if "CLIMB" in self._cleared:
                    self._roll_target_key = "goal"
                elif self._near_xy(state, target) and self._slow(state):
                    pass  # stay on current roll hint until obstacle clears
            if not self._deadend_mapped:
                if self._near_xy(state, target) and self._slow(state):
                    self._deadend_mapped = True
                    self._roll_target_key = "goal"
                    self._set_roll_target(self._roll_target_xy())
                    print(f"t={t:6.2f}s  Dead-end mapped — roll toward next corridor segment")
            elif self._roll_target_key == "goal" or "CLIMB" in self._cleared:
                at_goal = (
                    self._near_xy(state, layout.goal_xy, tol=0.40)
                    and self._slow(state, 0.10)
                    and on_platform
                )
                if at_goal and mode in (
                    ControlMode.ROLLING,
                    ControlMode.PRETAKEOFF,
                    ControlMode.IDLE,
                    ControlMode.UPRIGHT,
                ):
                    mm.request_idle()
                    self._enter(HybridMazePhase.DONE, t, "Target reached on platform")
                    return False

            self._ensure_rolling(state, mode, t)

            rolling_like = mode in (ControlMode.ROLLING, ControlMode.PRETAKEOFF, ControlMode.UPRIGHT)
            if rolling_like and self._deadend_mapped:
                reason = self._accumulate_confirm(mission_scan, state, dt)
                if (
                    reason is None
                    and "GAP" in self._cleared
                    and "CLIMB" not in self._cleared
                    and layout.near_climb_xy(state.position[:2])
                ):
                    if float(np.linalg.norm(state.velocity[:2])) < 0.10:
                        self._climb_stall_t += dt
                    else:
                        self._climb_stall_t = max(0.0, self._climb_stall_t - dt)
                    if self._climb_stall_t >= 1.2 and (
                        mission_scan.blocked or mission_scan.flyable
                    ):
                        reason = "CLIMB"
                slow_or_blocked = (
                    mission_scan.wall_block
                    or mission_scan.floor_gap
                    or mission_scan.climb_face
                    or mission_scan.blocked
                    or float(np.linalg.norm(state.velocity[:2])) < 0.15
                )
                if reason and slow_or_blocked:
                    self._climb_stall_t = 0.0
                    self._start_takeoff(reason, state, scan, t, hover)

        elif self.phase == HybridMazePhase.RECOVER:
            on_platform = layout.platform.contains_xy(state.position[:2], margin=0.15)
            self._set_ground_context(on_platform=on_platform)
            if mode == ControlMode.ROLLING:
                mm.request_takeoff()
            elif mode == ControlMode.PRETAKEOFF:
                bz = float(body_z_world(state.rotation)[2])
                if bz >= float(self._cfg.upright_cos_threshold) * 0.88:
                    mm.mode = ControlMode.UPRIGHT
                    mm.upright_timer = 0.12
            elif mode in (
                ControlMode.UPRIGHT,
                ControlMode.TAKEOFF,
                ControlMode.HOVER,
                ControlMode.FLIGHT,
            ):
                hover_z = float(self._flight_waypoints[0][2]) if self._flight_waypoints else hover
                if self._takeoff_reason == "CLIMB" and len(self._flight_waypoints) > 1:
                    hover_z = float(self._flight_waypoints[1][2])
                self._maybe_launch_flight(state, mode, t, hover_z=hover_z)
            elif (t - self._phase_t0) >= 2.5:
                bz = float(body_z_world(state.rotation)[2])
                hover_z = float(self._flight_waypoints[0][2]) if self._flight_waypoints else hover
                if bz >= 0.72 or mode == ControlMode.PRETAKEOFF:
                    mm.mode = ControlMode.UPRIGHT
                    mm.upright_timer = 0.08
                mm.request_takeoff()
                self._maybe_launch_flight(state, mode, t, hover_z=hover_z)

        elif self.phase == HybridMazePhase.FLY:
            on_platform = self._land_on_platform
            self._set_ground_context(on_platform=on_platform)
            self._fly_and_land(state, mode, t)

        elif self.phase == HybridMazePhase.LAND:
            self._finish_land_to_roll(state, mode, t, dt)

        elif self.phase == HybridMazePhase.DONE:
            mm.request_idle()
            return False

        return True
