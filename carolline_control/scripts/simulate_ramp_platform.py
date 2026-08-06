"""
CAROLLINE ramp + flat-platform hybrid demonstration (MuJoCo viewer).

Course: flat approach -> incline ramp -> flat top -> decline ramp -> flat exit

Mission:
  Roll up the incline onto the flat top -> upright -> takeoff ->
  fly forward to the end of the flat top -> land ->
  roll down the decline -> roll back to the start -> IDLE

Uses the existing CarollineController unchanged. Mission logic only sequences
waypoints / modes and (during rolling on slopes) feeds path-following velocity
commands through the planner's rolling_velocity hook.

Usage (from repo root):
    python carolline_control/scripts/simulate_ramp_platform.py
    python carolline_control/scripts/simulate_ramp_platform.py --seed 7
    python carolline_control/scripts/simulate_ramp_platform.py --no-viewer
"""

from __future__ import annotations

import argparse
import enum
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.sim.viewer_loop import tune_viewer_for_speed
from carolline_control.scripts.manual_teleop import update_tracking_camera
from carolline_control.rolling_ramp.path import RampGeometry
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode, ControllerConfig, RobotState


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

@dataclass
class RampMissionLayout:
    geometry: RampGeometry
    spawn_xy: np.ndarray
    top_arrive_xy: np.ndarray       # roll onto flat top, then upright
    fly_end_xy: np.ndarray          # fly / land near end of flat top
    bottom_xy: np.ndarray           # after decline
    home_xy: np.ndarray
    top_arrive_s: float
    fly_end_s: float
    bottom_s: float
    home_s: float
    hover_height_nominal: float

    @property
    def top_z(self) -> float:
        return float(self.geometry.rise)

    def com_z_top(self) -> float:
        return self.top_z + float(self.geometry.cage_radius)

    def hover_clear_z(self) -> float:
        return max(self.hover_height_nominal, self.com_z_top() + 0.55)

    @classmethod
    def default(cls, cage_radius: float, hover_height: float) -> "RampMissionLayout":
        geom = RampGeometry(
            angle_deg=4.0,
            flat_start=1.5,
            ramp_length=3.5,
            flat_top=4.5,
            flat_end=3.0,
            width=5.0,
            y_center=0.0,
            cage_radius=cage_radius,
        )
        # Arclength milestones along the course
        top_arrive_s = geom.flat_start + geom.ramp_length + 0.9
        fly_end_s = geom.flat_start + geom.ramp_length + geom.flat_top - 1.0
        bottom_s = geom.flat_start + 2.0 * geom.ramp_length + geom.flat_top + 1.0
        home_s = 0.35

        def xy_at(s: float) -> np.ndarray:
            p = geom.position_at_s(s)
            return np.array([float(p[0]), float(p[1])], dtype=float)

        return cls(
            geometry=geom,
            spawn_xy=xy_at(home_s),
            top_arrive_xy=xy_at(top_arrive_s),
            fly_end_xy=xy_at(fly_end_s),
            bottom_xy=xy_at(bottom_s),
            home_xy=xy_at(home_s),
            top_arrive_s=top_arrive_s,
            fly_end_s=fly_end_s,
            bottom_s=bottom_s,
            home_s=home_s,
            hover_height_nominal=hover_height,
        )


# ---------------------------------------------------------------------------
# Slope-aware rolling assist (mission-side only)
# ---------------------------------------------------------------------------

class ArcPathAssist:
    """Drive along RampGeometry at full rolling_max_speed (no soft derates)."""

    def __init__(self, geometry: RampGeometry, config: ControllerConfig) -> None:
        self.geometry = geometry
        self._config = config
        self.goal_s = 0.0
        self._s_des = 0.0
        self._s_act = 0.0
        self._lookahead = 2.0
        self._cross_kp = 1.0
        self._cross_kd = 0.70

    def set_goal(self, goal_s: float, state: RobotState | None = None) -> None:
        self.goal_s = float(np.clip(goal_s, 0.0, self.geometry.total_length))
        if state is not None:
            self._s_act = self.geometry.project_s(state.position)
            self._s_des = self._s_act

    def update(self, state: RobotState, dt: float) -> None:
        self._s_act = self.geometry.project_s(state.position)
        vmax = float(self._config.rolling_max_speed)
        # Keep carrot ahead/behind at full cruise — no track-gated slowdown.
        advance = vmax * dt
        if self.goal_s >= self._s_act:
            cap = min(self.goal_s, self._s_act + self._lookahead)
            self._s_des = min(cap, self._s_des + advance, self.goal_s)
        else:
            cap = max(self.goal_s, self._s_act - self._lookahead)
            self._s_des = max(cap, self._s_des - advance, self.goal_s)

    def velocity_command_xy(self, state: RobotState) -> np.ndarray:
        """Command full rolling speed along the path (same motor budget as flight)."""
        self._s_act = self.geometry.project_s(state.position)
        tangent = self.geometry.tangent_at_s(self._s_act)
        remaining = self.goal_s - self._s_act
        vmax = float(self._config.rolling_max_speed)

        if abs(remaining) < self._config.roll_position_tolerance:
            v_along = 0.0
        else:
            # Full-speed cruise; taper only in the last half-metre.
            taper = min(1.0, abs(remaining) / 0.5)
            v_along = float(np.sign(remaining)) * vmax * max(0.4, taper)

        cross = float(state.position[1] - self.geometry.y_center)
        v_lat = -self._cross_kp * cross - self._cross_kd * float(state.velocity[1])
        v_world = v_along * tangent + np.array([0.0, v_lat, 0.0], dtype=float)
        speed = float(np.linalg.norm(v_world))
        if speed > vmax:
            v_world *= vmax / speed
        return v_world[:2]

    def near_goal(self, state: RobotState, tol: float | None = None) -> bool:
        tol = self._config.roll_position_tolerance if tol is None else tol
        self._s_act = self.geometry.project_s(state.position)
        return abs(self._s_act - self.goal_s) < tol and float(np.linalg.norm(state.velocity[:2])) < 0.12


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def _add_live_markers(spec: mujoco.MjSpec, layout: RampMissionLayout) -> None:
    world = spec.worldbody

    def add_marker(name: str, pos: list[float], rgba: list[float], size: float) -> None:
        body = world.add_body(name=name, mocap=True, pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    spawn = layout.geometry.position_at_s(layout.home_s)
    goal = layout.geometry.position_at_s(layout.fly_end_s)
    add_marker("marker_home", [float(spawn[0]), float(spawn[1]), float(spawn[2])], [0.20, 0.85, 0.30, 0.95], 0.08)
    add_marker("marker_fly_end", [float(goal[0]), float(goal[1]), float(goal[2] + 0.15)], [0.95, 0.45, 0.10, 0.95], 0.09)
    add_marker(
        "marker_current_target",
        [float(layout.top_arrive_xy[0]), float(layout.top_arrive_xy[1]), layout.com_z_top()],
        [0.15, 0.85, 0.95, 0.95],
        0.09,
    )


def compile_mission_scene(model_path: Path, layout: RampMissionLayout) -> mujoco.MjModel:
    """Build incline / flat-top / decline course with live mocap markers.

    Keeps the base floor so a cage that leaves the track still rests on ground.
    """
    from carolline_control.rolling_ramp.scene import _add_segment_box, _ramp_segments

    geometry = layout.geometry
    spec = mujoco.MjSpec.from_file(str(model_path))
    world = spec.worldbody

    yc = geometry.y_center
    width = geometry.width
    thickness = 0.08
    overlap = 0.40
    friction = [1.05, 0.015, 0.008]
    rgba_flat = [0.35, 0.45, 0.55, 1]
    rgba_up = [0.45, 0.55, 0.35, 1]
    rgba_top = [0.55, 0.50, 0.35, 1]
    rgba_down = [0.45, 0.40, 0.55, 1]

    l0 = geometry.flat_start + overlap
    _add_segment_box(
        world,
        "track_flat_start",
        [l0 * 0.5 - overlap * 0.5, yc, -thickness * 0.5],
        [l0 * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_flat,
        friction,
    )
    _ramp_segments(geometry, world, friction, rgba_up, direction="up")
    _ramp_segments(geometry, world, friction, rgba_down, direction="down")

    lt = geometry.flat_top + overlap
    x_top = geometry.flat_start + geometry.ramp_run + lt * 0.5
    _add_segment_box(
        world,
        "track_flat_top",
        [float(x_top), yc, geometry.rise - thickness * 0.5],
        [lt * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_top,
        friction,
    )

    le = geometry.flat_end + overlap
    x_end = geometry.flat_start + 2 * geometry.ramp_run + geometry.flat_top + le * 0.5
    _add_segment_box(
        world,
        "track_flat_end",
        [float(x_end), yc, -thickness * 0.5],
        [le * 0.5, width * 0.5, thickness * 0.5],
        [1, 0, 0, 0],
        rgba_flat,
        friction,
    )

    total_x = geometry.flat_start + 2 * geometry.ramp_run + geometry.flat_top + geometry.flat_end
    for side, y_sign in (("left", -1), ("right", 1)):
        world.add_geom(
            name=f"wall_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[total_x * 0.5, yc + y_sign * (width * 0.5 + 0.06), 0.45],
            size=[total_x * 0.55, 0.05, 0.45],
            rgba=[0.25, 0.25, 0.25, 0.35],
            friction=friction,
            contype=1,
            conaffinity=1,
            condim=3,
        )

    _add_live_markers(spec, layout)
    model = spec.compile()
    model.opt.timestep = 0.004
    return model


def _mocap_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return -1
    return int(model.body_mocapid[bid])


def set_target_marker(model: mujoco.MjModel, data: mujoco.MjData, xyz: np.ndarray) -> None:
    mid = _mocap_id(model, "marker_current_target")
    if mid >= 0:
        data.mocap_pos[mid] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]


# ---------------------------------------------------------------------------
# Mission director
# ---------------------------------------------------------------------------

class MissionPhase(enum.Enum):
    ROLL_UP = "ROLL_UP"
    RECOVER = "RECOVER"
    TAKEOFF = "TAKEOFF"
    FLY_ALONG_TOP = "FLY_ALONG_TOP"
    LAND_ON_TOP = "LAND_ON_TOP"
    ROLL_DOWN = "ROLL_DOWN"
    ROLL_HOME = "ROLL_HOME"
    RECOVER_ROLL = "RECOVER_ROLL"
    ROLL_ESCAPE = "ROLL_ESCAPE"
    DONE = "DONE"


class RampPlatformDirector:
    """Waypoint / mode sequencing for ramp + platform hybrid locomotion."""

    def __init__(
        self,
        controller: CarollineController,
        layout: RampMissionLayout,
        path: ArcPathAssist,
    ) -> None:
        self.ctrl = controller
        self.layout = layout
        self.path = path
        self.phase = MissionPhase.ROLL_UP
        self._phase_t0 = 0.0
        self._hover_ready_t: float | None = None
        self._land_hold = 0.0
        self._cfg = controller.config
        self._landing_height_nominal = float(self._cfg.landing_height)
        self._hover_height_nominal = float(self._cfg.hover_height)
        self._spawn_nominal = self._cfg.spawn_xy.copy()
        self._stock_rolling_velocity = controller.planner.rolling_velocity
        self._last_roll_s: float | None = None
        self._last_progress_t = 0.0
        self._resume_phase: MissionPhase | None = None
        self._resume_goal_s = 0.0
        self._escape_goal_s = 0.0
        self._recovery_count = 0

    def begin(self, t: float, state: RobotState) -> None:
        self._enable_path_rolling(True)
        self.path.set_goal(self.layout.top_arrive_s, state)
        self._set_roll_target(self.layout.top_arrive_xy)
        self._set_elevated_context(True)
        self.ctrl.mode_manager.request_roll()
        self._enter(MissionPhase.ROLL_UP, t, "Roll up incline onto flat top")

    def _enter(self, phase: MissionPhase, t: float, msg: str) -> None:
        self.phase = phase
        self._phase_t0 = t
        self._hover_ready_t = None
        self._land_hold = 0.0
        self._last_roll_s = None
        self._last_progress_t = t
        print(f"t={t:6.2f}s  PHASE -> {phase.value}  |  {msg}")

    def _enable_path_rolling(self, enabled: bool) -> None:
        if enabled:
            self.ctrl.planner.rolling_velocity = self.path.velocity_command_xy
        else:
            self.ctrl.planner.rolling_velocity = self._stock_rolling_velocity

    def _set_roll_target(self, xy: np.ndarray) -> None:
        self._cfg.roll_target = np.asarray(xy, dtype=float).copy()

    def _set_elevated_context(self, elevated: bool, land_xy: np.ndarray | None = None) -> None:
        """Set the flight landing target; contact height comes from MuJoCo."""
        if elevated:
            com = self.layout.com_z_top()
            self._cfg.landing_height = com
            if land_xy is not None:
                self._cfg.spawn_xy = np.asarray(land_xy, dtype=float).copy()
        else:
            self._cfg.landing_height = self._landing_height_nominal
            self._cfg.spawn_xy = self._spawn_nominal.copy()

    def _rolling_stuck(self, state: RobotState, t: float, goal_s: float) -> bool:
        """Detect commanded rolling with no useful along-track progress."""
        s = self.layout.geometry.project_s(state.position)
        if abs(goal_s - s) < 0.45:
            self._last_roll_s = s
            self._last_progress_t = t
            return False

        if self._last_roll_s is None:
            self._last_roll_s = s
            self._last_progress_t = t
            return False

        direction = 1.0 if goal_s >= self._last_roll_s else -1.0
        progress = direction * (s - self._last_roll_s)
        if progress >= 0.12:
            self._last_roll_s = s
            self._last_progress_t = t
            return False

        return t - self._last_progress_t >= 1.5

    def _begin_roll_recovery(
        self,
        state: RobotState,
        t: float,
        resume_phase: MissionPhase,
        resume_goal_s: float,
    ) -> None:
        """Run stock arbitrary-attitude recovery, then back away and retry."""
        self._resume_phase = resume_phase
        self._resume_goal_s = resume_goal_s
        self._recovery_count += 1

        current_s = self.layout.geometry.project_s(state.position)
        travel_direction = 1.0 if resume_goal_s >= current_s else -1.0
        # Back away from the obstacle after becoming upright.
        self._escape_goal_s = float(
            np.clip(current_s - travel_direction * 0.65, 0.15, self.layout.geometry.total_length - 0.15)
        )
        self._enable_path_rolling(False)
        self.ctrl.mode_manager.request_ground_recovery(resume_rolling=True)
        self._enter(
            MissionPhase.RECOVER_ROLL,
            t,
            f"Rolling obstruction detected - reorient (attempt {self._recovery_count})",
        )

    def _resume_roll_after_escape(self, state: RobotState, t: float) -> None:
        assert self._resume_phase is not None
        resume_phase = self._resume_phase
        resume_goal = self._resume_goal_s
        self.path.set_goal(resume_goal, state)
        self.ctrl.rolling.reset()
        self.ctrl.mode_manager.request_roll()
        self._enter(resume_phase, t, "Obstacle cleared - resume rolling")

    def _set_flight_waypoints(self, waypoints: list[list[float]], hover_z: float) -> None:
        self._cfg.hover_height = hover_z
        wps = [np.array(w, dtype=float) for w in waypoints]
        self.ctrl.planner._mission_waypoints = [w.copy() for w in wps]
        self.ctrl.planner._waypoints = [w.copy() for w in wps]
        self.ctrl.planner.reset()

    def _near_xy(self, state: RobotState, xy: np.ndarray, tol: float | None = None) -> bool:
        tol = self._cfg.roll_position_tolerance if tol is None else tol
        return float(np.linalg.norm(state.position[:2] - xy)) < tol

    def _top_settled(self, state: RobotState) -> bool:
        com = self.layout.com_z_top()
        return (
            abs(float(state.position[2]) - com) < 0.12
            and float(np.linalg.norm(state.velocity)) < 0.10
            and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold
            and self._near_xy(state, self.layout.fly_end_xy, tol=0.45)
        )

    def current_target_xyz(self) -> np.ndarray:
        layout = self.layout
        mapping = {
            MissionPhase.ROLL_UP: layout.geometry.position_at_s(layout.top_arrive_s),
            MissionPhase.RECOVER: layout.geometry.position_at_s(layout.top_arrive_s),
            MissionPhase.TAKEOFF: layout.geometry.position_at_s(layout.top_arrive_s),
            MissionPhase.FLY_ALONG_TOP: np.array(
                [layout.fly_end_xy[0], layout.fly_end_xy[1], layout.hover_clear_z()], dtype=float
            ),
            MissionPhase.LAND_ON_TOP: layout.geometry.position_at_s(layout.fly_end_s),
            MissionPhase.ROLL_DOWN: layout.geometry.position_at_s(layout.bottom_s),
            MissionPhase.ROLL_HOME: layout.geometry.position_at_s(layout.home_s),
            MissionPhase.RECOVER_ROLL: layout.geometry.position_at_s(self._escape_goal_s),
            MissionPhase.ROLL_ESCAPE: layout.geometry.position_at_s(self._escape_goal_s),
            MissionPhase.DONE: layout.geometry.position_at_s(layout.home_s),
        }
        return mapping[self.phase]

    def _start_fly_along_top(self, state: RobotState, t: float) -> None:
        hover = self.layout.hover_clear_z()
        end = self.layout.fly_end_xy
        self._enable_path_rolling(False)
        self._set_elevated_context(True, land_xy=end)
        self._set_flight_waypoints([[float(end[0]), float(end[1]), hover]], hover_z=hover)
        # Avoid planner prepending distant roll_target as a home waypoint
        self._set_roll_target(state.position[:2].copy())
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        self.ctrl.mode_manager.request_flight()
        self._enter(MissionPhase.FLY_ALONG_TOP, t, "Fly forward along flat platform")

    def update(self, state: RobotState, mode: ControlMode, t: float, dt: float) -> bool:
        mm = self.ctrl.mode_manager
        layout = self.layout
        hover = layout.hover_clear_z()

        if self.phase in (
            MissionPhase.ROLL_UP,
            MissionPhase.ROLL_DOWN,
            MissionPhase.ROLL_HOME,
            MissionPhase.ROLL_ESCAPE,
        ):
            self.path.update(state, dt)

        if self.phase == MissionPhase.ROLL_UP:
            self._set_roll_target(layout.top_arrive_xy)
            self.path.set_goal(layout.top_arrive_s)
            if self._rolling_stuck(state, t, layout.top_arrive_s):
                self._begin_roll_recovery(state, t, MissionPhase.ROLL_UP, layout.top_arrive_s)
                return True
            at_top = self.path.near_goal(state) or (
                self._near_xy(state, layout.top_arrive_xy, tol=0.25)
                and float(state.position[2]) > layout.com_z_top() - 0.15
                and float(np.linalg.norm(state.velocity[:2])) < 0.08
            )
            if at_top:
                self._enable_path_rolling(False)
                self._set_elevated_context(True, land_xy=layout.top_arrive_xy)
                mm.request_takeoff()
                self._enter(MissionPhase.RECOVER, t, "Upright on flat top")

        elif self.phase == MissionPhase.RECOVER:
            self._set_elevated_context(True, land_xy=layout.top_arrive_xy)
            if mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                self._enter(MissionPhase.TAKEOFF, t, "Takeoff from flat top")

        elif self.phase == MissionPhase.TAKEOFF:
            self._set_elevated_context(True, land_xy=layout.top_arrive_xy)
            self._cfg.hover_height = hover
            if mode == ControlMode.HOVER:
                if self._hover_ready_t is None:
                    self._hover_ready_t = t
                elif t - self._hover_ready_t >= 1.0:
                    self._start_fly_along_top(state, t)
            elif mode == ControlMode.FLIGHT:
                self._start_fly_along_top(state, t)

        elif self.phase == MissionPhase.FLY_ALONG_TOP:
            self._set_elevated_context(True, land_xy=layout.fly_end_xy)
            self._cfg.hover_height = hover
            if mode == ControlMode.LANDING and not (
                self._near_xy(state, layout.fly_end_xy, tol=0.40)
                and abs(float(state.position[2]) - hover) < 0.45
            ):
                mm.request_flight()
            if self._near_xy(state, layout.fly_end_xy, tol=0.35) and abs(
                float(state.position[2]) - hover
            ) < 0.40:
                self._set_elevated_context(True, land_xy=layout.fly_end_xy)
                mm.request_landing()
                self._enter(MissionPhase.LAND_ON_TOP, t, "Land near end of flat platform")

        elif self.phase == MissionPhase.LAND_ON_TOP:
            self._set_elevated_context(True, land_xy=layout.fly_end_xy)
            if self._top_settled(state) or mode == ControlMode.IDLE:
                self._land_hold += dt
            else:
                self._land_hold = 0.0
            if self._land_hold >= 1.0:
                self._enable_path_rolling(True)
                self.path.set_goal(layout.bottom_s, state)
                self._set_roll_target(layout.bottom_xy)
                self.ctrl.rolling.reset()
                mm.request_roll()
                self._enter(MissionPhase.ROLL_DOWN, t, "Roll down decline ramp")

        elif self.phase == MissionPhase.ROLL_DOWN:
            # Keep roll_target slightly ahead so mode manager does not settle early.
            s_act = layout.geometry.project_s(state.position)
            ahead = layout.geometry.position_at_s(min(layout.bottom_s, s_act + 0.8))
            self._set_roll_target(ahead[:2])
            self.path.set_goal(layout.bottom_s)
            if self._rolling_stuck(state, t, layout.bottom_s):
                self._begin_roll_recovery(state, t, MissionPhase.ROLL_DOWN, layout.bottom_s)
                return True
            bottom_z = float(layout.geometry.position_at_s(layout.bottom_s)[2])
            at_bottom = (
                self.path.near_goal(state, tol=0.30)
                or self._near_xy(state, layout.bottom_xy, tol=0.30)
            ) and abs(float(state.position[2]) - bottom_z) < 0.18
            if at_bottom:
                self.path.set_goal(layout.home_s, state)
                self._set_roll_target(layout.home_xy)
                self.ctrl.rolling.reset()
                mm.request_roll()
                self._enter(MissionPhase.ROLL_HOME, t, "Roll back along course to start")

        elif self.phase == MissionPhase.ROLL_HOME:
            s_act = layout.geometry.project_s(state.position)
            ahead = layout.geometry.position_at_s(max(layout.home_s, s_act - 0.8))
            self._set_roll_target(ahead[:2])
            self.path.set_goal(layout.home_s)
            if self._rolling_stuck(state, t, layout.home_s):
                self._begin_roll_recovery(state, t, MissionPhase.ROLL_HOME, layout.home_s)
                return True
            home_z = float(layout.geometry.position_at_s(layout.home_s)[2])
            at_home = (
                self.path.near_goal(state, tol=0.25)
                and abs(float(state.position[2]) - home_z) < 0.15
                and abs(float(state.position[1])) < 0.45
            )
            if at_home:
                self._enable_path_rolling(False)
                self._set_elevated_context(False)
                mm.request_idle()
                self._enter(MissionPhase.DONE, t, "Home reached - IDLE")
                return False

        elif self.phase == MissionPhase.RECOVER_ROLL:
            # PRETAKEOFF handles arbitrary cage orientation using the existing
            # SO(3) recovery controller. UPRIGHT requests rolling, not takeoff.
            if mode == ControlMode.UPRIGHT:
                mm.request_roll()
            elif mode == ControlMode.ROLLING:
                self._enable_path_rolling(True)
                self.path.set_goal(self._escape_goal_s, state)
                escape_xy = layout.geometry.position_at_s(self._escape_goal_s)[:2]
                self._set_roll_target(escape_xy)
                self.ctrl.rolling.reset()
                self._enter(MissionPhase.ROLL_ESCAPE, t, "Reoriented - back away from obstruction")

        elif self.phase == MissionPhase.ROLL_ESCAPE:
            escape_xy = layout.geometry.position_at_s(self._escape_goal_s)[:2]
            self._set_roll_target(escape_xy)
            self.path.set_goal(self._escape_goal_s)
            escaped = self.path.near_goal(state, tol=0.25)
            if escaped or t - self._phase_t0 >= 2.0:
                self._resume_roll_after_escape(state, t)

        elif self.phase == MissionPhase.DONE:
            mm.request_idle()
            return False

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE ramp + platform hybrid demo")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "config.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--test-stuck",
        action="store_true",
        help="Inject one brief obstruction to verify automatic recovery",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw = load_raw_config(args.config)
    config = load_config(args.config)

    layout = RampMissionLayout.default(
        cage_radius=float(config.cage_radius),
        hover_height=float(config.hover_height),
    )

    config.initial_mode = ControlMode.ROLLING
    config.spawn_xy = layout.spawn_xy.copy()
    config.roll_target = layout.top_arrive_xy.copy()
    config.waypoints = [
        [float(layout.fly_end_xy[0]), float(layout.fly_end_xy[1]), float(layout.hover_clear_z())]
    ]

    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    model = compile_mission_scene(model_path, layout)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)

    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    path = ArcPathAssist(layout.geometry, config)
    director = RampPlatformDirector(controller, layout, path)

    spawn = layout.geometry.position_at_s(layout.home_s)
    tilt_deg = 50.0
    half = np.radians(tilt_deg) * 0.5
    quat = [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]
    data.qpos[:7] = [float(spawn[0]), float(spawn[1]), float(spawn[2]), *quat]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    state0 = estimator.estimate(data)
    director.begin(0.0, state0)

    dt = float(model.opt.timestep)
    last_mode_name = controller.mode_manager.mode.name
    print("Ramp + platform hybrid demo")
    print(
        f"  course: rise={layout.geometry.rise:.2f}m  angle={layout.geometry.angle_deg:.1f}deg  "
        f"length={layout.geometry.total_length:.1f}m"
    )
    print(f"  top arrive x={layout.top_arrive_xy[0]:.2f}  fly end x={layout.fly_end_xy[0]:.2f}")
    print(f"  mode: {last_mode_name}")
    print("-" * 60)

    telemetry = {
        "mode": last_mode_name,
        "requested_moment": np.zeros(3),
        "applied_moment": np.zeros(3),
        "motors": np.zeros(4),
        "contact_normal": np.array([0.0, 0.0, 1.0]),
        "allocation_scale": 1.0,
        "allocation_saturated": False,
        "max_applied_moment": 0.0,
        "max_motor": 0.0,
    }
    stuck_test_pose = data.qpos[:7].copy()

    def control_step(*, step_dt: float | None = None) -> bool:
        nonlocal last_mode_name
        use_dt = dt if step_dt is None else step_dt
        if args.test_stuck and 0.5 <= float(data.time) <= 2.5:
            # Verification-only synthetic obstruction. Normal viewer runs never
            # alter pose and use physical contacts exclusively.
            data.qpos[:7] = stuck_test_pose
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        state = estimator.estimate(data)
        motor, cmd, mode, target = controller.compute(state, use_dt)
        cont = director.update(state, mode, float(state.time), use_dt)

        diagnostics = controller.last_diagnostics
        requested_moment = diagnostics.moment_body.copy()
        if diagnostics.ground_allocation_active:
            applied_moment = diagnostics.achievable_contact_torque.copy()
        else:
            achieved_wrench = controller.mixer._alloc @ np.asarray(motor.thrusts, dtype=float)
            applied_moment = achieved_wrench[1:].copy()
        telemetry["mode"] = mode.name
        telemetry["requested_moment"] = requested_moment
        telemetry["applied_moment"] = applied_moment
        telemetry["motors"] = np.asarray(motor.thrusts, dtype=float).copy()
        telemetry["contact_normal"] = diagnostics.contact_normal_world.copy()
        telemetry["allocation_scale"] = diagnostics.allocation_scale
        telemetry["allocation_saturated"] = diagnostics.allocation_scale > 1.0
        telemetry["max_applied_moment"] = max(
            float(telemetry["max_applied_moment"]),
            float(np.linalg.norm(applied_moment)),
        )
        telemetry["max_motor"] = max(
            float(telemetry["max_motor"]),
            float(np.max(np.abs(motor.thrusts))),
        )

        if mode.name != last_mode_name:
            print(
                f"t={state.time:6.2f}s  MODE  {last_mode_name} -> {mode.name}  "
                f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})"
            )
            last_mode_name = mode.name

        set_target_marker(model, data, director.current_target_xyz())
        data.ctrl[:] = motor.thrusts
        return cont

    done_since: float | None = None
    cam_look = layout.geometry.position_at_s(layout.geometry.total_length * 0.45)

    def maybe_finish(t: float) -> bool:
        nonlocal done_since
        if director.phase != MissionPhase.DONE:
            return False
        if done_since is None:
            done_since = t
            return False
        if t - done_since > 2.0:
            print(f"t={t:6.2f}s  Mission complete.")
            return True
        return False

    if args.no_viewer:
        while data.time < args.duration:
            still = control_step()
            if (not still or director.phase == MissionPhase.DONE) and maybe_finish(float(data.time)):
                break
            mujoco.mj_step(model, data)
    else:
        viewer_substeps = 20
        frame_dt = dt * viewer_substeps
        with mujoco.viewer.launch_passive(model, data) as viewer:
            tune_viewer_for_speed(viewer)
            viewer.cam.azimuth = 115.0
            viewer.cam.elevation = -18.0
            track_distance = 10.0
            update_tracking_camera(viewer, data.qpos[:3], look_height=0.35, distance=track_distance)
            while viewer.is_running() and data.time < args.duration:
                frame_start = time.perf_counter()
                still = control_step(step_dt=frame_dt)
                update_tracking_camera(viewer, data.qpos[:3], look_height=0.35, distance=track_distance)
                req = np.asarray(telemetry["requested_moment"])
                applied = np.asarray(telemetry["applied_moment"])
                motors = np.asarray(telemetry["motors"])
                normal = np.asarray(telemetry["contact_normal"])
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        (
                            f"Mode: {telemetry['mode']}\n"
                            f"Phase: {director.phase.value}\n"
                            "Body torque [N.m]\n"
                            "Requested\n"
                            "Applied\n"
                            "Motor thrust [N]\n"
                            "F1/F2\n"
                            "F3/F4\n"
                            "Contact normal / allocation"
                        ),
                        (
                            "\n\n"
                            f"|M|={np.linalg.norm(req):5.2f}  "
                            f"[{req[0]:+5.2f}, {req[1]:+5.2f}, {req[2]:+5.2f}]\n"
                            f"|M|={np.linalg.norm(applied):5.2f}  "
                            f"[{applied[0]:+5.2f}, {applied[1]:+5.2f}, {applied[2]:+5.2f}]\n"
                            "\n"
                            f"{motors[0]:+5.2f} / {motors[1]:+5.2f}\n"
                            f"{motors[2]:+5.2f} / {motors[3]:+5.2f}\n"
                            f"[{normal[0]:+4.2f}, {normal[1]:+4.2f}, {normal[2]:+4.2f}]  "
                            f"scale={float(telemetry['allocation_scale']):4.2f}  "
                            f"sat={'YES' if telemetry['allocation_saturated'] else 'no'}"
                        ),
                    )
                )
                if (not still or director.phase == MissionPhase.DONE) and maybe_finish(float(data.time)):
                    break
                for _ in range(viewer_substeps):
                    mujoco.mj_step(model, data)
                viewer.sync(state_only=True)
                elapsed = time.perf_counter() - frame_start
                sleep_s = (1.0 / 30.0) - elapsed
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    print(f"Final mode: {controller.mode_manager.mode.name}  phase={director.phase.value}")
    print(
        f"Peak applied body torque: {float(telemetry['max_applied_moment']):.2f} N.m  "
        f"peak motor thrust: {float(telemetry['max_motor']):.2f} N"
    )


if __name__ == "__main__":
    main()
