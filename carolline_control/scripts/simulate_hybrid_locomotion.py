"""
CAROLLINE hybrid locomotion demonstration (MuJoCo viewer).

Sequences the existing controllers through rolling + flight over a raised
platform without modifying controller code, gains, or the state machine.

Mission:
  Roll to platform -> Takeoff -> Fly onto platform -> Land ->
  Roll across platform -> Takeoff -> Descend to ground -> Land ->
  Roll to final goal -> IDLE

Usage (from repo root):
    python carolline_control/scripts/simulate_hybrid_locomotion.py
    python carolline_control/scripts/simulate_hybrid_locomotion.py --seed 7
    python carolline_control/scripts/simulate_hybrid_locomotion.py --no-viewer
"""

from __future__ import annotations

import argparse
import enum
import sys
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
from carolline_control.utils.so3 import body_z_world
from carolline_control.utils.types import ControlMode, ControllerConfig


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class PlatformSpec:
    """Raised box platform along +X."""

    x_center: float = 6.0
    y_center: float = 0.0
    length: float = 4.0
    width: float = 3.0
    height: float = 0.45

    @property
    def x_min(self) -> float:
        return self.x_center - 0.5 * self.length

    @property
    def x_max(self) -> float:
        return self.x_center + 0.5 * self.length

    @property
    def top_z(self) -> float:
        return self.height

    def com_z(self, cage_radius: float) -> float:
        return self.top_z + cage_radius

    def contains_xy(self, xy: np.ndarray, margin: float = 0.15) -> bool:
        return (
            self.x_min + margin <= float(xy[0]) <= self.x_max - margin
            and abs(float(xy[1]) - self.y_center) <= 0.5 * self.width - margin
        )


@dataclass
class MissionLayout:
    spawn_xy: np.ndarray
    approach_xy: np.ndarray          # roll target before platform
    platform_land_xy: np.ndarray     # land / hover over platform
    platform_far_xy: np.ndarray      # roll across to far edge
    ground_land_xy: np.ndarray       # descend / land past platform
    final_goal_xy: np.ndarray
    platform: PlatformSpec
    ground_z: float
    hover_height: float
    cage_radius: float

    @classmethod
    def default(cls, cage_radius: float, hover_height: float, ground_z: float) -> "MissionLayout":
        plat = PlatformSpec()
        return cls(
            spawn_xy=np.array([0.0, 0.0], dtype=float),
            approach_xy=np.array([plat.x_min - 0.8, 0.0], dtype=float),
            platform_land_xy=np.array([plat.x_center - 0.3, 0.0], dtype=float),
            # Keep far roll target inward so PRETAKEOFF starts before the lip.
            platform_far_xy=np.array([plat.x_max - 1.2, 0.0], dtype=float),
            ground_land_xy=np.array([plat.x_max + 1.5, 0.0], dtype=float),
            final_goal_xy=np.array([plat.x_max + 4.0, 0.0], dtype=float),
            platform=plat,
            ground_z=ground_z,
            hover_height=hover_height,
            cage_radius=cage_radius,
        )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def compile_hybrid_scene(
    model_path: str | Path,
    layout: MissionLayout,
    config: ControllerConfig,
) -> mujoco.MjModel:
    """Load base scene, add platform + mocap markers for goals."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    world = spec.worldbody
    plat = layout.platform

    world.add_geom(
        name="hybrid_platform",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[plat.x_center, plat.y_center, 0.5 * plat.height],
        size=[0.5 * plat.length, 0.5 * plat.width, 0.5 * plat.height],
        rgba=[0.55, 0.48, 0.35, 1.0],
        friction=[0.9, 0.02, 0.01],
        contype=1,
        conaffinity=1,
        condim=6,
    )

    def add_marker(name: str, pos: list[float], rgba: list[float], size: float) -> None:
        body = world.add_body(name=name, mocap=True, pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
        )

    # Final goal (orange) and current target (cyan) — updated live via mocap
    add_marker(
        "marker_final_goal",
        [float(layout.final_goal_xy[0]), float(layout.final_goal_xy[1]), 0.12],
        [0.95, 0.45, 0.10, 0.95],
        0.10,
    )
    add_marker(
        "marker_current_target",
        [float(layout.approach_xy[0]), float(layout.approach_xy[1]), 0.12],
        [0.15, 0.85, 0.95, 0.95],
        0.09,
    )
    add_marker(
        "marker_spawn",
        [float(layout.spawn_xy[0]), float(layout.spawn_xy[1]), 0.10],
        [0.20, 0.85, 0.30, 0.90],
        0.07,
    )

    model = spec.compile()
    # Match runtime timestep used by the existing stack when XML says 0.004
    return model


def _mocap_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return -1
    # mocap body index = body_mocapid[bid]
    return int(model.body_mocapid[bid])


def set_target_marker(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray, z: float = 0.12) -> None:
    mid = _mocap_id(model, "marker_current_target")
    if mid >= 0:
        data.mocap_pos[mid] = [float(xy[0]), float(xy[1]), float(z)]


# ---------------------------------------------------------------------------
# Mission phases (external sequencing only)
# ---------------------------------------------------------------------------

class MissionPhase(enum.Enum):
    ROLL_TO_PLATFORM = "ROLL_TO_PLATFORM"
    RECOVER_FOR_TAKEOFF = "RECOVER_FOR_TAKEOFF"
    TAKEOFF_1 = "TAKEOFF_1"
    FLY_TO_PLATFORM = "FLY_TO_PLATFORM"
    LAND_ON_PLATFORM = "LAND_ON_PLATFORM"
    ROLL_ACROSS_PLATFORM = "ROLL_ACROSS_PLATFORM"
    RECOVER_AT_EDGE = "RECOVER_AT_EDGE"
    TAKEOFF_2 = "TAKEOFF_2"
    FLY_TO_GROUND = "FLY_TO_GROUND"
    LAND_ON_GROUND = "LAND_ON_GROUND"
    ROLL_TO_GOAL = "ROLL_TO_GOAL"
    DONE = "DONE"


class HybridMissionDirector:
    """Waypoint / mode sequencing around the existing CarollineController."""

    def __init__(self, controller: CarollineController, layout: MissionLayout) -> None:
        self.ctrl = controller
        self.layout = layout
        self.phase = MissionPhase.ROLL_TO_PLATFORM
        self._phase_t0 = 0.0
        self._hover_ready_t: float | None = None
        self._land_hold = 0.0
        self._cfg = controller.config
        self._landing_height_nominal = float(self._cfg.landing_height)
        self._hover_height_nominal = float(self._cfg.hover_height)
        self._spawn_nominal = self._cfg.spawn_xy.copy()

    def begin(self, t: float) -> None:
        self._enter(MissionPhase.ROLL_TO_PLATFORM, t, "Roll toward platform approach")

    def _enter(self, phase: MissionPhase, t: float, msg: str) -> None:
        self.phase = phase
        self._phase_t0 = t
        self._hover_ready_t = None
        self._land_hold = 0.0
        print(f"t={t:6.2f}s  PHASE -> {phase.value}  |  {msg}")

    def _set_roll_target(self, xy: np.ndarray) -> None:
        self._cfg.roll_target = np.asarray(xy, dtype=float).copy()

    def _set_ground_context(self, *, on_platform: bool) -> None:
        """Set landing target; support height is measured from MuJoCo contact."""
        if on_platform:
            com_z = self.layout.platform.com_z(self.layout.cage_radius)
            self._cfg.landing_height = com_z
            self._cfg.spawn_xy = self.layout.platform_land_xy.copy()
        else:
            self._cfg.landing_height = self._landing_height_nominal
            self._cfg.spawn_xy = self._spawn_nominal.copy()

    def _set_flight_waypoints(self, waypoints: list[list[float]], hover_z: float) -> None:
        self._cfg.hover_height = hover_z
        wps = [np.array(w, dtype=float) for w in waypoints]
        self.ctrl.planner._mission_waypoints = [w.copy() for w in wps]
        self.ctrl.planner._waypoints = [w.copy() for w in wps]
        self.ctrl.planner.reset()

    def _near_xy(self, state, xy: np.ndarray, tol: float | None = None) -> bool:
        tol = self._cfg.roll_position_tolerance if tol is None else tol
        return float(np.linalg.norm(state.position[:2] - xy)) < tol

    def _platform_settled(self, state) -> bool:
        plat = self.layout.platform
        com_z = plat.com_z(self.layout.cage_radius)
        return (
            plat.contains_xy(state.position[:2], margin=0.25)
            and abs(float(state.position[2]) - com_z) < 0.10
            and float(np.linalg.norm(state.velocity)) < 0.10
            and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold
        )

    def _ground_settled(self, state) -> bool:
        return (
            state.on_ground
            and float(np.linalg.norm(state.velocity)) < 0.08
            and float(body_z_world(state.rotation)[2]) >= self._cfg.upright_cos_threshold
        )

    def current_target_xy(self) -> np.ndarray:
        mapping = {
            MissionPhase.ROLL_TO_PLATFORM: self.layout.approach_xy,
            MissionPhase.RECOVER_FOR_TAKEOFF: self.layout.approach_xy,
            MissionPhase.TAKEOFF_1: self.layout.approach_xy,
            MissionPhase.FLY_TO_PLATFORM: self.layout.platform_land_xy,
            MissionPhase.LAND_ON_PLATFORM: self.layout.platform_land_xy,
            MissionPhase.ROLL_ACROSS_PLATFORM: self.layout.platform_far_xy,
            MissionPhase.RECOVER_AT_EDGE: self.layout.platform_far_xy,
            MissionPhase.TAKEOFF_2: self.layout.platform_far_xy,
            MissionPhase.FLY_TO_GROUND: self.layout.ground_land_xy,
            MissionPhase.LAND_ON_GROUND: self.layout.ground_land_xy,
            MissionPhase.ROLL_TO_GOAL: self.layout.final_goal_xy,
            MissionPhase.DONE: self.layout.final_goal_xy,
        }
        return mapping[self.phase]

    def _hover_clear_z(self) -> float:
        plat = self.layout.platform
        return max(self._hover_height_nominal, plat.top_z + self.layout.cage_radius + 0.55)

    def _start_fly_to_platform(self, state, t: float) -> None:
        hover_clear = self._hover_clear_z()
        layout = self.layout
        mm = self.ctrl.mode_manager
        self._set_flight_waypoints(
            [
                [
                    float(layout.platform_land_xy[0]),
                    float(layout.platform_land_xy[1]),
                    hover_clear,
                ]
            ],
            hover_z=hover_clear,
        )
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        mm.request_flight()
        self._enter(MissionPhase.FLY_TO_PLATFORM, t, "Fly onto platform")

    def _start_fly_to_ground(self, state, t: float) -> None:
        layout = self.layout
        mm = self.ctrl.mode_manager
        self._set_ground_context(on_platform=False)
        self._cfg.hover_height = self._hover_height_nominal
        self._set_flight_waypoints(
            [
                [
                    float(layout.ground_land_xy[0]),
                    float(layout.ground_land_xy[1]),
                    self._hover_height_nominal,
                ]
            ],
            hover_z=self._hover_height_nominal,
        )
        self.ctrl.planner.begin_flight(state, yaw=self.ctrl._flight_yaw)
        mm.request_flight()
        self._enter(MissionPhase.FLY_TO_GROUND, t, "Descend / fly back to ground")

    def update(self, state, mode: ControlMode, t: float, dt: float) -> bool:
        """Advance mission phase. Returns False when mission should end soon."""
        mm = self.ctrl.mode_manager
        layout = self.layout
        hover_clear = self._hover_clear_z()

        if self.phase == MissionPhase.ROLL_TO_PLATFORM:
            # Do not force ROLLING every frame — that fights natural PRETAKEOFF.
            self._set_ground_context(on_platform=False)
            self._set_roll_target(layout.approach_xy)
            at_approach = (
                self._near_xy(state, layout.approach_xy)
                and float(np.linalg.norm(state.velocity[:2])) < 0.08
            )
            if mode == ControlMode.PRETAKEOFF or (
                mode in (ControlMode.UPRIGHT, ControlMode.TAKEOFF) and at_approach
            ):
                self._enter(MissionPhase.RECOVER_FOR_TAKEOFF, t, "Recover upright before first takeoff")
            elif mode == ControlMode.ROLLING and at_approach:
                mm.request_takeoff()
                self._enter(MissionPhase.RECOVER_FOR_TAKEOFF, t, "Recover upright before first takeoff")

        elif self.phase == MissionPhase.RECOVER_FOR_TAKEOFF:
            self._set_ground_context(on_platform=False)
            if mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                self._enter(MissionPhase.TAKEOFF_1, t, "First takeoff")

        elif self.phase == MissionPhase.TAKEOFF_1:
            self._set_ground_context(on_platform=False)
            if mode == ControlMode.HOVER:
                if self._hover_ready_t is None:
                    self._hover_ready_t = t
                elif t - self._hover_ready_t >= 1.0:
                    self._start_fly_to_platform(state, t)
            elif mode == ControlMode.FLIGHT:
                # Mode manager advanced HOVER->FLIGHT after ~3s; redirect waypoints.
                self._start_fly_to_platform(state, t)

        elif self.phase == MissionPhase.FLY_TO_PLATFORM:
            self._set_ground_context(on_platform=False)
            # Hold FLIGHT until we deliberately land (planner may flag mission_complete).
            if mode == ControlMode.LANDING and not (
                self._near_xy(state, layout.platform_land_xy, tol=0.40)
                and abs(float(state.position[2]) - hover_clear) < 0.45
            ):
                mm.request_flight()
            if self._near_xy(state, layout.platform_land_xy, tol=0.35) and abs(
                float(state.position[2]) - hover_clear
            ) < 0.40:
                self._set_ground_context(on_platform=True)
                self._cfg.spawn_xy = layout.platform_land_xy.copy()
                mm.request_landing()
                self._enter(MissionPhase.LAND_ON_PLATFORM, t, "Land on platform")

        elif self.phase == MissionPhase.LAND_ON_PLATFORM:
            self._set_ground_context(on_platform=True)
            self._cfg.spawn_xy = layout.platform_land_xy.copy()
            if self._platform_settled(state) or mode == ControlMode.IDLE:
                self._land_hold += dt
            else:
                self._land_hold = 0.0
            if self._land_hold >= 1.0:
                self._set_roll_target(layout.platform_far_xy)
                self.ctrl.rolling.reset()
                mm.request_roll()
                self._enter(MissionPhase.ROLL_ACROSS_PLATFORM, t, "Roll across platform")

        elif self.phase == MissionPhase.ROLL_ACROSS_PLATFORM:
            self._set_ground_context(on_platform=True)
            self._set_roll_target(layout.platform_far_xy)
            at_edge = (
                self._near_xy(state, layout.platform_far_xy)
                and float(np.linalg.norm(state.velocity[:2])) < 0.08
            )
            if mode == ControlMode.PRETAKEOFF or (
                mode in (ControlMode.UPRIGHT, ControlMode.TAKEOFF) and at_edge
            ):
                self._enter(MissionPhase.RECOVER_AT_EDGE, t, "Platform edge reached - recover for takeoff")
            elif mode == ControlMode.ROLLING and at_edge:
                mm.request_takeoff()
                self._enter(MissionPhase.RECOVER_AT_EDGE, t, "Platform edge reached - recover for takeoff")

        elif self.phase == MissionPhase.RECOVER_AT_EDGE:
            self._set_ground_context(on_platform=True)
            if mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                self._enter(MissionPhase.TAKEOFF_2, t, "Second takeoff from platform edge")

        elif self.phase == MissionPhase.TAKEOFF_2:
            self._set_ground_context(on_platform=True)
            if mode == ControlMode.HOVER:
                if self._hover_ready_t is None:
                    self._hover_ready_t = t
                elif t - self._hover_ready_t >= 1.0:
                    self._start_fly_to_ground(state, t)
            elif mode == ControlMode.FLIGHT:
                self._start_fly_to_ground(state, t)

        elif self.phase == MissionPhase.FLY_TO_GROUND:
            self._set_ground_context(on_platform=False)
            self._cfg.hover_height = self._hover_height_nominal
            if mode == ControlMode.LANDING and not (
                self._near_xy(state, layout.ground_land_xy, tol=0.35)
                and abs(float(state.position[2]) - self._hover_height_nominal) < 0.40
            ):
                mm.request_flight()
            if self._near_xy(state, layout.ground_land_xy, tol=0.30) and abs(
                float(state.position[2]) - self._hover_height_nominal
            ) < 0.30:
                self._cfg.spawn_xy = layout.ground_land_xy.copy()
                self._cfg.landing_height = self._landing_height_nominal
                mm.request_landing()
                self._enter(MissionPhase.LAND_ON_GROUND, t, "Land on ground past platform")

        elif self.phase == MissionPhase.LAND_ON_GROUND:
            self._set_ground_context(on_platform=False)
            self._cfg.spawn_xy = layout.ground_land_xy.copy()
            if mode == ControlMode.IDLE or self._ground_settled(state):
                self._land_hold += dt
            else:
                self._land_hold = 0.0
            if self._land_hold >= 1.0:
                self._set_roll_target(layout.final_goal_xy)
                self.ctrl.rolling.reset()
                mm.request_roll()
                self._enter(MissionPhase.ROLL_TO_GOAL, t, "Roll to final goal")

        elif self.phase == MissionPhase.ROLL_TO_GOAL:
            self._set_ground_context(on_platform=False)
            self._set_roll_target(layout.final_goal_xy)
            at_goal = (
                self._near_xy(state, layout.final_goal_xy)
                and float(np.linalg.norm(state.velocity[:2])) < 0.06
            )
            if at_goal and mode in (ControlMode.ROLLING, ControlMode.PRETAKEOFF, ControlMode.IDLE):
                mm.request_idle()
                self._enter(MissionPhase.DONE, t, "Final goal reached - IDLE")
                return False

        elif self.phase == MissionPhase.DONE:
            mm.request_idle()
            return False

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE hybrid locomotion viewer demo")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "carolline_control" / "config.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Headless run (for smoke tests); still prints phase/mode transitions",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    raw = load_raw_config(args.config)
    config = load_config(args.config)
    ground_z = float(raw.get("ground_z", 0.40))
    layout = MissionLayout.default(
        cage_radius=float(config.cage_radius),
        hover_height=float(config.hover_height),
        ground_z=ground_z,
    )

    # Seed mission fields the controller already uses
    config.initial_mode = ControlMode.ROLLING
    config.spawn_xy = layout.spawn_xy.copy()
    config.roll_target = layout.approach_xy.copy()
    config.waypoints = [
        [float(layout.platform_land_xy[0]), float(layout.platform_land_xy[1]), float(config.hover_height)]
    ]

    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model = compile_hybrid_scene(model_path, layout, config)
    data = mujoco.MjData(model)
    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)
    controller.mode_manager.request_roll()

    # Spawn on ground before platform (mild upright-leaning roll attitude)
    tilt_deg = 55.0
    half = np.radians(tilt_deg) * 0.5
    # Tilt about +Y so rolling along +X is natural
    quat = [float(np.cos(half)), 0.0, float(np.sin(half)), 0.0]
    data.qpos[:7] = [
        float(layout.spawn_xy[0]),
        float(layout.spawn_xy[1]),
        ground_z,
        *quat,
    ]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    director = HybridMissionDirector(controller, layout)
    director.begin(0.0)

    dt = float(model.opt.timestep)
    last_mode_name = controller.mode_manager.mode.name
    print("Hybrid locomotion demo")
    print(
        f"  platform: x=[{layout.platform.x_min:.1f},{layout.platform.x_max:.1f}]  "
        f"height={layout.platform.height:.2f}m"
    )
    print(f"  final goal: ({layout.final_goal_xy[0]:.1f}, {layout.final_goal_xy[1]:.1f})")
    print(f"  mode: {last_mode_name}")
    print("-" * 60)

    def control_step() -> bool:
        nonlocal last_mode_name
        state = estimator.estimate(data)
        motor, cmd, mode, target = controller.compute(state, dt)
        cont = director.update(state, mode, float(state.time), dt)

        if mode.name != last_mode_name:
            print(
                f"t={state.time:6.2f}s  MODE  {last_mode_name} -> {mode.name}  "
                f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})"
            )
            last_mode_name = mode.name

        tgt_z = 0.12
        if mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.TAKEOFF, ControlMode.LANDING):
            tgt_z = float(target.position[2])
        set_target_marker(model, data, director.current_target_xy(), z=tgt_z)
        data.ctrl[:] = motor.thrusts
        return cont

    done_since: float | None = None

    def run_until_done(step_fn) -> None:
        nonlocal done_since
        while data.time < args.duration:
            still_running = control_step()
            if not still_running or director.phase == MissionPhase.DONE:
                if done_since is None:
                    done_since = float(data.time)
                elif float(data.time) - done_since > 2.0:
                    print(f"t={data.time:6.2f}s  Mission complete.")
                    break
            step_fn()

    if args.no_viewer:
        run_until_done(lambda: mujoco.mj_step(model, data))
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [layout.platform.x_center, 0.0, 0.6]
            viewer.cam.distance = 14.0
            viewer.cam.azimuth = 120.0
            viewer.cam.elevation = -20.0

            def _viewer_step() -> None:
                mujoco.mj_step(model, data)
                viewer.sync()

            while viewer.is_running() and data.time < args.duration:
                still_running = control_step()
                if not still_running or director.phase == MissionPhase.DONE:
                    if done_since is None:
                        done_since = float(data.time)
                    elif float(data.time) - done_since > 2.0:
                        print(f"t={data.time:6.2f}s  Mission complete.")
                        break
                _viewer_step()

    print(f"Final mode: {controller.mode_manager.mode.name}  phase={director.phase.value}")


if __name__ == "__main__":
    main()
