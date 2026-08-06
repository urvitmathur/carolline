"""Manual CAROLLINE test rig — replicate paper/video demos with keyboard control.

Hold numpad keys to move; release to stop.  MuJoCo viewer UI enabled by default.

Run from repo root:
    python carolline_control/scripts/manual_teleop.py
    python carolline_control/scripts/manual_teleop.py --scene ramp
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import glfw
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
from carolline_control.utils.types import ControlMode, ControllerConfig
from carolline_control.visualization.markers import compile_model_with_markers

# Movement — arrow keys (+ numpad); WASD are used by the MuJoCo viewer
FORWARD_KEYS = {glfw.KEY_UP, glfw.KEY_KP_8}
BACK_KEYS = {glfw.KEY_DOWN, glfw.KEY_KP_2}
LEFT_KEYS = {glfw.KEY_LEFT, glfw.KEY_KP_4}
RIGHT_KEYS = {glfw.KEY_RIGHT, glfw.KEY_KP_6}
UP_KEYS = {glfw.KEY_PAGE_UP, glfw.KEY_KP_9}
DOWN_KEYS = {glfw.KEY_PAGE_DOWN, glfw.KEY_KP_3}
YAW_LEFT_KEYS = {glfw.KEY_Q, glfw.KEY_KP_7}
YAW_RIGHT_KEYS = {glfw.KEY_E, glfw.KEY_KP_1}
ALL_MOVE_KEYS = (
    FORWARD_KEYS | BACK_KEYS | LEFT_KEYS | RIGHT_KEYS | UP_KEYS | DOWN_KEYS | YAW_LEFT_KEYS | YAW_RIGHT_KEYS
)

# Mode keys: top-row digits (avoid MuJoCo F-keys, Tab, Space, arrows, [ ] camera keys)
MODE_KEY_MAP: dict[int, ControlMode] = {
    glfw.KEY_1: ControlMode.ROLLING,
    glfw.KEY_2: ControlMode.PRETAKEOFF,
    glfw.KEY_3: ControlMode.UPRIGHT,
    glfw.KEY_4: ControlMode.TAKEOFF,
    glfw.KEY_5: ControlMode.HOVER,
    glfw.KEY_6: ControlMode.FLIGHT,
    glfw.KEY_7: ControlMode.LANDING,
    glfw.KEY_0: ControlMode.IDLE,
}

HELP_TEXT = """\
CAROLLINE manual test rig

MODES (tap number key):
  1 ROLLING   2 PRETAKEOFF   3 UPRIGHT   4 TAKEOFF
  5 HOVER     6 FLIGHT       7 LANDING   0 IDLE (motors off — pure physics)

MOVE (hold — release to stop):
  Numpad 8/2/4/6  forward / back / left / right (arrow keys match)
  Page Up/Down   up / down (air modes)
  Q / E          spin on axis (rolling) / yaw (air)
  X              stop          R   hold position on slope (gravity comp)
  - / =          speed down / up
  /              toggle this help

Release movement keys on a ramp to coast back down; press R to actively hold.
Press 0 (IDLE) to cut all motors — cage falls and rolls under gravity only.
"""


class SceneName(str, Enum):
    FLAT = "flat"
    RAMP = "ramp"
    COURSE = "course"
    PLATFORM = "platform"
    TERRAIN = "terrain"


def camera_ground_basis(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return unit (forward, right) on the ground plane from MuJoCo orbit camera angles.

    Forward is toward the top of the screen (into the scene); right is screen-right.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    offset_x = math.cos(el) * math.cos(az)
    offset_y = math.cos(el) * math.sin(az)
    # Camera sits at lookat + offset; negate for screen-forward (top of viewport).
    forward = np.array([offset_x, offset_y], dtype=float)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        forward = np.array([0.0, 1.0], dtype=float)
    else:
        forward /= norm
    # Right-hand tangent on the ground plane (Z up).
    right = np.array([-forward[1], forward[0]], dtype=float)
    return forward, right


def world_move_direction(vx: int, vy: int, forward_axis: str) -> np.ndarray:
    """Map screen keys to fixed world axes (camera rotation ignored).

    forward_axis=x  ->  up / back = +/-X, left / right = +/-Y  (ramp scenes)
    forward_axis=y  ->  up / back = +/-Y, left / right = +/-X  (default flat)
    """
    longitudinal = float(vy)
    if forward_axis == "x":
        return np.array([longitudinal, float(vx)], dtype=float)
    lateral = -float(vx)
    return np.array([lateral, longitudinal], dtype=float)


def update_tracking_camera(
    viewer,
    position: np.ndarray,
    *,
    look_height: float = 0.30,
    distance: float | None = None,
) -> None:
    """Orbit camera that keeps the drone centered in view."""
    viewer.cam.lookat[0] = float(position[0])
    viewer.cam.lookat[1] = float(position[1])
    viewer.cam.lookat[2] = float(position[2]) + look_height
    if distance is not None:
        viewer.cam.distance = distance


def _apply_teleop_tuning(config: ControllerConfig) -> None:
    config.pre_takeoff_omega_gain = max(config.pre_takeoff_omega_gain, 10.0)
    config.pre_takeoff_omega_limit = max(config.pre_takeoff_omega_limit, 16.0)
    config.pre_takeoff_omega_kp = np.array(
        [max(v, t) for v, t in zip(config.pre_takeoff_omega_kp, [26.0, 26.0, 8.0])]
    )
    config.rolling_omega_kp = np.array(
        [max(v, t) for v, t in zip(config.rolling_omega_kp, [9.0, 9.0, 3.5])]
    )
    config.motor_slew_rate = max(config.motor_slew_rate, 1200.0)


def _resolve_model_path(config: ControllerConfig) -> Path:
    path = Path(config.model_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_scene(scene: SceneName, config: ControllerConfig) -> tuple[mujoco.MjModel, float]:
    model_path = _resolve_model_path(config)
    if scene == SceneName.FLAT:
        return compile_model_with_markers(model_path, config), 0.002
    if scene == SceneName.RAMP:
        from carolline_control.scripts.simulate_30deg_ramp_hold import compile_scene, load_ramp_scene_params
        params = load_ramp_scene_params()
        print(f"Ramp scene: {params['angle_deg']:.1f} deg from rolling_ramp_config.yaml")
        return compile_scene(model_path, config.cage_radius), 0.004
    if scene == SceneName.COURSE:
        from carolline_control.scripts.simulate_ramp_platform import (
            RampMissionLayout,
            compile_mission_scene,
        )
        layout = RampMissionLayout.default(config.cage_radius, config.hover_height)
        return compile_mission_scene(model_path, layout), 0.004
    if scene == SceneName.PLATFORM:
        from carolline_control.scripts.simulate_hybrid_locomotion import (
            MissionLayout,
            compile_hybrid_scene,
        )
        layout = MissionLayout.default(config.cage_radius, config.hover_height)
        return compile_hybrid_scene(model_path, layout, config), 0.004
    if scene == SceneName.TERRAIN:
        from carolline_control.terrain.scene import apply_terrain_mobility_tuning, compile_terrain_scene, load_terrain_config

        terrain_raw = load_terrain_config()
        apply_terrain_mobility_tuning(config, terrain_raw)
        model, _, _ = compile_terrain_scene(
            REPO_ROOT / "mujoco_menagerie-main/skydio_x2/scene.xml",
            config=terrain_raw,
            cage_radius=config.cage_radius,
        )
        sim = terrain_raw.get("simulation", {})
        return model, float(sim.get("timestep", 0.004))
    raise ValueError(f"Unknown scene: {scene}")


@dataclass
class KeyInputState:
    """Track held keys from the viewer callback (press + repeat refresh)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    pressed: set[int] = field(default_factory=set)
    key_times: dict[int, float] = field(default_factory=dict)
    pending_mode: ControlMode | None = None
    pending_speed_down: bool = False
    pending_speed_up: bool = False
    pending_toggle_help: bool = False
    roll_hold: bool = False
    last_label: str = "-"
    # Must exceed OS key-repeat initial delay (~500 ms on Windows).
    hold_timeout: float = 2.0

    def on_key(self, keycode: int) -> None:
        now = time.perf_counter()
        with self.lock:
            if keycode in MODE_KEY_MAP:
                self.pending_mode = MODE_KEY_MAP[keycode]
                self.last_label = f"mode:{MODE_KEY_MAP[keycode].name}"
                return
            if keycode in (glfw.KEY_X, glfw.KEY_KP_5):
                self._clear_move_keys_unlocked()
                self.roll_hold = False
                self.last_label = "stop"
                return
            if keycode in (glfw.KEY_R, glfw.KEY_KP_0):
                self._clear_move_keys_unlocked()
                self.roll_hold = True
                self.last_label = "roll_hold"
                return
            if keycode == glfw.KEY_MINUS or keycode == glfw.KEY_KP_SUBTRACT:
                self.pending_speed_down = True
                self.last_label = "speed-"
                return
            if keycode == glfw.KEY_EQUAL or keycode == glfw.KEY_KP_ADD:
                self.pending_speed_up = True
                self.last_label = "speed+"
                return
            if keycode == glfw.KEY_SLASH:
                self.pending_toggle_help = True
                return
            if keycode in ALL_MOVE_KEYS:
                self.pressed.add(keycode)
                self.key_times[keycode] = now
                self.roll_hold = False
                if keycode in FORWARD_KEYS:
                    self.last_label = "fwd"
                elif keycode in BACK_KEYS:
                    self.last_label = "back"
                elif keycode in LEFT_KEYS:
                    self.last_label = "left"
                elif keycode in RIGHT_KEYS:
                    self.last_label = "right"
                elif keycode in UP_KEYS:
                    self.last_label = "up"
                elif keycode in DOWN_KEYS:
                    self.last_label = "down"
                elif keycode in YAW_LEFT_KEYS:
                    self.last_label = "yaw-"
                elif keycode in YAW_RIGHT_KEYS:
                    self.last_label = "yaw+"

    def _clear_move_keys_unlocked(self) -> None:
        self.pressed = {k for k in self.pressed if k not in ALL_MOVE_KEYS}
        for key in list(self.key_times.keys()):
            if key in ALL_MOVE_KEYS:
                del self.key_times[key]

    def _active(self, keys: set[int]) -> bool:
        now = time.perf_counter()
        active = False
        for key in keys:
            if key not in self.pressed:
                continue
            if now - self.key_times.get(key, 0.0) < self.hold_timeout:
                active = True
            else:
                self.pressed.discard(key)
        return active

    def direction(self) -> tuple[int, int, int, int, bool]:
        with self.lock:
            if self.roll_hold:
                return 0, 0, 0, 0, True
            vx = 1 if self._active(LEFT_KEYS) else (-1 if self._active(RIGHT_KEYS) else 0)
            vy = 1 if self._active(FORWARD_KEYS) else (-1 if self._active(BACK_KEYS) else 0)
            vz = 1 if self._active(UP_KEYS) else (-1 if self._active(DOWN_KEYS) else 0)
            yaw = 1 if self._active(YAW_RIGHT_KEYS) else (-1 if self._active(YAW_LEFT_KEYS) else 0)
            return vx, vy, vz, yaw, False

    def consume_pending(self) -> tuple[ControlMode | None, bool, bool, bool]:
        with self.lock:
            mode = self.pending_mode
            self.pending_mode = None
            sd = self.pending_speed_down
            self.pending_speed_down = False
            su = self.pending_speed_up
            self.pending_speed_up = False
            th = self.pending_toggle_help
            self.pending_toggle_help = False
            return mode, sd, su, th

    def clear_motion(self) -> None:
        with self.lock:
            self._clear_move_keys_unlocked()
            self.roll_hold = False

    def release_stale(self) -> None:
        """Drop keys that stopped repeating (viewer does not send key-up events)."""
        now = time.perf_counter()
        with self.lock:
            stale = [
                key
                for key in self.pressed
                if key in ALL_MOVE_KEYS and now - self.key_times.get(key, 0.0) >= self.hold_timeout
            ]
            for key in stale:
                self.pressed.discard(key)
                self.key_times.pop(key, None)


@dataclass
class TeleopState:
    mode: ControlMode = ControlMode.ROLLING
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate: float = 0.0
    speed_scale: float = 1.0
    show_help: bool = False
    ground_speed: float = 3.5
    air_xy_speed: float = 2.2
    air_z_speed: float = 1.0
    yaw_speed: float = 1.0
    vel_time_constant: float = 0.06
    _roll_hold: bool = False

    def stop(self) -> None:
        self.velocity_xy[:] = 0.0
        self.velocity_world[:] = 0.0
        self.yaw_rate = 0.0

    @staticmethod
    def _smooth(current: float, target: float, dt: float, tau: float) -> float:
        if tau <= 1e-6:
            return target
        alpha = 1.0 - math.exp(-dt / tau)
        return current + alpha * (target - current)

    def _smooth_vec(self, current: np.ndarray, target: np.ndarray, dt: float, tau: float) -> np.ndarray:
        if tau <= 1e-6:
            return target.copy()
        alpha = 1.0 - math.exp(-dt / tau)
        return current + alpha * (target - current)

    def update(
        self,
        vx: int,
        vy: int,
        vz: int,
        yaw: int,
        roll_hold: bool,
        dt: float,
        rolling_max_speed: float,
        move_basis: tuple[np.ndarray, np.ndarray] | None = None,
        world_direction: np.ndarray | None = None,
    ) -> None:
        if roll_hold or self.mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT, ControlMode.IDLE):
            target_xy = np.zeros(2)
            target_z = 0.0
            target_yaw = 0.0
        else:
            if world_direction is not None:
                direction = np.asarray(world_direction, dtype=float)
            elif move_basis is not None and (vx != 0 or vy != 0):
                forward, right = move_basis
                direction = forward * float(vy) + right * float(vx)
            else:
                direction = np.array([float(vx), float(vy)], dtype=float)
            max_xy = self.ground_speed * self.speed_scale
            if self.mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.LANDING):
                max_xy = self.air_xy_speed * self.speed_scale
            norm = float(np.linalg.norm(direction))
            target_xy = direction / norm * max_xy if norm > 1e-6 else np.zeros(2)
            target_z = float(vz) * self.air_z_speed * self.speed_scale
            target_yaw = float(yaw) * self.yaw_speed * self.speed_scale

        tau = self.vel_time_constant
        self.velocity_xy = self._smooth_vec(self.velocity_xy, target_xy, dt, tau)
        cap = rolling_max_speed if self.mode == ControlMode.ROLLING else self.air_xy_speed * self.speed_scale * 1.3
        xy_norm = float(np.linalg.norm(self.velocity_xy))
        if xy_norm > cap:
            self.velocity_xy *= cap / xy_norm

        if self.mode in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.LANDING):
            self.velocity_world[0] = self.velocity_xy[0]
            self.velocity_world[1] = self.velocity_xy[1]
            self.velocity_world[2] = self._smooth(
                self.velocity_world[2], target_z, dt, tau
            )
        else:
            self.velocity_world[:] = 0.0
        self.yaw_rate = self._smooth(self.yaw_rate, target_yaw, dt, tau)
        self._roll_hold = roll_hold


def _random_ground_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    tilt_deg = random.uniform(40.0, 80.0)
    heading = random.uniform(0.0, 2.0 * np.pi)
    axis = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
    half = np.radians(tilt_deg) * 0.5
    return [
        float(spawn_xy[0]), float(spawn_xy[1]), ground_z,
        float(np.cos(half)), float(axis[0] * np.sin(half)), float(axis[1] * np.sin(half)), float(axis[2] * np.sin(half)),
    ]


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 80) -> None:
    """Short passive settle so the cage rests on contacts without long tumbling."""
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _sim_step(
    controller: CarollineController,
    estimator: StateEstimator,
    teleop: TeleopState,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dt: float,
) -> ControlMode:
    if teleop.mode == ControlMode.ROLLING:
        controller.mode_manager.request_roll_hold(
            bool(getattr(teleop, "_roll_hold", False))
        )

    state = estimator.estimate(data)
    vel_xy = teleop.velocity_xy.copy()
    vel_w = teleop.velocity_world.copy()
    if teleop.mode not in (ControlMode.FLIGHT, ControlMode.HOVER, ControlMode.LANDING):
        vel_w = np.zeros(3)

    motor, _, mode, _ = controller.compute_manual(
        state, dt, teleop.mode,
        velocity_xy=vel_xy,
        velocity_world=vel_w,
        yaw_rate=teleop.yaw_rate,
    )
    data.ctrl[:] = motor.thrusts
    mujoco.mj_step(model, data)
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE manual test rig")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scene", choices=[s.value for s in SceneName], default=SceneName.FLAT.value)
    parser.add_argument("--substeps", type=int, default=8, help="Physics steps per frame (default 8)")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target viewer refresh rate")
    parser.add_argument("--realtime", action="store_true", help="Cap sim to wall clock")
    parser.add_argument(
        "--spawn-upright",
        action="store_true",
        help="Spawn in a stable side-roll attitude instead of random tilt",
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="Arrow keys use fixed world directions (ignore camera rotation)",
    )
    parser.add_argument(
        "--forward-axis",
        choices=("x", "y"),
        default=None,
        help="World axis for forward key: x=up-ramp (+X), y=+Y (default: x for ramp scene, else y)",
    )
    args = parser.parse_args()

    if args.forward_axis is None:
        args.forward_axis = "x" if args.scene == SceneName.RAMP.value else "y"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    config = load_config(args.config)
    raw = load_raw_config(args.config)
    _apply_teleop_tuning(config)

    model, default_dt = _load_scene(SceneName(args.scene), config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])
    elif args.scene != SceneName.FLAT.value:
        model.opt.timestep = default_dt
    data = mujoco.MjData(model)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)

    ground_z = float(raw.get("ground_z", 0.40))
    if args.spawn_upright:
        half = math.radians(55.0) * 0.5
        qpos = [
            float(config.spawn_xy[0]),
            float(config.spawn_xy[1]),
            ground_z,
            float(math.cos(half)),
            0.0,
            float(math.sin(half)),
            0.0,
        ]
    elif raw.get("random_initial_orientation", True):
        qpos = _random_ground_qpos(list(config.spawn_xy), ground_z)
    else:
        qpos = raw.get("initial_qpos")
    if qpos:
        data.qpos[:7] = np.asarray(qpos, dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        _settle(model, data)

    dt = float(model.opt.timestep)
    substeps = max(1, args.substeps)
    keys = KeyInputState()
    teleop = TeleopState(mode=ControlMode.ROLLING, ground_speed=config.rolling_max_speed)
    if args.scene == SceneName.TERRAIN.value:
        from carolline_control.terrain.scene import load_terrain_config

        mob = load_terrain_config().get("mobility", {})
        teleop.ground_speed = float(mob.get("teleop_ground_speed", config.rolling_max_speed))
        teleop.vel_time_constant = float(mob.get("teleop_time_constant", 0.030))
    last_mode = teleop.mode.name

    print(HELP_TEXT)
    if args.world_frame:
        axis_label = "+X" if args.forward_axis == "x" else "+Y"
        print(
            f"World-frame teleop: forward={axis_label}, rotating the view does not change direction."
        )
    else:
        print("Camera-relative teleop: arrow forward follows the screen (view rotation changes direction).")
    print("Press / for help. 0=IDLE. X=stop.")

    def on_key(keycode: int) -> None:
        keys.on_key(keycode)

    track_distance = 6.0
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        tune_viewer_for_speed(viewer)
        viewer.cam.distance = track_distance
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -20.0
        update_tracking_camera(viewer, data.qpos[:3], look_height=0.30, distance=track_distance)
        wall_t0 = time.perf_counter()
        sim_t0 = data.time

        while viewer.is_running():
            pending_mode, speed_down, speed_up, toggle_help = keys.consume_pending()
            if pending_mode is not None and pending_mode != teleop.mode:
                if pending_mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT):
                    keys.clear_motion()
                    teleop.stop()
                    controller.pre_takeoff.reset(estimator.estimate(data))
                elif pending_mode in (ControlMode.TAKEOFF, ControlMode.HOVER, ControlMode.FLIGHT):
                    keys.clear_motion()
                    teleop.stop()
                    state = estimator.estimate(data)
                    controller._hover_anchor_xy = state.position[:2].copy()
                    controller._flight_yaw = float(np.arctan2(state.rotation[1, 0], state.rotation[0, 0]))
                if pending_mode == ControlMode.ROLLING:
                    controller.rolling.reset()
                    controller.mode_manager.request_roll_hold(False)
                teleop.mode = pending_mode
            if speed_down:
                teleop.speed_scale = max(0.3, teleop.speed_scale * 0.85)
            if speed_up:
                teleop.speed_scale = min(2.5, teleop.speed_scale * 1.15)
            if toggle_help:
                teleop.show_help = not teleop.show_help

            keys.release_stale()
            vx, vy, vz, yaw, roll_hold = keys.direction()
            move_basis: tuple[np.ndarray, np.ndarray] | None = None
            world_dir: np.ndarray | None = None
            if vx != 0 or vy != 0:
                if args.world_frame:
                    world_dir = world_move_direction(vx, vy, args.forward_axis)
                else:
                    move_basis = camera_ground_basis(viewer.cam.azimuth, viewer.cam.elevation)
            frame_dt = dt * substeps
            frame_start = time.perf_counter()
            teleop.update(
                vx,
                vy,
                vz,
                yaw,
                roll_hold,
                frame_dt,
                config.rolling_max_speed,
                move_basis=move_basis,
                world_direction=world_dir,
            )

            for sub_i in range(substeps):
                if sub_i == 0:
                    mode = _sim_step(controller, estimator, teleop, model, data, frame_dt)
                else:
                    mujoco.mj_step(model, data)
                if mode.name != last_mode:
                    print(f"t={data.time:6.2f}s  {last_mode} -> {mode.name}")
                    last_mode = mode.name

            state = estimator.estimate(data)
            update_tracking_camera(viewer, state.position, look_height=0.30)
            motors = np.asarray(data.ctrl, dtype=float)
            status = (
                f"Mode: {teleop.mode.name}  input: {keys.last_label}  "
                f"cmd [{teleop.velocity_xy[0]:+.2f},{teleop.velocity_xy[1]:+.2f}] m/s  "
                f"on_ground={state.on_ground}  contact={state.contact_valid}  "
                f"motors [{motors[0]:+.1f},{motors[1]:+.1f},{motors[2]:+.1f},{motors[3]:+.1f}] N"
            )
            if teleop.show_help:
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        HELP_TEXT + f"\n\n{status}\n"
                        f"pos=({state.position[0]:.2f},{state.position[1]:.2f},{state.position[2]:.2f})  "
                        f"speed={float(np.linalg.norm(state.velocity[:2])):.2f} m/s",
                        "",
                    )
                )
            else:
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                        status,
                        "",
                    )
                )

            viewer.sync(state_only=True)

            if args.realtime:
                delay = (data.time - sim_t0) - (time.perf_counter() - wall_t0)
                if delay > 0.0:
                    time.sleep(min(delay, 0.02))
            else:
                elapsed = time.perf_counter() - frame_start
                sleep_s = (1.0 / max(args.target_fps, 1.0)) - elapsed
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    print(f"Done. Final mode: {controller.mode_manager.mode.name}")


if __name__ == "__main__":
    main()
