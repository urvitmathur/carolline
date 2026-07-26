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
from carolline_control.utils.types import ControlMode, ControllerConfig
from carolline_control.visualization.markers import compile_model_with_markers

# Movement — arrow keys primary (numpad kept as alternate)
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
    glfw.KEY_8: ControlMode.IDLE,
}

HELP_TEXT = """\
CAROLLINE manual test rig

MODES (tap number key):
  1 ROLLING   2 PRETAKEOFF   3 UPRIGHT   4 TAKEOFF
  5 HOVER     6 FLIGHT       7 LANDING   8 IDLE (motors off — pure physics)

MOVE (hold — release to stop):
  Arrow keys     forward / back / left / right
  Page Up/Down   up / down (air modes)
  Q / E          spin on axis (rolling) / yaw (air)
  X              stop          R   hold position on slope (gravity comp)
  - / =          speed down / up
  /              toggle this help

Release movement keys on a ramp to coast back down; press R to actively hold.
Press 8 (IDLE) to cut all motors — cage falls and rolls under gravity only.
"""


class SceneName(str, Enum):
    FLAT = "flat"
    RAMP = "ramp"
    COURSE = "course"
    PLATFORM = "platform"


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
    raise ValueError(f"Unknown scene: {scene}")


@dataclass
class KeyInputState:
    """Track held keys via repeat timestamps (callback runs on viewer thread)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    key_times: dict[int, float] = field(default_factory=dict)
    pending_mode: ControlMode | None = None
    pending_speed_down: bool = False
    pending_speed_up: bool = False
    pending_toggle_help: bool = False
    roll_hold: bool = False
    last_label: str = "-"
    hold_timeout: float = 0.28

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
        for key in list(self.key_times.keys()):
            if key in ALL_MOVE_KEYS:
                del self.key_times[key]

    def _active(self, keys: set[int]) -> bool:
        now = time.perf_counter()
        return any(now - self.key_times.get(k, 0.0) < self.hold_timeout for k in keys)

    def direction(self) -> tuple[int, int, int, int, bool]:
        with self.lock:
            if self.roll_hold:
                return 0, 0, 0, 0, True
            vx = -1 if self._active(LEFT_KEYS) else (1 if self._active(RIGHT_KEYS) else 0)
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
    ) -> None:
        if roll_hold or self.mode in (ControlMode.PRETAKEOFF, ControlMode.UPRIGHT, ControlMode.IDLE):
            target_xy = np.zeros(2)
            target_z = 0.0
            target_yaw = 0.0
        else:
            direction = np.array([float(vx), float(vy)])
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


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 600) -> None:
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
    parser.add_argument("--substeps", type=int, default=20, help="Physics steps per frame (default 20)")
    parser.add_argument("--realtime", action="store_true", help="Cap sim to wall clock")
    args = parser.parse_args()

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
    if raw.get("random_initial_orientation", True):
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
    last_mode = teleop.mode.name

    print(HELP_TEXT)
    print("Hold arrow keys to move. Release to stop. 1=rolling 6=flight.")

    def on_key(keycode: int) -> None:
        keys.on_key(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -20.0
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
                teleop.mode = pending_mode
            if speed_down:
                teleop.speed_scale = max(0.3, teleop.speed_scale * 0.85)
            if speed_up:
                teleop.speed_scale = min(2.5, teleop.speed_scale * 1.15)
            if toggle_help:
                teleop.show_help = not teleop.show_help

            vx, vy, vz, yaw, roll_hold = keys.direction()
            frame_dt = dt * substeps
            teleop.update(vx, vy, vz, yaw, roll_hold, frame_dt, config.rolling_max_speed)

            for _ in range(substeps):
                mode = _sim_step(controller, estimator, teleop, model, data, dt)
                if mode.name != last_mode:
                    print(f"t={data.time:6.2f}s  {last_mode} -> {mode.name}")
                    last_mode = mode.name

            if teleop.show_help:
                vx, vy, vz, yaw, hold = keys.direction()
                state = estimator.estimate(data)
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        HELP_TEXT + f"\n\nMode: {teleop.mode.name}  hold: {keys.last_label}\n"
                        f"cmd [{teleop.velocity_xy[0]:+.1f},{teleop.velocity_xy[1]:+.1f}] "
                        f"z={teleop.velocity_world[2]:+.1f}  pos z={state.position[2]:.2f}",
                        "",
                    )
                )
            else:
                viewer.clear_texts()

            viewer.sync()

            if args.realtime:
                delay = (data.time - sim_t0) - (time.perf_counter() - wall_t0)
                if delay > 0.0:
                    time.sleep(min(delay, 0.02))

    print(f"Done. Final mode: {controller.mode_manager.mode.name}")


if __name__ == "__main__":
    main()
