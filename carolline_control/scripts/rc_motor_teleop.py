"""RC motor teleop — drive rotors directly like a radio transmitter (passthrough).

Bypasses velocity controllers and the mission FSM.  Channel values map to
per-motor thrust in Newtons on data.ctrl (bidirectional ±13 N per CAROLLINE).

Modes:
  direct  — 4 channels → 4 motors independently (bench / trim / motor test)
  acro    — 2-stick TX: throttle + roll/pitch/yaw → motor mixer → 4 motors

Keyboard simulates sticks or per-motor trims; plug in a gamepad/RC USB receiver
if GLFW sees it as joystick 1.

Run from repo root:
    python carolline_control/scripts/rc_motor_teleop.py
    python carolline_control/scripts/rc_motor_teleop.py --mode acro
    python carolline_control/scripts/rc_motor_teleop.py --list-joysticks
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.esc_mapper import EscThrustMapper
from carolline_control.controllers.motor_mixer import MotorMixer
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.types import MotorCommand
from carolline_control.visualization.markers import compile_model_with_markers

# Direct mode: per-motor channel trim (hold to ramp channel up/down)
MOTOR_UP_KEYS = {
    0: {glfw.KEY_KP_1},
    1: {glfw.KEY_KP_4},
    2: {glfw.KEY_KP_7},
    3: {glfw.KEY_KP_3},
}
MOTOR_DOWN_KEYS = {
    0: {glfw.KEY_KP_2},
    1: {glfw.KEY_KP_5},
    2: {glfw.KEY_KP_8},
    3: {glfw.KEY_KP_6},
}

# Acro mode: arrow keys = pitch/roll, W/S = throttle, Q/E = yaw (numpad kept as alternate)
THROTTLE_UP = {glfw.KEY_W, glfw.KEY_KP_9}
THROTTLE_DOWN = {glfw.KEY_S, glfw.KEY_KP_3}
YAW_LEFT = {glfw.KEY_Q, glfw.KEY_KP_7}
YAW_RIGHT = {glfw.KEY_E, glfw.KEY_KP_1}
PITCH_UP = {glfw.KEY_UP, glfw.KEY_KP_8}
PITCH_DOWN = {glfw.KEY_DOWN, glfw.KEY_KP_2}
ROLL_LEFT = {glfw.KEY_LEFT, glfw.KEY_KP_4}
ROLL_RIGHT = {glfw.KEY_RIGHT, glfw.KEY_KP_6}

ALL_RC_KEYS = (
    THROTTLE_UP | THROTTLE_DOWN | YAW_LEFT | YAW_RIGHT | PITCH_UP | PITCH_DOWN | ROLL_LEFT | ROLL_RIGHT
    | MOTOR_UP_KEYS[0] | MOTOR_UP_KEYS[1] | MOTOR_UP_KEYS[2] | MOTOR_UP_KEYS[3]
    | MOTOR_DOWN_KEYS[0] | MOTOR_DOWN_KEYS[1] | MOTOR_DOWN_KEYS[2] | MOTOR_DOWN_KEYS[3]
)


class DriveMode(str, Enum):
    DIRECT = "direct"
    ACRO = "acro"


HELP_DIRECT = """\
RC DIRECT — 4 channels → 4 motors (N)

  Motor 1:  Numpad 1 / 2
  Motor 2:  Numpad 4 / 5
  Motor 3:  Numpad 7 / 8
  Motor 4:  Numpad 3 / 6

  0 or X     all motors zero
  H          hover (~3.25 N each motor)
  /          toggle help

Channel -1..+1  →  thrust -13..+13 N per motor (bidirectional).
"""

HELP_ACRO = """\
RC ACRO — 2-stick transmitter → motor mixer

  W / S          throttle up / down
  Arrow keys     pitch (up/down) and roll (left/right)
  Q / E          yaw left / right

  0 or X         zero throttle / moments
  H              hover preset
  /              toggle help

Sticks map to collective thrust + body moments, then mixer → 4 motors.
Note: arrows also rotate the MuJoCo camera while held.
"""


@dataclass
class RCInput:
    """RC channel state in [-1, 1] (center = 0, full up = +1)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    channels: np.ndarray = field(default_factory=lambda: np.zeros(4))
    sticks: np.ndarray = field(default_factory=lambda: np.zeros(4))  # throttle, roll, pitch, yaw
    key_times: dict[int, float] = field(default_factory=dict)
    pending_zero: bool = False
    pending_hover: bool = False
    pending_toggle_help: bool = False
    hold_timeout: float = 0.22
    ramp_rate: float = 2.5
    use_joystick: bool = False
    joystick_id: int = glfw.JOYSTICK_1

    def on_key(self, keycode: int) -> None:
        now = time.perf_counter()
        with self.lock:
            if keycode in (glfw.KEY_X, glfw.KEY_KP_0, glfw.KEY_0):
                self.pending_zero = True
                return
            if keycode == glfw.KEY_H:
                self.pending_hover = True
                return
            if keycode == glfw.KEY_SLASH:
                self.pending_toggle_help = True
                return
            if keycode in ALL_RC_KEYS:
                self.key_times[keycode] = now

    def _active(self, keys: set[int]) -> bool:
        now = time.perf_counter()
        return any(now - self.key_times.get(k, 0.0) < self.hold_timeout for k in keys)

    def _stick_axis(self, pos_keys: set[int], neg_keys: set[int]) -> float:
        if self._active(pos_keys):
            return 1.0
        if self._active(neg_keys):
            return -1.0
        return 0.0

    def poll_joystick(self) -> None:
        if not self.use_joystick or not glfw.joystick_present(self.joystick_id):
            return
        axes = glfw.get_joystick_axes(self.joystick_id)
        if axes is None or len(axes) < 4:
            return
        with self.lock:
            # Typical gamepad: 0=left X, 1=left Y, 2=right X, 3=right Y
            lx, ly, rx, ry = float(axes[0]), float(axes[1]), float(axes[2]), float(axes[3])
            self.sticks[0] = float(np.clip(-ly, -1.0, 1.0))  # throttle (up = +)
            self.sticks[1] = float(np.clip(rx, -1.0, 1.0))   # roll
            self.sticks[2] = float(np.clip(-ry, -1.0, 1.0))  # pitch
            self.sticks[3] = float(np.clip(lx, -1.0, 1.0))   # yaw

    def update_keyboard(self, dt: float, mode: DriveMode) -> None:
        with self.lock:
            if self.use_joystick and glfw.joystick_present(self.joystick_id):
                return
            if mode == DriveMode.DIRECT:
                for i in range(4):
                    delta = 0.0
                    if self._active(MOTOR_UP_KEYS[i]):
                        delta += self.ramp_rate * dt
                    if self._active(MOTOR_DOWN_KEYS[i]):
                        delta -= self.ramp_rate * dt
                    self.channels[i] = float(np.clip(self.channels[i] + delta, -1.0, 1.0))
            else:
                targets = np.array([
                    self._stick_axis(THROTTLE_UP, THROTTLE_DOWN),
                    self._stick_axis(ROLL_RIGHT, ROLL_LEFT),
                    self._stick_axis(PITCH_UP, PITCH_DOWN),
                    self._stick_axis(YAW_RIGHT, YAW_LEFT),
                ])
                alpha = 1.0 - math.exp(-dt / 0.05)
                self.sticks = self.sticks + alpha * (targets - self.sticks)

    def consume_pending(self) -> tuple[bool, bool, bool]:
        with self.lock:
            z, h, t = self.pending_zero, self.pending_hover, self.pending_toggle_help
            self.pending_zero = False
            self.pending_hover = False
            self.pending_toggle_help = False
            return z, h, t

    def zero(self) -> None:
        with self.lock:
            self.channels[:] = 0.0
            self.sticks[:] = 0.0

    def set_hover_channels(self, hover_thrust: float, motor_max: float) -> None:
        ch = float(np.clip(hover_thrust / motor_max, -1.0, 1.0))
        with self.lock:
            self.channels[:] = ch
            self.sticks[0] = 0.35
            self.sticks[1:] = 0.0


def _channels_to_thrust(channels: np.ndarray, motor_min: float, motor_max: float) -> np.ndarray:
    """Map [-1,1] channels to bidirectional thrust [N]."""
    thrust = np.asarray(channels, dtype=float) * motor_max
    return np.clip(thrust, motor_min, motor_max)


def _acro_to_motors(
    sticks: np.ndarray,
    mixer: MotorMixer,
    mass: float,
    gravity: float,
    motor_max: float,
) -> np.ndarray:
    """Map 4 stick axes to motor thrust via open-loop mixer (no attitude loop)."""
    throttle, roll, pitch, yaw = [float(x) for x in sticks[:4]]
    hover = mass * gravity
    # throttle stick: -1 ≈ idle, 0 ≈ hover, +1 ≈ max climb
    collective = hover * (1.0 + 0.85 * throttle)
    collective = float(np.clip(collective, 0.0, 4.0 * motor_max * 0.95))
    moment = np.array([
        0.35 * roll * hover,
        0.35 * pitch * hover,
        0.08 * yaw * hover,
    ])
    return mixer.mix(collective, moment).thrusts


def _random_ground_qpos(spawn_xy: list[float], ground_z: float) -> list[float]:
    tilt_deg = random.uniform(40.0, 80.0)
    heading = random.uniform(0.0, 2.0 * np.pi)
    axis = np.array([np.cos(heading), np.sin(heading), 0.0])
    half = np.radians(tilt_deg) * 0.5
    return [
        float(spawn_xy[0]), float(spawn_xy[1]), ground_z,
        float(np.cos(half)), float(axis[0] * np.sin(half)), float(axis[1] * np.sin(half)), float(axis[2] * np.sin(half)),
    ]


def _list_joysticks() -> None:
    if not glfw.init():
        print("GLFW init failed")
        return
    try:
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(jid):
                name = glfw.get_joystick_name(jid) or "unknown"
                print(f"  Joystick {jid}: {name}")
    finally:
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="RC direct motor teleop")
    parser.add_argument("--config", default=str(REPO_ROOT / "carolline_control" / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=[m.value for m in DriveMode], default=DriveMode.DIRECT.value)
    parser.add_argument("--joystick", type=int, default=0, help="GLFW joystick id (0=JOYSTICK_1, -1=keyboard only)")
    parser.add_argument("--esc", action="store_true", help="Apply ESC thrust curve before ctrl")
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--list-joysticks", action="store_true")
    args = parser.parse_args()

    if args.list_joysticks:
        _list_joysticks()
        return

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mode = DriveMode(args.mode)
    config = load_config(args.config)
    raw = load_raw_config(args.config)

    model = compile_model_with_markers(config.model_path, config)
    if "timestep" in raw:
        model.opt.timestep = float(raw["timestep"])
    data = mujoco.MjData(model)

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    mixer = MotorMixer(config)
    esc = EscThrustMapper(
        thrust_forward_max=config.esc_thrust_forward_max,
        thrust_reverse_max=config.esc_thrust_reverse_max,
        reverse_efficiency=config.esc_reverse_efficiency,
        enabled=args.esc,
    )

    spawn = list(config.spawn_xy)
    ground_z = float(raw.get("ground_z", 0.40))
    qpos = _random_ground_qpos(spawn, ground_z) if raw.get("random_initial_orientation", True) else raw.get("initial_qpos")
    if qpos:
        data.qpos[:7] = np.asarray(qpos, dtype=float)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(400):
            mujoco.mj_step(model, data)

    dt = float(model.opt.timestep)
    substeps = max(1, args.substeps)
    hover_thrust = config.mass * config.gravity / 4.0

    rc = RCInput(
        use_joystick=args.joystick >= 0,
        joystick_id=glfw.JOYSTICK_1 + max(args.joystick, 0),
    )
    show_help = True
    help_text = HELP_DIRECT if mode == DriveMode.DIRECT else HELP_ACRO

    print(help_text)
    print(f"Mode: {mode.value}  motors [{config.motor_min}, {config.motor_max}] N")
    if rc.use_joystick:
        print(f"Joystick id {rc.joystick_id} (use --list-joysticks to inspect)")

    def on_key(keycode: int) -> None:
        rc.on_key(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 5.5
        wall_t0 = time.perf_counter()
        sim_t0 = data.time

        while viewer.is_running():
            rc.poll_joystick()
            zero, hover, toggle = rc.consume_pending()
            if zero:
                rc.zero()
            if hover:
                rc.set_hover_channels(hover_thrust, config.motor_max)
            if toggle:
                show_help = not show_help

            frame_dt = dt * substeps
            rc.update_keyboard(frame_dt, mode)

            with rc.lock:
                channels = rc.channels.copy()
                sticks = rc.sticks.copy()

            if mode == DriveMode.DIRECT:
                motor_cmd = _channels_to_thrust(channels, config.motor_min, config.motor_max)
            else:
                motor_cmd = _acro_to_motors(sticks, mixer, config.mass, config.gravity, config.motor_max)

            motor_cmd = esc.apply(MotorCommand(thrusts=np.asarray(motor_cmd, dtype=float))).thrusts

            for _ in range(substeps):
                data.ctrl[:] = motor_cmd
                mujoco.mj_step(model, data)

            state = estimator.estimate(data)
            if show_help:
                if mode == DriveMode.DIRECT:
                    extra = f"ch [{channels[0]:+.2f} {channels[1]:+.2f} {channels[2]:+.2f} {channels[3]:+.2f}]"
                else:
                    extra = f"sticks T/R/P/Y [{sticks[0]:+.2f} {sticks[1]:+.2f} {sticks[2]:+.2f} {sticks[3]:+.2f}]"
                viewer.set_texts(
                    (
                        int(mujoco.mjtFontScale.mjFONTSCALE_150),
                        int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                        help_text + f"\n\n{extra}\n"
                        f"Motors [N]: {motor_cmd[0]:+.2f} {motor_cmd[1]:+.2f} "
                        f"{motor_cmd[2]:+.2f} {motor_cmd[3]:+.2f}\n"
                        f"pos z={state.position[2]:.2f}  |ctrl|={np.linalg.norm(motor_cmd):.1f}",
                        "",
                    )
                )
            else:
                viewer.clear_texts()

            viewer.sync()

            delay = (data.time - sim_t0) - (time.perf_counter() - wall_t0)
            if delay > 0.0:
                time.sleep(min(delay, 0.02))

    print("RC teleop ended.")


if __name__ == "__main__":
    main()
