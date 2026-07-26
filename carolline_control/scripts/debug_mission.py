"""Quick mission diagnostic: mode timeline, position, contacts."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.utils.so3 import body_z_world
from carolline_control.visualization.markers import compile_model_with_markers


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    config_path = REPO_ROOT / "carolline_control" / "config.yaml"
    raw = load_raw_config(config_path)
    config = load_config(config_path)

    model = compile_model_with_markers(config.model_path, config)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    print(f"Model: {config.model_path}")
    print(f"ngeom={model.ngeom}, timestep={dt}, cage geoms in estimator={len(StateEstimator(model, config)._cage_geom_ids)}")

    estimator = StateEstimator(model, config)
    estimator.fill_inertial_params(config)
    controller = CarollineController(config)

    spawn = raw["mission"]["spawn_xy"]
    gz = float(raw.get("ground_z", 0.40))
    tilt = random.uniform(35, 85)
    heading = random.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(heading), np.sin(heading), 0.0])
    half = np.radians(tilt) * 0.5
    qpos = [spawn[0], spawn[1], gz, np.cos(half), *(axis * np.sin(half))]
    data.qpos[:7] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    modes: Counter[str] = Counter()
    transitions: list[tuple[float, str, str]] = []
    last_mode = controller.mode_manager.mode.name
    duration = 90.0
    samples = []

    while data.time < duration:
        state = estimator.estimate(data)
        motor, _, mode, _ = controller.compute(state, dt)
        data.ctrl[:] = motor.thrusts
        if mode.name != last_mode:
            transitions.append((data.time, last_mode, mode.name))
            last_mode = mode.name
        modes[mode.name] += 1
        if int(data.time / 0.5) != int((data.time - dt) / 0.5):
            samples.append(
                (
                    data.time,
                    mode.name,
                    state.position[:2].copy(),
                    float(np.linalg.norm(state.position[:2] - config.roll_target)),
                    state.contact_valid,
                    state.on_ground,
                    float(body_z_world(state.rotation)[2]),
                    float(np.linalg.norm(state.velocity[:2])),
                )
            )
        mujoco.mj_step(model, data)

    print(f"Mass={config.mass:.3f} kg")
    print("Mode counts:", dict(modes))
    print("Transitions:")
    for t, a, b in transitions:
        print(f"  t={t:6.2f}s  {a} -> {b}")
    print("Samples every 0.5s:")
    for row in samples[::2]:
        t, m, xy, err, cv, og, bz, spd = row
        print(
            f"  t={t:5.1f} {m:10s} pos=({xy[0]:5.2f},{xy[1]:5.2f}) "
            f"err={err:5.2f} contact={cv} ground={og} bz={bz:.2f} spd={spd:.2f}"
        )
    final = samples[-1]
    print(f"Final dist to roll target: {final[3]:.3f} m")


if __name__ == "__main__":
    main()
