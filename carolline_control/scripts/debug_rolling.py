"""Debug rolling wrench commands."""
import random
import sys

import mujoco
import numpy as np

sys.path.insert(0, r"E:\CAROLLINE_PROJECT")
from carolline_control.carolline_controller import CarollineController
from carolline_control.config_loader import load_config, load_raw_config
from carolline_control.controllers.state_estimator import StateEstimator
from carolline_control.main import _random_ground_qpos
from carolline_control.visualization.markers import compile_model_with_markers

random.seed(42)
np.random.seed(42)
config = load_config(r"E:\CAROLLINE_PROJECT\carolline_control\config.yaml")
raw = load_raw_config(r"E:\CAROLLINE_PROJECT\carolline_control\config.yaml")
model = compile_model_with_markers(config.model_path, config)
data = mujoco.MjData(model)
est = StateEstimator(model, config)
est.fill_inertial_params(config)
ctrl = CarollineController(config)
spawn = raw.get("mission", {}).get("spawn_xy", [0, 0])
ground_z = float(raw.get("ground_z", 0.40))
qpos = _random_ground_qpos(spawn, ground_z)
data.qpos[:7] = qpos
data.qvel[:] = 0
mujoco.mj_forward(model, data)
dt = model.opt.timestep

next_print = 0.0
for _ in range(12000):
    state = est.estimate(data)
    motor, cmd, mode, target = ctrl.compute(state, dt)
    if state.time >= next_print:
        v_xy = ctrl.planner.rolling_velocity(state)
        print(
            f"t={state.time:.2f} pos={state.position[:2]} v={state.velocity[:2]} "
            f"v_cmd={v_xy} |M|={np.linalg.norm(cmd.moment_body):.2f} "
            f"|omega|={np.linalg.norm(state.omega_body):.3f} motors={motor.thrusts} "
            f"sum={motor.thrusts.sum():.2f} thrust={cmd.thrust:.2f}"
        )
        next_print += 0.5
    data.ctrl[:] = motor.thrusts
    mujoco.mj_step(model, data)
