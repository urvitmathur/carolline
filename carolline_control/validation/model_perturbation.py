"""Apply environment uncertainty to MuJoCo model (validation layer only)."""

from __future__ import annotations

import mujoco
import numpy as np

from carolline_control.validation.randomization import RunParameters


def _scale_geom_friction(model: mujoco.MjModel, geom_name: str, scale: float) -> None:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid >= 0:
        model.geom_friction[gid, :] *= scale


def apply_model_perturbations(model: mujoco.MjModel, params: RunParameters) -> None:
    """Perturb mass, inertia, and contact friction before controller reads inertial params."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    if body_id >= 0:
        model.body_mass[body_id] *= params.mass_scale
        model.body_inertia[body_id] *= params.inertia_scale

    _scale_geom_friction(model, "floor", params.ground_friction_scale)
    _scale_geom_friction(model, "cage_collision", params.cage_friction_scale)

    # Rolling resistance maps to tangential friction component (index 1).
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id, 1] *= params.rolling_resistance_scale
        model.geom_friction[floor_id, 2] *= params.rolling_resistance_scale

    cage_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cage_collision")
    if cage_id >= 0:
        model.geom_friction[cage_id, 1] *= params.rolling_resistance_scale


def apply_gain_overrides(config, gain_overrides: dict[str, float]):
    """Return a copy of ControllerConfig with scaled gains (controller code unchanged)."""
    from dataclasses import replace

    if not gain_overrides:
        return config
    updates: dict[str, float] = {}
    for key, scale in gain_overrides.items():
        if hasattr(config, key):
            base = float(getattr(config, key))
            updates[key] = base * scale
    return replace(config, **updates) if updates else config
