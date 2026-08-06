"""Draw rangefinder rays in the MuJoCo passive viewer."""

from __future__ import annotations

import mujoco
import numpy as np

from carolline_control.navigation.perception import ObstacleScan, RangePerception


def _ray_color(distance: float, *, max_range: float, stop_distance: float, blocked: bool) -> np.ndarray:
    if distance < stop_distance:
        return np.array([0.95, 0.18, 0.12, 0.92], dtype=float)
    if distance >= max_range * 0.92:
        return np.array([0.25, 0.88, 0.38, 0.42], dtype=float)
    t = (distance - stop_distance) / max(max_range - stop_distance, 1e-6)
    return np.array([0.95, 0.82, 0.12, 0.55 + 0.25 * t], dtype=float)


def draw_rangefinder_rays(
    viewer,
    perception: RangePerception,
    data: mujoco.MjData,
    scan: ObstacleScan,
) -> None:
    """Render sensor rays into viewer.user_scn (clears previous user geoms)."""
    user_scn = viewer.user_scn
    user_scn.ngeom = 0
    ngeom = 0
    max_geoms = int(user_scn.maxgeom)
    hit_radius = 0.035

    for site_id, is_up, distance in perception.iter_ray_readings(data):
        if ngeom >= max_geoms:
            break

        origin = np.asarray(data.site_xpos[site_id], dtype=float)
        xmat = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        direction = -xmat[:, 2]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        end = origin + direction * float(distance)

        blocked = (not is_up) and distance < perception.stop_distance
        rgba = _ray_color(
            distance,
            max_range=perception.max_range,
            stop_distance=perception.stop_distance,
            blocked=blocked,
        )
        width = 0.011 if is_up else 0.014
        if (not is_up) and abs(distance - scan.forward_min_m) < 1e-4 and scan.blocked:
            rgba = np.array([1.0, 0.05, 0.05, 1.0], dtype=float)
            width = 0.022

        geom = user_scn.geoms[ngeom]
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            origin,
            end,
        )
        geom.rgba[:] = rgba
        ngeom += 1

        if distance < perception.max_range * 0.92 and ngeom < max_geoms:
            hit = user_scn.geoms[ngeom]
            mujoco.mjv_initGeom(
                hit,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([hit_radius, 0.0, 0.0], dtype=float),
                end,
                np.eye(3).flatten(),
                rgba,
            )
            ngeom += 1

    user_scn.ngeom = ngeom
