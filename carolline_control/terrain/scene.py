"""Build MuJoCo scene with mixed flat / hilly / mountainous heightfield terrain."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from carolline_control.terrain.heightmap import TerrainLayout, sample_terrain_height, write_heightmap_png

CAROLLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERRAIN_CONFIG = CAROLLINE_ROOT / "terrain_config.yaml"
TERRAIN_ASSET_DIR = CAROLLINE_ROOT / "terrain" / "assets"


def load_terrain_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_TERRAIN_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_terrain_mobility_tuning(config, raw: dict | None = None) -> dict:
    """Boost rolling limits and gains for mixed terrain (flat + hills + mountains)."""
    raw = raw or load_terrain_config()
    mob = raw.get("mobility", {})
    config.rolling_max_speed = float(mob.get("rolling_max_speed", 4.75))
    config.rolling_kp = float(mob.get("rolling_kp", max(config.rolling_kp, 0.85)))
    config.rolling_kd = float(mob.get("rolling_kd", max(config.rolling_kd, 1.20)))
    kp = np.asarray(mob.get("rolling_omega_kp", [12.0, 12.0, 4.5]), dtype=float)
    config.rolling_omega_kp = np.array(
        [max(a, b) for a, b in zip(config.rolling_omega_kp, kp)]
    )
    config.motor_slew_rate = max(config.motor_slew_rate, float(mob.get("motor_slew_rate", 1400.0)))
    if "friction" in mob and "terrain" in raw:
        raw["terrain"]["friction"] = mob["friction"]
    return mob


def terrain_layout_from_config(raw: dict) -> TerrainLayout:
    t = raw.get("terrain", {})
    return TerrainLayout(
        half_x=float(t.get("half_x", 14.0)),
        half_y=float(t.get("half_y", 14.0)),
        max_height=float(t.get("max_height", 1.45)),
        base_thickness=float(t.get("base_thickness", 0.04)),
        nrow=int(t.get("nrow", 128)),
        ncol=int(t.get("ncol", 128)),
        flat_x_end=float(t.get("flat_x_end", -7.0)),
        hills_x_end=float(t.get("hills_x_end", 2.0)),
        seed=int(t.get("seed", 7)),
    )


def compile_terrain_scene(
    base_scene_path: str | Path,
    *,
    config: dict | None = None,
    cage_radius: float = 0.40,
    cache_heightmap: bool = True,
) -> tuple[mujoco.MjModel, TerrainLayout, np.ndarray]:
    """Return compiled model, layout, and height samples for spawn placement."""
    raw = config or load_terrain_config()
    layout = terrain_layout_from_config(raw)
    png_name = f"heightmap_seed{layout.seed}_{layout.nrow}x{layout.ncol}.png"
    png_path = TERRAIN_ASSET_DIR / png_name
    if cache_heightmap and png_path.exists():
        from PIL import Image

        gray = np.asarray(Image.open(png_path).convert("L"), dtype=float) / 255.0
        heights = gray * layout.max_height
    else:
        heights, png_path = write_heightmap_png(png_path, layout)

    spec = mujoco.MjSpec.from_file(str(base_scene_path))
    world = spec.worldbody

    for geom in spec.geoms:
        if geom.name == "floor":
            geom.contype = 0
            geom.conaffinity = 0

    spec.add_hfield(
        name="mobility_terrain_hf",
        file=str(png_path),
        size=[layout.half_x, layout.half_y, layout.max_height, layout.base_thickness],
    )
    friction = raw.get("terrain", {}).get("friction", [1.05, 0.012, 0.006])
    world.add_geom(
        name="mobility_terrain",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="mobility_terrain_hf",
        pos=[0.0, 0.0, 0.0],
        friction=friction,
        rgba=[0.38, 0.48, 0.36, 1.0],
        contype=1,
        conaffinity=1,
        condim=6,
    )

    # Safety void floor far below the playable area.
    world.add_geom(
        name="terrain_void_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, -3.0],
        size=[layout.half_x + 2.0, layout.half_y + 2.0, 0.05],
        rgba=[0.15, 0.18, 0.22, 0.4],
        contype=1,
        conaffinity=1,
        condim=3,
    )

    # Perimeter berms to keep the cage inside the heightfield.
    wall_h = 0.55
    wall_t = 0.25
    span_x = layout.half_x + 0.6
    span_y = layout.half_y + 0.6
    for name, pos, half in [
        ("terrain_wall_xmin", [-layout.half_x - wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("terrain_wall_xmax", [layout.half_x + wall_t, 0.0, wall_h * 0.5], [wall_t, span_y, wall_h * 0.5]),
        ("terrain_wall_ymin", [0.0, -layout.half_y - wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
        ("terrain_wall_ymax", [0.0, layout.half_y + wall_t, wall_h * 0.5], [span_x, wall_t, wall_h * 0.5]),
    ]:
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=half,
            rgba=[0.30, 0.34, 0.38, 0.85],
            contype=1,
            conaffinity=1,
            condim=3,
        )

    # Visual markers for mobility checkpoints (mocap, no collision).
    markers = raw.get("checkpoints", [])
    for idx, pt in enumerate(markers):
        x, y = float(pt[0]), float(pt[1])
        z = sample_terrain_height(x, y, heights, layout) + cage_radius + 0.02
        body = world.add_body(name=f"terrain_checkpoint_{idx}", mocap=True, pos=[x, y, z])
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.10, 0.0, 0.0],
            rgba=[0.95, 0.55, 0.15, 0.9],
            contype=0,
            conaffinity=0,
        )

    model = spec.compile()
    sim = raw.get("simulation", {})
    model.opt.timestep = float(sim.get("timestep", 0.004))
    return model, layout, heights
