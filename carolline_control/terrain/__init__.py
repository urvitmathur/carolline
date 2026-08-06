"""Procedural terrain scenes for CAROLLINE mobility testing."""

from carolline_control.terrain.heightmap import TerrainLayout, generate_heightmap
from carolline_control.terrain.scene import apply_terrain_mobility_tuning, compile_terrain_scene, load_terrain_config, sample_terrain_height

__all__ = [
    "TerrainLayout",
    "apply_terrain_mobility_tuning",
    "compile_terrain_scene",
    "generate_heightmap",
    "load_terrain_config",
    "sample_terrain_height",
]
