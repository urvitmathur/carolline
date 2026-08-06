# Chapter 13 — Terrain and Scenes

**Previous:** [12-autonomous-navigation.md](12-autonomous-navigation.md) | **Next:** [14-scripts-and-entry-points.md](14-scripts-and-entry-points.md)

---

## Terrain heightmap

File: [terrain/heightmap.py](../../terrain/heightmap.py)

Generates mixed terrain zones along +X:

| Zone | X range | Character |
|------|---------|-----------|
| Flat | x < flat_x_end (-7) | Spawn / easy rolling |
| Hills | flat_x_end → hills_x_end (2) | Rolling difficulty, ditches |
| Mountains | beyond hills_x_end | Steep (mostly perimeter) |

**`sample_terrain_height(x, y, heights, layout)`** — bilinear lookup from PNG heightmap.

**`TerrainLayout`** — grid size, max height, seed for reproducibility.

---

## Terrain scene builder

File: [terrain/scene.py](../../terrain/scene.py)

| Function | Role |
|----------|------|
| `load_terrain_config()` | Read [terrain_config.yaml](../../terrain_config.yaml) |
| `compile_terrain_scene()` | Inject heightfield into MuJoCo XML |
| `apply_terrain_mobility_tuning()` | Boost rolling gains from YAML `mobility` section |

Used by `simulate_terrain_mobility.py` and indirectly by course scene.

---

## Course scene integration

[navigation/course_scene.py](../../navigation/course_scene.py) calls terrain compilation then:

- Adds wall geoms positioned with `sample_terrain_height()` so walls sit on local ground
- Registers rangefinder sites on cage ribs

---

## Rolling ramp experiment

Directory: [rolling_ramp/](../../rolling_ramp/)

Separate inclined-ramp study (not the autonomous course):

| File | Role |
|------|------|
| `path.py` | Ramp centerline geometry |
| `scene.py` | MuJoCo ramp platform scene |
| `controller.py` | Ramp-specific rolling assist |
| `plots.py` | Ramp experiment plots |

Config: [rolling_ramp_config.yaml](../../rolling_ramp_config.yaml)  
Script: [simulate_ramp_platform.py](../../scripts/simulate_ramp_platform.py)

---

## Checkpoint

Run terrain mobility demo:

```powershell
python carolline_control/scripts/simulate_terrain_mobility.py
```

**Next:** [Chapter 14 — Scripts and entry points](14-scripts-and-entry-points.md)
