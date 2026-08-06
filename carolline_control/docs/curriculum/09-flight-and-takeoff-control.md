# Chapter 09 — Flight and Takeoff Control

**Previous:** [08-rolling-and-ground-control.md](08-rolling-and-ground-control.md) | **Next:** [10-actuation-pipeline.md](10-actuation-pipeline.md)

---

## Flight controller

File: [controllers/flight_controller.py](../../controllers/flight_controller.py)

Geometric controller (Lee et al. 2010, Eqs. 17–23).

### Core law

Position/velocity errors → desired force vector:

```
f_des = -kx*ex - kv*ev + m*a_ff   (horizontal)
f_des_z = -kx_z*ez - kv_z*ev_z + m*(g + a_ff_z)
```

Then:

1. Limit horizontal tilt via `MAX_TILT_SINE = 0.38`
2. Build body thrust direction `b3`
3. `thrust = dot(f_des, body_z_world(R))`
4. `desired_rotation = desired_rotation_from_thrust_direction(b3, yaw)`

### Mode-specific branches

| Mode | Extra logic |
|------|-------------|
| TAKEOFF | Min vertical force ramp; damp horizontal velocity from rolling |
| HOVER | Extra velocity damping when target vel ≈ 0 |
| LANDING | Yaw frozen to current heading |
| FLIGHT | Low-altitude tilt scaling near ground |

Safety: `min_flight_center_z`, `min_airborne_thrust_fraction` prevent ground scrape.

### Helper methods

- `takeoff_thrust()` — TAKEOFF mode wrapper
- `hover()` — fixed XY anchor
- `compute_velocity()` — teleop velocity tracking (no position lead)

---

## Orientation generator

File: [controllers/orientation_generator.py](../../controllers/orientation_generator.py)

- `hover_yaw(yaw)` — level attitude with specified heading
- Used when inverted or idle

---

## Tuning for smooth flight

From log analysis (see [Chapter 16](16-experiments-and-studies.md)):

| Parameter | Smoother flight |
|-----------|-----------------|
| Lower `kx`, `kv` | Less aggressive corrections |
| Longer `segment_duration` | Gentler references |
| Lower `motor_slew_rate` in flight | Less thrust stepping |

---

## Checkpoint

```powershell
python carolline_control/scripts/validate_takeoff.py
```

Trace PRETAKEOFF → UPRIGHT → TAKEOFF → HOVER in logs.

**Next:** [Chapter 10 — Actuation pipeline](10-actuation-pipeline.md)
