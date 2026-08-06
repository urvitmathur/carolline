# Chapter 08 — Rolling and Ground Control

**Previous:** [07-mode-manager-and-planner.md](07-mode-manager-and-planner.md) | **Next:** [09-flight-and-takeoff-control.md](09-flight-and-takeoff-control.md)

---

## Rolling controller

File: [controllers/rolling_controller.py](../../controllers/rolling_controller.py)

Implements paper **Sec. III-A**: attitude-invariant velocity tracking at the cage contact point.

### `compute(state, desired_velocity_xy, ...)`

1. Clamp desired speed to `rolling_max_speed`.
2. If velocity command ~0 and not holding → zero torque (coast on friction).
3. Else: compute desired contact angular rate via [ground_dynamics.py](../../controllers/ground_dynamics.py).
4. Return `ControlCommand` with **thrust=0**, `moment_body=tracking_torque`.

Rolling uses **torque only**—no collective lift until takeoff.

### `allocate(state, command)`

Maps body moment to four motor thrusts using contact-frame pseudoinverse (`mixer.mix_prioritize_moment`).

---

## Ground dynamics

File: [controllers/ground_dynamics.py](../../controllers/ground_dynamics.py)

- Finds cage-terrain contacts from MuJoCo.
- Estimates support normal and contact point relative to COM.
- `desired_omega()` — no-slip rolling kinematics (links to `rolling_omega_world` in so3.py).
- `tracking_torque()` — PD on angular velocity error with `rolling_omega_kp`.

---

## Pre-takeoff / recovery

File: [controllers/pre_takeoff_controller.py](../../controllers/pre_takeoff_controller.py)

Used in **PRETAKEOFF** and **UPRIGHT** modes:

- Drives body-z toward world-up using body-rate control.
- `pre_takeoff_omega_kp`, `pre_takeoff_omega_limit` cap spin rates.
- `allocate()` distributes recovery torque across motors (may use reverse thrust).

Triggered when:

- Wall blocked → director `request_takeoff()`
- Landing while tilted → mode manager back to PRETAKEOFF
- Rolling stall timeout

---

## Stall detection (mode manager)

| Config key | Role |
|------------|------|
| `rolling_stall_speed` | Below this speed counts as stalled |
| `rolling_stall_time` | Seconds before forced recovery |

Director has separate stall logic in `_rolling_stuck()` for terrain hops.

---

## Navigation rolling follower

See [Chapter 12](12-autonomous-navigation.md). The follower produces `desired_velocity_xy`; rolling controller executes it.

---

## Checkpoint

```powershell
python carolline_control/scripts/debug_rolling.py
```

Observe contact normal and motor spread in output.

**Next:** [Chapter 09 — Flight and takeoff control](09-flight-and-takeoff-control.md)
