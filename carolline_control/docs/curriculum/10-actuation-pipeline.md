# Chapter 10 — Actuation Pipeline

**Previous:** [09-flight-and-takeoff-control.md](09-flight-and-takeoff-control.md) | **Next:** [11-alternative-controllers-lqr-mpc.md](11-alternative-controllers-lqr-mpc.md)

---

## Signal flow (aerial modes)

```
ControlCommand (thrust, Rd)
    → AttitudeController.compute_desired_omega
    → TorqueController.compute (e_R, e_Omega)
    → MotorMixer.mix(thrust, moment)
    → EscThrustMapper
    → motor slew limiter
    → data.ctrl[0:4]
```

---

## Attitude controller

File: [controllers/attitude_controller.py](../../controllers/attitude_controller.py)

- Flight: `omega_d_ff = 0`; regulation via torque `-kR * e_R`
- Rolling: passes through `desired_omega_body` from rolling controller

---

## Torque controller

File: [controllers/torque_controller.py](../../controllers/torque_controller.py)

Implements Lee Eq. 11:

```
moment = -kR * e_R - kOmega * e_Omega + omega × (I * omega)
```

Uses `attitude_error(R, Rd)` from so3.py.

---

## Motor mixer

File: [controllers/motor_mixer.py](../../controllers/motor_mixer.py)

4×4 wrench matrix `W` from rotor positions and yaw drag coefficients:

- Row 0: collective thrust
- Rows 1–3: roll, pitch, yaw moments

### `mix(thrust, moment)`

1. Start at equal split thrust/4
2. Add moment correction via pseudoinverse
3. Iteratively clip to `[motor_min, motor_max]` while preserving collective

### `mix_prioritize_moment`

Full wrench inverse for ground rolling when collective ≈ 0.

---

## ESC mapper

File: [controllers/esc_mapper.py](../../controllers/esc_mapper.py)

Maps ideal Newton thrust → ESC PWM curve (paper Fig. 5):

- Forward max 13 N, reverse max 10 N
- `reverse_efficiency` scales reverse thrust

Enabled via `esc_mapping_enabled` in config.

---

## Motor slew rate

In [carolline_controller.py](../../carolline_controller.py) `_slew_motors()`:

```python
max_step = motor_slew_rate * dt
```

Course terrain tuning raises this to 1400 N/s for snappy rolling; lowering it in flight reduces thrust chatter.

---

## Saturation diagnostics

`ControlDiagnostics.motor_saturated` — true when any motor hits min/max after mixing.

Logged as column in flight CSV; high saturation in PRETAKEOFF recovery is normal.

---

## Checkpoint

Open a flight log and plot columns `m1`–`m4` vs `thrust`. See [Chapter 15](15-logging-analysis-validation.md).

**Next:** [Chapter 11 — LQR and MPC](11-alternative-controllers-lqr-mpc.md)
