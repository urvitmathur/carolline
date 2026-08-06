# Chapter 4 — Control System Design

## 4.1 Introduction

This chapter presents the discrete mode manager, ground rolling controller, pre-takeoff recovery, geometric flight controller, actuation pipeline, and alternative LQR/MPC studies.

## 4.2 Mode manager

Operating modes follow CAROLLINE paper nomenclature: `PRETAKEOFF`, `UPRIGHT`, `TAKEOFF`, `HOVER`, `FLIGHT`, `ROLLING`, `LANDING`, `IDLE`. Figure 4.1 summarises principal transitions.

**Rolling completion** — When ground position error and speed fall below thresholds for 0.5 s, transition to `PRETAKEOFF` for uprighting before flight.

**Contact loss** — If ground contact is lost longer than `contact_loss_grace` (0.30 s) during rolling, the system enters `PRETAKEOFF` with `resume_rolling_after_recovery` set—preventing silent aerial drift.

**Rolling stall** — Low speed with large position error triggers recovery to avoid indefinite sliding against obstacles.

**Takeoff → hover** — Altitude within tolerance, upright attitude, velocity gates, and **at-or-below hover height** prevent locking in during overshoot.

## 4.3 Ground rolling controller

The rolling controller tracks commanded planar velocity \(\mathbf{v}_{des}\) from the planner or teleop. A PD law produces desired body angular rates; `ground_dynamics` maps these to motor thrusts via a contact-frame allocation matrix \(\mathbf{Q}\) such that \(\boldsymbol{\tau}_{contact} \approx \mathbf{Q} \mathbf{T}\) for motor thrust vector \(\mathbf{T}\).

Bidirectional thrust enables **braking and reverse** without reversing cage orientation. Roll-hold mode zeros velocity commands while maintaining contact torque for position holding on slopes.

## 4.4 Pre-takeoff recovery

From arbitrary orientations, the pre-takeoff controller drives body \(z\) toward world up using thrust fraction and SO(3) logarithm torque:

\[
\boldsymbol{\omega}_{des} = k_\omega \log(\mathbf{R}^T \mathbf{R}_{des})
\]

Ground allocation prioritises moment over collective thrust. Recovery success was defined when body-\(z\) alignment exceeded `upright_cos_threshold` (0.92) and angular rates fell below tolerance for `pre_upright_settle_time`.

## 4.5 Flight controller

Geometric position control computes desired force:

\[
\mathbf{f}_{des} = -k_x \mathbf{e}_x - k_v \mathbf{e}_v + m(\mathbf{g}\mathbf{e}_3 + \ddot{\mathbf{p}}_{des})
\]

with \(k_{x,z}\), \(k_{v,z}\) often larger on altitude. Thrust direction \(\mathbf{b}_3\) is derived from horizontal components with tilt limit (`MAX_TILT_SINE`). Attitude tracking uses Lee Eq. 11 torque law with gains \((k_R, k_\Omega)\).

**Takeoff tuning** — Approach braking fades climb assist as altitude error shrinks and applies extra damping when crossing hover height. Soft gains \((k_{R,pre}, k_{\Omega,pre})\) apply during `TAKEOFF` only; ground modes use dedicated allocators that return before the aerial gain branch.

## 4.6 Actuation pipeline

For aerial modes the cascade is:

1. `FlightController` → thrust + desired rotation \(\mathbf{R}_d\)
2. `AttitudeController` → desired body rate \(\boldsymbol{\omega}_d\)
3. `TorqueController` → moment \(\mathbf{m}\)
4. `MotorMixer.mix` → four thrusts
5. `EscThrustMapper` → PWM telemetry (thrust unchanged in sim)
6. Slew limiter → rate-limited `data.ctrl`

Saturation diagnostics record spread and clip flags for validation CSVs.

## 4.7 Alternative controllers

LQR and MPC variants (`FlightMpcController`, `RollingMpcController`) were benchmarked against geometric baselines. Results CSVs in `plots/lqr_comparison/` show comparable hover regulation with different tuning sensitivity. Geometric control remained the mission default for consistency with CAROLLINE literature.

## 4.8 Chapter discussion

Two integration bugs discovered during development illustrate the value of mode-aware design: (1) contact-loss timer incremented without recovery action until explicitly wired to `PRETAKEOFF`; (2) takeoff soft-gain branch was unreachable because rolling and recovery paths returned early from `_allocate`. Fixing these improved rolling safety and reduced takeoff Z overshoot from 0.29 m to under 0.01 m on seed 42. Remaining challenges include motor saturation during aggressive terrain hops and yaw authority limits on high-friction surfaces.
