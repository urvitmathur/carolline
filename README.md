# CAROLLINE Control

MuJoCo implementation of CAROLLINE, a caged rolling drone with hybrid ground
and aerial locomotion. The ground controller implements the paper's
contact-frame rolling cascade, bidirectional thrust allocation, arbitrary-pose
recovery, ramp contact estimation, gravity compensation, and rolling hold.

## Simulation scope

This repository is a **physics-based MuJoCo simulation** of CAROLLINE. It
implements the paper's rolling and pre-takeoff controllers plus a full hybrid
ground-to-flight mission stack. The following are intentionally out of scope:

- Physical robot hardware (motors, ESCs, custom flight controller)
- Onboard embedded firmware and wireless ground control
- Motion-capture fusion on real hardware

For simulation work, bidirectional thrust is commanded in Newtons through an
optional ESC mapping layer (paper Sec. IV-C, Fig. 5) before actuators.

## Features

- Omnidirectional cage rolling with bidirectional ±13 N rotor thrust
- Ribbed spherical cage model with vertical battery placement (`scene_cage.xml`)
- Contact-point and support-normal estimation from MuJoCo contacts
- Bidirectional ESC thrust mapping with reverse-thrust compensation
- Rolling on flat ground, platforms, and inclined ramps
- SO(3)-based recovery from arbitrary cage orientations
- Geometric aerial position and attitude control
- Automatic rolling-stall recovery
- Ground vs flight energy / COT benchmark (paper Sec. V-C)
- Live torque, motor-thrust, contact-normal, and saturation telemetry

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r carolline_control\requirements.txt
```

The modified Skydio X2 model required by the simulations is included under
`mujoco_menagerie-main/skydio_x2`. The default scene uses a smooth spherical
cage (`scene.xml`). For ribbed cage visuals, set `model_path` to
`scene_cage.xml` in `config.yaml` (uses a single collision sphere for stable rolling).

## Run

Main rolling and flight mission:

```powershell
python -m carolline_control.main
python -m carolline_control.main --sensor-only --no-viewer --seed 42
```

Ramp and platform mission:

```powershell
python carolline_control/scripts/simulate_ramp_platform.py
```

Inclined-ramp position hold:

```powershell
python carolline_control/scripts/simulate_30deg_ramp_hold.py
```

Hybrid raised-platform mission:

```powershell
python carolline_control/scripts/simulate_hybrid_locomotion.py
```

Manual keyboard test rig (replicate video demos — rolling, recovery, flight):

```powershell
python carolline_control/scripts/manual_teleop.py
python carolline_control/scripts/manual_teleop.py --scene ramp
python carolline_control/scripts/manual_teleop.py --scene course
python carolline_control/scripts/manual_teleop.py --scene platform
```

Use **arrow keys** to roll (hold to move, X to stop). **1–8** for modes. **Page Up/Down** for vertical in air; **Q/E** yaw; **R** roll-hold; **/** help.
MuJoCo viewer UI is enabled (joints, controls, visualization toggles).

RC motor teleop (direct transmitter → motors, bypasses velocity controller):

```powershell
python carolline_control/scripts/rc_motor_teleop.py
python carolline_control/scripts/rc_motor_teleop.py --mode acro
python carolline_control/scripts/rc_motor_teleop.py --list-joysticks
```

**direct** mode: per-motor trim on numpad (1/2, 4/5, 7/8, 3/6). **acro** mode: W/S throttle, **arrow keys** pitch/roll, Q/E yaw.
**H** = hover thrust, **0** or **X** = zero motors. Optional USB gamepad via `--joystick 0`.

## LQR comparative study (geometric vs LQR)

Side-by-side benchmarks for your comparative analysis:

```powershell
python carolline_control/scripts/compare_controllers.py
python carolline_control/scripts/compare_flight_controllers.py
python carolline_control/scripts/compare_rolling_controllers.py
python tests/test_lqr_gains.py
```

Outputs land in `carolline_control/plots/lqr_comparison/` (CSV metrics + plots).
Tune LQR weights in `carolline_control/lqr_config.yaml`.

| Domain | Geometric baseline | LQR variant |
|--------|-------------------|-------------|
| Aerial | Lee SO(3) position + attitude PD | 12-state hover LQR on wrench |
| Ground | Paper rolling PD + allocation | 5-state velocity/rate LQR on torque |

## Tests and validation

```powershell
python tests/test_ground_dynamics.py
python tests/test_esc_mapper.py
python tests/test_state_estimator.py
python carolline_control/scripts/validate_recovery.py
python carolline_control/scripts/validate_monte_carlo.py
python carolline_control/scripts/validate_energy_cot.py
```

### Sensor-only estimation (hardware-realistic mode)

By default the simulator reads pose from MuJoCo `qpos` (oracle mode). Use
`--sensor-only` to drive controllers from IMU sensors only:

- Orientation: `body_quat`
- Angular rate: `body_gyro`
- Velocity: `body_vel`
- Acceleration: `body_linacc`
- Position: dead-reckoned from known spawn + velocity integration

```powershell
python -m carolline_control.main --sensor-only --no-viewer --seed 42
python carolline_control/scripts/validate_recovery.py --sensor-only --trials 20
python carolline_control/scripts/validate_monte_carlo.py --sensor-only --trials 10
python -m carolline_control.validation --sensor-only --campaign nominal --runs 5
```

Negative rotor values are intentional: the CAROLLINE model uses bidirectional
thrust, and opposing rotor forces generate ground torque without unwanted
collective lift.

## Acknowledgements

- [CAROLLINE research paper](https://doi.org/10.1109/LRA.2025.3608651)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
