# CAROLLINE Control

MuJoCo implementation of CAROLLINE, a caged rolling drone with hybrid ground
and aerial locomotion. The ground controller implements the paper's
contact-frame rolling cascade, bidirectional thrust allocation, arbitrary-pose
recovery, ramp contact estimation, gravity compensation, and rolling hold.

## Features

- Omnidirectional cage rolling with bidirectional ±13 N rotor thrust
- Contact-point and support-normal estimation from MuJoCo contacts
- Rolling on flat ground, platforms, and inclined ramps
- SO(3)-based recovery from arbitrary cage orientations
- Geometric aerial position and attitude control
- Automatic rolling-stall recovery
- Live torque, motor-thrust, contact-normal, and saturation telemetry

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r carolline_control\requirements.txt
```

The modified Skydio X2 model required by the simulations is included under
`mujoco_menagerie-main/skydio_x2`.

## Run

Main rolling and flight mission:

```powershell
python -m carolline_control.main
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

## Tests

```powershell
python tests/test_ground_dynamics.py
python carolline_control/scripts/validate_recovery.py
python carolline_control/scripts/validate_monte_carlo.py
```

Negative rotor values are intentional: the CAROLLINE model uses bidirectional
thrust, and opposing rotor forces generate ground torque without unwanted
collective lift.

## Acknowledgements

- [CAROLLINE research paper](https://doi.org/10.1109/LRA.2021.3097253)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)