# Chapter 14 — Scripts and Entry Points

**Previous:** [13-terrain-and-scenes.md](13-terrain-and-scenes.md) | **Next:** [15-logging-analysis-validation.md](15-logging-analysis-validation.md)

---

## Primary entry points

| Script | Command | Purpose |
|--------|---------|---------|
| [main.py](../../main.py) | `python -m carolline_control.main` | Default hybrid mission |
| [validation/__main__.py](../../validation/__main__.py) | `python -m carolline_control.validation` | Monte Carlo CLI |

---

## Simulation scripts

| Script | Purpose |
|--------|---------|
| [simulate_autonomous_course.py](../../scripts/simulate_autonomous_course.py) | Full terrain course + director + log/plot/record |
| [simulate_hybrid_locomotion.py](../../scripts/simulate_hybrid_locomotion.py) | Hybrid ground-flight mission variant |
| [simulate_terrain_mobility.py](../../scripts/simulate_terrain_mobility.py) | Rolling on generated terrain |
| [simulate_ramp_platform.py](../../scripts/simulate_ramp_platform.py) | Inclined ramp platform |
| [simulate_30deg_ramp_hold.py](../../scripts/simulate_30deg_ramp_hold.py) | 30° ramp hold test |

---

## Teleop / debug

| Script | Purpose |
|--------|---------|
| [manual_teleop.py](../../scripts/manual_teleop.py) | Keyboard mode + velocity teleop, tracking camera |
| [rc_motor_teleop.py](../../scripts/rc_motor_teleop.py) | Direct per-motor thrust keyboard |
| [debug_rolling.py](../../scripts/debug_rolling.py) | Rolling contact diagnostics |
| [debug_mission.py](../../scripts/debug_mission.py) | Mission planner debug |

---

## Validation runners

| Script | Purpose |
|--------|---------|
| [validate_recovery.py](../../scripts/validate_recovery.py) | PRETAKEOFF recovery from random poses |
| [validate_takeoff.py](../../scripts/validate_takeoff.py) | Takeoff phase validation |
| [validate_upright_stabilization.py](../../scripts/validate_upright_stabilization.py) | Upright settle tests |
| [validate_monte_carlo.py](../../scripts/validate_monte_carlo.py) | Monte Carlo batch (script wrapper) |
| [validate_energy_cot.py](../../scripts/validate_energy_cot.py) | Energy / cost-of-transport benchmark |

---

## Controller comparison / studies

| Script | Purpose |
|--------|---------|
| [compare_controllers.py](../../scripts/compare_controllers.py) | Geometric vs LQR vs MPC |
| [compare_flight_controllers.py](../../scripts/compare_flight_controllers.py) | Flight controllers only |
| [compare_rolling_controllers.py](../../scripts/compare_rolling_controllers.py) | Rolling controllers only |
| [run_lqr_study.py](../../scripts/run_lqr_study.py) | LQR parameter study |
| [run_controller_study.py](../../scripts/run_controller_study.py) | Combined controller study |

---

## Analysis

| Script | Purpose |
|--------|---------|
| [quantitative_analysis.py](../../scripts/quantitative_analysis.py) | Flight log stats + diagnostic PNGs |
| [analyze_log.py](../../scripts/analyze_log.py) | Quick rolling log summary |
| [analyze_hover.py](../../scripts/analyze_hover.py) | Hover stability analysis |
| [system_analysis.py](../../scripts/system_analysis.py) | System-level log analysis |
| [rolling_ramp_analysis.py](../../scripts/rolling_ramp_analysis.py) | Ramp experiment analysis |

---

## Common CLI flags

| Flag | Scripts | Meaning |
|------|---------|---------|
| `--no-viewer` | Most sims | Headless, faster |
| `--sensor-only` | main, course, validation | IMU-only state est |
| `--config PATH` | Most | Override config.yaml |
| `--record` | autonomous course | Save MP4 |
| `--log PATH` | main, course | CSV output path |

---

## Checkpoint

Pick three scripts from different categories and run `--help` on each.

**Next:** [Chapter 15 — Logging, analysis, validation](15-logging-analysis-validation.md)
