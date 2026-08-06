# Chapter 11 — Alternative Controllers (LQR and MPC)

**Previous:** [10-actuation-pipeline.md](10-actuation-pipeline.md) | **Next:** [12-autonomous-navigation.md](12-autonomous-navigation.md)

---

## Purpose

The **default stack** uses geometric PD controllers (Lee 2010). The repo also includes **LQR** and **MPC** implementations for **benchmarking and research**—they are not used in `main.py` or the autonomous course by default.

---

## LQR package

Directory: [controllers/lqr/](../../controllers/lqr/)

| File | Role |
|------|------|
| `flight_lqr.py` | Linearized flight LQR |
| `rolling_lqr.py` | Rolling LQR |
| `dare_solver.py` | Discrete algebraic Riccati equation |
| `config_loader.py` | Loads [lqr_config.yaml](../../lqr_config.yaml) weights |
| `comparison_metrics.py` | Study metrics |

Config: state/input weight matrices `Q`, `R` per mode.

---

## MPC package

Directory: [controllers/mpc/](../../controllers/mpc/)

| File | Role |
|------|------|
| `flight_mpc.py` | Finite-horizon flight MPC |
| `rolling_mpc.py` | Rolling MPC |
| `finite_horizon.py` | QP solver wrapper |
| `config_loader.py` | [mpc_config.yaml](../../mpc_config.yaml) |

---

## Comparison scripts

| Script | Purpose |
|--------|---------|
| [compare_controllers.py](../../scripts/compare_controllers.py) | Geometric vs LQR vs MPC |
| [compare_flight_controllers.py](../../scripts/compare_flight_controllers.py) | Flight-only |
| [compare_rolling_controllers.py](../../scripts/compare_rolling_controllers.py) | Rolling-only |
| [run_lqr_study.py](../../scripts/run_lqr_study.py) | Batch LQR study → CSV in `plots/lqr_comparison/` |
| [run_controller_study.py](../../scripts/run_controller_study.py) | Combined study |

Results land in `carolline_control/plots/controller_comparison/` and `plots/lqr_comparison/`.

---

## When to read source

- Tuning default geometric gains → Chapters 09–10, not LQR/MPC.
- Writing a paper comparison → read `flight_lqr.py` / `flight_mpc.py` and comparison scripts.
- Unit tests: [tests/test_mpc_gains.py](../../tests/test_mpc_gains.py), parent `tests/test_lqr_gains.py`.

---

## Checkpoint

Skim `compare_controllers.py` CLI help:

```powershell
python carolline_control/scripts/compare_controllers.py --help
```

**Next:** [Chapter 12 — Autonomous navigation](12-autonomous-navigation.md)
