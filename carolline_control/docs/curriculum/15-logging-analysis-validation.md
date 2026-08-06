# Chapter 15 — Logging, Analysis, and Validation

**Previous:** [14-scripts-and-entry-points.md](14-scripts-and-entry-points.md) | **Next:** [16-experiments-and-studies.md](16-experiments-and-studies.md)

---

## Flight logger

File: [logging/logger.py](../../logging/logger.py)

Writes CSV with columns for state, desired trajectory, errors, motors, attitude:

| Column group | Examples |
|--------------|----------|
| Pose | `px, py, pz, vx, vy, vz` |
| Desired | `des_px, des_py, des_pz, des_vx, ...` |
| Errors | `ex, ey, ez, evx, ...` |
| Attitude | `roll_deg, des_roll_deg, ...` |
| Motors | `m1, m2, m3, m4, thrust, motor_spread` |
| Flags | `motor_saturated, on_ground` |

Used by [main.py](../../main.py) default log at `logs/flight_log.csv`.

---

## Course logger extension

File: [navigation/course_logger.py](../../navigation/course_logger.py)

Adds mission columns:

- `phase`, `checkpoint`, `cmd_speed`, `forward_min_m`, `mission_blocked`

---

## Pipeline tracer

File: [logging/pipeline_tracer.py](../../logging/pipeline_tracer.py)

Optional debug (`--debug-pipeline` in main):

- Logs mode changes, saturation events, >10% control deltas
- Raises `PipelineAbort` on NaN/Inf

---

## Trajectory plots

File: [plots/trajectory_tracking.py](../../plots/trajectory_tracking.py)

`plot_actual_vs_desired(log_path, output_path)`:

- 3-panel X/Y/Z vs time
- Optional flight-only and rolling ground-track PNGs

Course variant: [course_analysis.py](../../navigation/course_analysis.py) adds phase shading and speed plot.

Output: `plots/course/` after autonomous run.

---

## Quantitative analysis

File: [scripts/quantitative_analysis.py](../../scripts/quantitative_analysis.py)

Reads enriched flight log → summary text + PNGs:

- Motor thrusts, attitude tracking, position errors, control effort

---

## Validation framework

Package: [validation/](../../validation/)

```powershell
python -m carolline_control.validation --campaign recovery --runs 50
```

| Module | Role |
|--------|------|
| [cli.py](../../validation/cli.py) | Argument parsing |
| [campaigns.py](../../validation/campaigns.py) | Campaign definitions |
| [runner.py](../../validation/runner.py) | Execute single run |
| [monte_carlo.py](../../validation/monte_carlo.py) | Batch runs |
| [metrics.py](../../validation/metrics.py) | Per-run metrics |
| [statistics.py](../../validation/statistics.py) | Aggregate stats + plots |
| [report.py](../../validation/report.py) | PDF report generation |
| [observer.py](../../validation/observer.py) | Step-by-step recorder |
| [disturbances.py](../../validation/disturbances.py) | Wind, pushes, etc. |
| [randomization.py](../../validation/randomization.py) | Parameter sweeps |

Config: [validation/validation.yaml](../../validation/validation.yaml)

Results stored under `validation/results/`.

---

## Log path helpers

File: [logging/paths.py](../../logging/paths.py)

- `resolve_repo_path()` — relative to repo root
- `fallback_log_path()` — timestamped backup if file locked (Windows/Excel)

---

## Checkpoint

After a course run, open:

- `carolline_control/logs/course_flight_log.csv`
- `carolline_control/plots/course/course_actual_vs_desired.png`

Compare `px` vs `des_px` during `FLIGHT` rows.

**Next:** [Chapter 16 — Experiments and studies](16-experiments-and-studies.md)
