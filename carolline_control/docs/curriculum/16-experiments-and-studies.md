# Chapter 16 — Experiments, Studies, and Tuning

**Previous:** [15-logging-analysis-validation.md](15-logging-analysis-validation.md) | **Next:** [appendix-file-index.md](appendix-file-index.md)

---

## Validation campaigns

Paper-style experiments live in [validation/validation.yaml](../../validation/validation.yaml):

| Campaign | Script | Measures |
|----------|--------|----------|
| Recovery | `validate_recovery.py` | Upright success from random tilt |
| Takeoff | `validate_takeoff.py` | Phase timing, altitude error |
| Monte Carlo | `validate_monte_carlo.py` | Success rate, tracking RMS |
| Energy / COT | `validate_energy_cot.py` | Ground vs flight energy |

Outputs in `logs/*_validation.csv`, `logs/*_paper.csv`.

---

## Controller comparison studies

Results CSVs:

| Path | Content |
|------|---------|
| `plots/lqr_comparison/study_results.csv` | LQR study |
| `plots/controller_comparison/study_geometric_lqr_mpc.csv` | Three-way compare |

Run fresh studies:

```powershell
python carolline_control/scripts/run_lqr_study.py
python carolline_control/scripts/run_controller_study.py
```

---

## Autonomous course tuning (from log analysis)

Typical improvements for smoother flight:

| Lever | File | Suggestion |
|-------|------|------------|
| Slower flight references | `config.yaml` `segment_duration` | 6 → 10–12 s for wall hops |
| Softer tracking | `config.yaml` `kx`, `kv` | Reduce ~25% |
| Attitude damping | `kOmega` | Increase slightly |
| Motor smoothness | `motor_slew_rate` | Lower in flight (400–600 N/s) |
| Ditch hop altitude | `course_layout.terrain_hop_waypoints` | +0.15 m cruise clearance |
| Rolling on hills | `course_config.yaml` `mobility` | Already tuned for terrain |

Wall 1 fly-over historically shows largest position error (~2 m peak); terrain hop cruise is relatively smooth (~0.45 m RMS).

---

## Recommended learning path after this curriculum

1. **Modify a checkpoint** in `course_config.yaml` — rerun course sim.
2. **Add a third wall** — update `course_layout` leg indices and test perception filter.
3. **Tune one gain** — log before/after with `course_analysis` plots.
4. **Run recovery Monte Carlo** — quantify upright success rate.
5. **Compare LQR vs geometric** on same mission segment.

---

## Where to get help in code

| Question | Read first |
|----------|------------|
| Why did mode change? | `mode_manager.py`, console logs |
| Why did director fly? | `director.py` `_blocked_timer`, `perception.py` |
| Why motors saturated? | `motor_mixer.py`, `ControlDiagnostics` |
| Why rolling slow? | `rolling_follower.py` grade scale, terrain |
| Full file list | [appendix-file-index.md](appendix-file-index.md) |

---

## Course complete

You now have a map of the entire CAROLLINE codebase. Re-read chapters as you edit code; use the appendix as a lookup table.

**Next:** [Appendix — File index](appendix-file-index.md)
