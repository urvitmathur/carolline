# Appendices

## Appendix A — Key configuration parameters

| Parameter | File | Value |
|-----------|------|-------|
| Hover height | config.yaml | 1.0 m |
| Motor limits | config.yaml | ±13 N |
| kx, kv | config.yaml | 4.5, 6.0 |
| kR, kOmega | config.yaml | 5.5, 1.6 |
| contact_loss_grace | config.yaml | 0.30 s |
| Maze arena | maze_config.yaml | 9 m square |
| Map resolution | maze_config.yaml | 0.03 m |

## Appendix B — Reproducibility commands

```powershell
python -m carolline_control.main --seed 42
python carolline_control/scripts/validate_recovery.py --trials 20
python carolline_control/scripts/validate_takeoff.py --trials 10
python carolline_control/scripts/simulate_slam_maze.py --no-viewer --oracle
python carolline_control/scripts/simulate_hybrid_maze.py --no-viewer
python carolline_control/plots/trajectory_tracking.py
python carolline_control/plots/report_figures/generate_report_figures.py
```

## Appendix C — Validation CSV locations

- `carolline_control/logs/flight_log.csv`
- `carolline_control/logs/recovery_validation.csv`
- `carolline_control/logs/takeoff_validation.csv`
- `carolline_control/logs/monte_carlo_validation.csv`

## Appendix D — Module index (abbreviated)

| Module | Role |
|--------|------|
| carolline_controller.py | Control integrator |
| controllers/mode_manager.py | Mode FSM |
| controllers/flight_controller.py | SE(3) flight |
| controllers/rolling_controller.py | Ground rolling |
| navigation/slam_stack.py | SLAM integration |
| navigation/director.py | Course FSM |
| scripts/simulate_slam_maze.py | Maze entry point |

Full index: `docs/curriculum/appendix-file-index.md`.

## Appendix E — LinkedIn portfolio

Insert screenshot showing project description added to LinkedIn profile (module requirement). Placeholder: `[LinkedIn screenshot to be inserted by author]`.

## Appendix F — Report metrics snapshot

See `docs/dissertation/report_data.yaml` for frozen numeric results used in Table 5.1.
