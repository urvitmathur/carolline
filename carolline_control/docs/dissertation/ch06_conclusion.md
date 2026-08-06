# Chapter 6 — Conclusion

## 6.1 Achievements

The project achieved the following against the objectives set in Chapter 1:

| Objective | Achievement |
|-----------|-------------|
| O1 — Rolling to target | Completed in hybrid mission; ROLLING → PRETAKEOFF at 4.59 s |
| O2 — Arbitrary recovery | 100% success over 20 trials, tilts up to 172° |
| O3 — Controlled takeoff | 100% hover achievement; Z overshoot reduced to negligible on tuned seed |
| O4 — Flight and landing | Full mission to IDLE in 56.3 s (seed 42) |
| O5 — Autonomous course | Implemented director, perception, terrain scenes (see codebase) |
| O6 — Maze SLAM | Oracle rolling mission reached goal in 105.6 s; maps generated |
| O7 — Hybrid maze | Perception-driven WALL/GAP/CLIMB takeoffs; goal in 94.9 s, 0.40 m error |

Deliverables include the MuJoCo simulation package, validation suite, trajectory and ESC characterisation figures, hybrid maze report plots, and this report.

## 6.2 Discussion

The work confirmed that CAROLLINE’s hybrid behaviour is **reproducible in simulation** with modular Python controllers mirroring the published architecture. The original contributions—autonomous course FSM, exploration SLAM, Gazebo-style maze rolling SLAM, and the obstacle-driven hybrid maze—integrate above the control stack through mode requests rather than ad hoc motor commands, preserving safety interlocks.

The hybrid maze result is particularly significant: three distinct obstacle classes were detected by forward/up/down rangefinders and each triggered exactly one fly-over sequence, demonstrating that geographic filtering and confirmation timers can prevent spurious takeoffs without sacrificing responsiveness.

Problems encountered included: exploration fly-over altitude ratcheting; SLAM odometry replacing full attitude; maze replan spam; unreachable takeoff soft-gain branch; and `print_estimator_mode` API mismatches in entry scripts. Each was diagnosed through logged mode timelines and fixed with minimal diffs—consistent with professional simulation practice.

Sensor-only Monte Carlo failure highlights the **estimation gap**: without global pose corrections, long missions do not complete even when local controllers validate at 100%. This is an honest limitation for a simulation-only MSc project and motivates future MoCap or GPS fusion work.

## 6.3 Conclusions

1. A complete CAROLLINE hybrid locomotion stack was implemented in MuJoCo with rolling, recovery, flight, and landing validated on reference seeds.
2. Bidirectional ESC and contact-frame rolling allocation were modelled sufficiently for controller tuning and energy comparisons.
3. Mission directors and a Python SLAM pipeline enabled rolling exploration of walled courses and orthogonal mazes without ROS.
4. The obstacle-driven hybrid maze completed in 94.9 s via WALL, GAP, and CLIMB takeoff triggers—demonstrating perception-driven hybrid autonomy beyond scripted flight segments.
5. Quantitative validation (recovery, takeoff, trajectory tracking) met success criteria under oracle estimation; full sensor-only mission robustness was not achieved in archived Monte Carlo runs.
6. The simulation repository provides a reproducible baseline for hardware migration and further autonomy research.

## 6.4 Future work

Recommended extensions include:

- **Hardware transfer** — Flash real BLHeli curves; validate contact estimation on physical cage.
- **Persistent SLAM maps** — Load saved occupancy grids on subsequent runs (`--load-map`).
- **Closed-loop maze completion** — Tune `--slam-nav` until goal reached without oracle pose.
- **Hybrid maze under sensor-only** — Repeat WALL/GAP/CLIMB course with integrated SLAM odometry.
- **Ribbed cage collision** — Replace single-sphere approximation with segmented contact geometry.
- **MoCap / GPS fusion** — Replace oracle mode for long-horizon missions.
- **LinkedIn project portfolio** — Post summary with diagram and link to repository (appendix requirement).

## 6.5 Final remarks

The project delivered a distinction-calibre integration of hybrid robot control and autonomous navigation in simulation, with clear separation between reproduced paper behaviour and original SLAM and hybrid-maze extensions. The codebase, validation logs, and figures accompanying this report enable independent reproduction using documented seeds and configuration files.
