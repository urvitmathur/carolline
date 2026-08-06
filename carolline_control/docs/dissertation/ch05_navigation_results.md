# Chapter 5 — Autonomous Navigation, SLAM, and Results

## 5.1 Introduction

This chapter describes mission directors, rangefinder perception, the SLAM stack, maze environments, and quantitative experimental results from validation scripts and reference missions. Results are grouped into: (i) scripted hybrid locomotion (`main.py`), (ii) autonomous terrain course navigation, (iii) rolling-only SLAM maze, and (iv) the original obstacle-driven hybrid maze.

## 5.2 Mission architecture

Two director families orchestrate high-level behaviour above `CarollineController`.

The **HybridCourseDirector** implements a finite-state machine over phases such as `ROLL_TO_CHECKPOINT`, fly-over, terrain hop, and final approach. Each cycle:

1. `RangePerception.scan` produces an `ObstacleScan` (forward minimum range, upward clearance, blocked and flyable flags).
2. `filter_scan_for_mission` removes false blocks near goal approach or after walls are cleared.
3. The director updates phase timers and calls `mode_manager.request_takeoff()` / `request_flight()` as needed.
4. `TerrainRollingFollower` supplies the `rolling_velocity` hook to the controller.

The **HybridMazeDirector** extends this pattern for a roll/fly maze with phases `ROLL`, `RECOVER`, `FLY`, `LAND`, and `DONE`. Rolling targets advance through geographic checkpoints (dead-end spur, hop channel, gap trench, platform shaft); takeoff is triggered only when rangefinder classification confirms an obstacle class for 0.35 s. After each flight segment the vehicle lands, resumes rolling, and marks the cleared obstacle type so the same trigger cannot fire twice.

The **ExplorationDirector** and **RollingSlamDirector** cover unknown environments: map while rolling, replan with A* when blocked, and optionally fly over obstacles (exploration) or remain rolling-only (maze).

## 5.3 Perception

`RangePerception` mounts twenty-four forward rangefinders spanning ±75° (6 m range), one upward sensor for fly-over clearance (> 0.9 m typical), and one **downward** sensor for floor-gap and platform-edge detection.

From raw rays the scan derives:

- **wall_block** — forward obstruction with sufficient upward clearance to fly over;
- **floor_gap** — low altitude with downward range ≥ 1.5 m (trench or drop);
- **climb_face** — blocked ahead with upward clearance while geographically near the raised platform.

A ray is **blocked** if forward range < 0.55 m; **flyable** if blocked ahead but upward clearance exceeds threshold. `filter_scan_for_hybrid_maze` applies checkpoint context and cleared-obstacle memory so hop and climb triggers fire only in the correct map regions.

## 5.4 Scripted hybrid mission (baseline)

The reference hybrid mission (`main.py --seed 42`) executed roll → recovery → takeoff → hover → cardinal waypoint flight → landing in **56.3 s**, reaching IDLE with takeoff peak altitude 0.995 m (target 1.0 m). Figure 5.1 shows actual versus desired position; altitude tracking improved after takeoff approach braking.

## 5.5 Recovery and takeoff validation

Recovery validation (`validate_recovery.py`, 20 trials) achieved **100% success** from tilts between 26° and 172° with mean recovery time **1.52 s**. Takeoff validation (`validate_takeoff.py`, 10 trials) achieved **100% hover** with mean time-to-hover **2.06 s**. Pre-brake tuning exhibited mean vertical overshoot **0.289 m**; after approach braking on seed 42 overshoot was reduced to negligible levels. These unit tests isolate low-level control from navigation estimation.

## 5.6 Energy and cost-of-transport

The energy script (`validate_energy_cot.py`) compares rolling versus flight power proxies along matched paths. Results support the qualitative claim that ground rolling is energetically favourable where terrain permits—consistent with CAROLLINE paper Section V-C motivation. LQR and MPC comparison scripts in `plots/lqr_comparison/` provide supplementary controller metrics.

## 5.7 Obstacle-driven hybrid maze

The hybrid maze (`simulate_hybrid_maze.py`) is a 12 m × 8 m arena with internal walls, a low wall hop, a floor gap trench, and a 1.2 m raised platform. The vehicle must map the dead-end spur, fly over the low wall (**WALL**, t ≈ 8.2 s), hop the gap (**GAP**, t ≈ 42.6 s), and climb to the platform (**CLIMB**, t ≈ 66.0 s) before rolling to the goal.

The reference run completed successfully in **94.9 s** with final goal distance **0.40 m**. Thirteen phase transitions were logged (three recover→fly→land cycles plus final roll to DONE). Mode-manager states (`ROLLING`, `PRETAKEOFF`, `TAKEOFF`, `HOVER`, `FLIGHT`, `LANDING`) nested inside director phases as expected.

Figure 5.3 shows the XY trajectory with wall geometry and checkpoint labels; the path alternates between ground rolling (low curvature segments) and aerial arcs over obstacles. Figure 5.4 plots altitude versus time with horizontal bands marking flight segments; peak cruise altitude remained below 2.5 m. Figure 5.5 presents the director phase timeline (ROLL / RECOVER / FLY / LAND / DONE) aligned with takeoff reason annotations. Figure 5.6 compares actual and desired XYZ position, showing tight tracking during hover and flight legs and larger lateral error only during rolling approach to narrow gap waypoints.

## 5.8 Autonomous terrain course

The autonomous course (`simulate_autonomous_course.py`) uses `HybridCourseDirector` on heightfield and platform scenes. Checkpoints, wall fly-overs, and terrain hops follow the same perception pipeline as Section 5.3 but with scripted checkpoint sequencing rather than maze-specific geographic filters. This mission validated integration of `TerrainRollingFollower` with the mode stack prior to the hybrid maze benchmark.

## 5.9 SLAM maze (rolling-only)

The SLAM stack integrates planar odometry EKF, log-odds occupancy mapping with Bresenham ray casting, scan-to-map localisation over \((x, y, \psi)\) windows, and A* planning on an inflated grid.

The maze is a 9 m square arena with orthogonal wooden walls, nine internal segments, and a 1.5 m entrance gap on the north side. The vehicle rolls only—no flight phase—mapping with rangefinders and replanning on blocked/stuck timers.

Reference oracle run: **goal reached in 105.6 s**, final distance to goal **0.45 m**, success flag set. Figure 5.2 compares the ground-truth maze with the cleaned SLAM occupancy grid along the traversed rolling path (oracle mapping pose).

## 5.10 Oracle versus sensor-only navigation

Oracle pose (`StateEstimator` reading MuJoCo `qpos`) was used for controller development and SLAM map debugging. `--sensor-only` mode integrates IMU attitude and velocity without global corrections. `--slam-nav` fuses planar SLAM odometry for closed-loop navigation.

Monte Carlo full-mission campaigns under `--sensor-only` showed **0% success** on the archived 30-trial batch—indicating that integrated sensor drift and allocator stress require further tuning before claiming robustness. Recovery and takeoff unit validations remained at 100% under oracle estimation, separating **low-level control reliability** from **integrated navigation robustness**.

## 5.11 Experimental summary

Table 5.1 summarises validation metrics extracted from project logs (`report_data.yaml`, seed 42 where applicable).

**Table 5.1: Validation summary**

| Experiment | Metric | Result |
|------------|--------|--------|
| Hybrid mission (`main.py`) | Mission duration | 56.3 s |
| Hybrid mission | Takeoff → hover | 6.55 s → 7.28 s |
| Hybrid mission | Takeoff Z peak | 0.995 m (target 1.0 m) |
| Hybrid mission | Final mode | IDLE |
| Recovery (20 trials) | Success rate | 100% |
| Recovery | Tilt range | 26° – 172° |
| Recovery | Mean recovery time | 1.52 s |
| Takeoff (10 trials) | Hover achieved | 100% |
| Takeoff | Mean time to hover | 2.06 s |
| Takeoff | Mean Z overshoot (pre-brake tune) | 0.289 m |
| Takeoff | Z overshoot after tune (seed 42) | ~0 m |
| Hybrid maze (perception) | Mission duration | 94.9 s |
| Hybrid maze | Final goal error | 0.40 m |
| Hybrid maze | Takeoff reasons | WALL, GAP, CLIMB |
| Hybrid maze | Success | Yes |
| Maze SLAM (oracle) | Time to goal | 105.6 s |
| Maze SLAM | Final goal error | 0.45 m |
| Monte Carlo (`--sensor-only`) | Success rate (30 trials) | 0% |

Figure 5.1 supports the scripted hybrid mission rows; Figures 5.3–5.6 document the hybrid maze; Figure 5.2 supports the SLAM maze rows.

## 5.12 Chapter discussion

Autonomous extensions demonstrated that rangefinder FSM logic and SLAM can coexist with the CAROLLINE mode stack without ROS. The hybrid maze is the strongest evidence that **perception-driven mode switching** scales beyond scripted waypoints: three structurally different obstacles each triggered the correct takeoff reason once.

Principal limitations observed were: SLAM pose smear when mapping used drifting estimates; replan spam near maze walls before cooldown tuning; fly-over altitude bugs in early exploration builds (later capped with `max_flight_height`); and zero Monte Carlo success under sensor-only estimation. Distinction-level reporting requires separating **oracle-validated** results from **SLAM-closed-loop** results—the latter remains an active development path.
