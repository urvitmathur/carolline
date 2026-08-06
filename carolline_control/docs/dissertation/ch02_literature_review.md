# Chapter 2 — Literature Review

## 2.1 Introduction

This chapter surveys literature relevant to caged hybrid rolling–flying vehicles, the CAROLLINE architecture, geometric aerial control, actuation modelling, autonomous navigation, SLAM, and physics simulation. The review informs the design choices documented in Chapters 3–5.

## 2.2 Hybrid locomotion and CAROLLINE

Ground–air hybrid robots span designs from wheeled copters to platforms that use the rotor cage itself as the rolling contact surface. CAROLLINE (Lee et al., 2025) distinguishes itself by exploiting **bidirectional rotor thrust** to generate contact torques for omnidirectional rolling while retaining standard quadrotor flight after an explicit upright recovery phase. The paper defines a modular control architecture: pre-takeoff ground recovery, rolling velocity tracking in the contact frame, takeoff and hover, SE(3) waypoint flight, and landing.

The literature on hybrid systems emphasises **mode switching** and the need to avoid unsafe transitions—for example, entering aerial hover while the cage is still nominally a ground vehicle. CAROLLINE addresses this with discrete operating modes and timed settle gates before takeoff. The present project adopted this finite-state structure verbatim in software (`ModeManager`) and extended it with contact-loss detection during rolling.

## 2.3 Ground rolling control

Rolling inside a spherical cage is not equivalent to wheeled kinematics: the contact point moves on the cage surface, and thrust allocation must respect motor limits while producing a desired contact wrench. CAROLLINE’s ground controller uses a contact-frame velocity loop and a thrust allocation matrix that prioritises attitude torque while permitting near-null collective thrust—essential for rolling without lifting off unintentionally.

Related work on rolling spheres and ballbots (e.g. inverse kinematics on SO(3)) informed the expectation that **yaw authority on the ground is weak** relative to roll and pitch, because propeller drag torque about the vertical axis is small. The simulation encodes this through elevated yaw weights in the ground pseudoinverse and through `yaw_drag_coeff` in the motor mixer.

## 2.4 Geometric aerial control

Aerial tracking follows Lee, Leghorn, and McClamroch (2010), who formulate position and attitude control on SE(3) using thrust direction and a desired rotation matrix constructed from the commanded acceleration vector and a fixed yaw. CAROLLINE uses this **geometric tracking** approach rather than a linearised PID cascade, which improves behaviour under large tilts during takeoff and aggressive waypoint changes.

Alternative linear-quadratic regulators (LQR) and model-predictive controllers (MPC) were implemented in the repository for comparative studies (`compare_controllers.py`, `lqr_config.yaml`). Literature on LQR for quadrotors typically linearises about hover; geometric control was retained as the baseline because it matches the reference paper and handles the hybrid transition region more robustly in simulation.

## 2.5 Actuation: bidirectional ESCs and thrust–torque coupling

Commercial multirotors normally use unidirectional ESCs. CAROLLINE’s BLHeli bidirectional firmware maps PWM signals in microseconds (centre ≈ 1500 µs) to positive and negative thrust. The paper characterises this relationship with **polynomial fits** from bench tests (Figure 5 in Lee et al., 2025): thrust versus ESC signal, and reaction torque versus thrust through a coefficient μ.

The simulation implements a **piecewise-linear ESC abstraction** (`EscThrustMapper`) with forward and reverse saturation (13 N / 10 N) and reverse-efficiency compensation. Yaw torque from each motor is modelled as proportional to thrust via gear components in the MuJoCo actuator definition. This is sufficient for controller tuning but understates nonlinearities near the mid-band dead zone.

## 2.6 Autonomous navigation without ROS

Classical mobile-robot navigation stacks (ROS `move_base`, costmaps, AMCL) separate localisation, mapping, and planning. Recent research explores **lightweight FSM directors** for hybrid platforms where flight is triggered only when rolling is infeasible. The course-navigation subsystem in this project follows that pattern: a director reads rangefinder scans, filters false positives near mission phase boundaries, and requests mode changes (`request_takeoff`, `request_flight`) rather than commanding motor thrust directly.

Perception uses MuJoCo rangefinder sensors arranged in a forward fan on the cage—analogous to a 2D lidar slice at a fixed height. Fly-over waypoints are generated from wall geometry (`WallSpec`, `fly_over_waypoints`). This mirrors literature on **reactive obstacle avoidance** combined with global checkpoint sequences, but does not implement full kinodynamic planning in 3D.

## 2.7 SLAM for ground robots

Occupancy grid mapping with **log-odds updates** along ray casts is standard (Moravec and Elfes, 1985; Thrun, Burgard, and Fox, 2005). Scan matching aligns current range readings to the map to correct odometry drift—a simplified alternative to particle-filter SLAM for 2D ground platforms.

The rolling SLAM extension in this project fused:

- **Odometry** from contact-aware integration and a planar pose EKF;
- **Mapping** from 24 forward rangefinders;
- **Localisation** via grid search over (x, y, yaw);
- **Planning** with A* on an inflated occupancy grid.

Literature on rolling SLAM often assumes wheeled odometry; cage rolling introduces slip and attitude coupling that degrades naive dead reckoning. The project therefore supports oracle mapping pose during development and `--slam-nav` for integrated testing.

## 2.8 Physics simulation with MuJoCo

MuJoCo provides contact-rich rigid-body simulation suitable for rolling and flight in one model. The Skydio X2 Menagerie asset supplies inertial parameters and rotor geometry; the cage is added as an additional collision sphere and visual mesh. Simulation-based controller development is established practice when hardware access is limited, but **sim-to-real gaps** remain in contact friction, ESC curves, and sensor noise—acknowledged in Chapter 6.

## 2.9 Chapter discussion

The literature confirms that CAROLLINE occupies a distinct niche: cage rolling with bidirectional thrust and explicit recovery before flight. Gaps identified for this project were: (1) a reproducible open simulation stack with mission autonomy; (2) SLAM-based maze exploration without ROS; and (3) validation metrics comparable to paper claims (recovery success, energy cost of transport). Chapters 3–5 describe how these gaps were addressed in software and what quantitative results were obtained.
