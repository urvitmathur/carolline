# Chapter 1 — Introduction

## 1.1 Background and motivation

Mobile robots operating in unstructured environments must often traverse terrain that is alternately traversable on wheels and obstructed by steps, ditches, or vertical barriers. Fixed-mode platforms—pure ground robots or pure aerial vehicles—trade efficiency against versatility. Hybrid systems that combine rolling and flying locomotion address this trade-off by using ground contact where possible and flight only where necessary.

CAROLLINE (Lee et al., 2025) is a caged quadcopter whose spherical protective cage enables omnidirectional rolling on flat and inclined surfaces using differential rotor thrust, while retaining full multirotor flight after an upright recovery sequence. The cage provides collision protection and a stable contact geometry for rolling, but introduces non-standard actuation: bidirectional electronic speed controllers (ESCs) allow negative thrust for ground torque generation without flipping the vehicle.

Prior published work focused on controller design and hardware demonstration. The present project asked whether the full hybrid behaviour—rolling, recovery, flight, landing, and autonomous mission execution—could be reproduced and extended in simulation, and whether rangefinder-based mapping and planning could enable rolling exploration of unknown environments without a ROS navigation stack.

## 1.2 Project aims

The project pursued three aims:

1. **Reproduce CAROLLINE hybrid locomotion in MuJoCo** — Implement rolling, pre-takeoff recovery, takeoff, hover, waypoint flight, and landing using the geometric control framework cited in the CAROLLINE paper, with bidirectional motor modelling and mode-managed transitions.

2. **Implement mission-level autonomy** — Build a finite-state mission director that rolls through checkpoints, detects walls with cage-mounted rangefinders, executes fly-over manoeuvres, and handles terrain hops on heightfield scenes.

3. **Implement rolling-only SLAM navigation** — Construct a lightweight Python SLAM stack (occupancy mapping, scan matching, A* planning) and validate it in a Gazebo-style orthogonal maze with rolling-only locomotion.

4. **Demonstrate perception-driven hybrid maze navigation** — Build an obstacle-driven roll/fly course in which forward, upward, and downward rangefinders classify wall blocks, floor gaps, and climb faces, triggering distinct takeoff reasons (WALL, GAP, CLIMB) without waypoint scripting of flight segments.

## 1.3 Objectives

The following measurable objectives were defined:

| ID | Objective | Verification |
|----|-----------|--------------|
| O1 | Rolling reaches a designated ground target and triggers recovery | `main.py`, mode log |
| O2 | Recovery from arbitrary tilt succeeds reliably | `validate_recovery.py` |
| O3 | Takeoff reaches hover height with acceptable overshoot | `validate_takeoff.py`, trajectory plot |
| O4 | Cardinal waypoint flight and landing complete | `main.py --seed 42` |
| O5 | Autonomous course navigates walls and terrain | `simulate_autonomous_course.py` |
| O6 | Maze mission maps walls and reaches goal while rolling | `simulate_slam_maze.py` |
| O7 | Hybrid maze completes via perception-triggered takeoffs | `simulate_hybrid_maze.py` |

## 1.4 Contributions

The repository deliverables comprised approximately 103 Python modules organised as:

- **Control stack** — `CarollineController` integrating mode manager, rolling controller, pre-takeoff recovery, geometric flight controller, attitude/torque loop, motor mixer, and ESC mapper.
- **Navigation** — Course director, exploration director, SLAM stack, maze scene compiler, and waypoint followers.
- **Validation** — Recovery, takeoff, Monte Carlo, and energy scripts with CSV logging and trajectory plots.
- **Documentation** — Curriculum notes and this dissertation package.

Original extensions beyond paper reproduction include the autonomous terrain course, exploration SLAM in random obstacle fields, the orthogonal maze rolling-SLAM benchmark, and the obstacle-driven hybrid maze with WALL/GAP/CLIMB takeoff classification.

## 1.5 Scope and limitations

All results reported herein were obtained in **simulation** using MuJoCo 3.x and the Skydio X2 model from MuJoCo Menagerie with a spherical cage collision envelope. No physical robot was built or flight-tested. State estimation was primarily **oracle** (ground-truth pose from `qpos`) unless `--sensor-only` or `--slam-nav` modes were explicitly enabled. ESC characterisation used a piecewise-linear model rather than experimentally fitted polynomials.

## 1.6 Report structure

Chapter 2 reviews hybrid locomotion, geometric flight control, actuation, navigation, SLAM, and simulation literature. Chapter 3 describes the MuJoCo plant model, scenes, and software architecture. Chapter 4 presents the control system design including mode logic and actuation pipeline. Chapter 5 covers autonomous navigation, SLAM, and experimental results. Chapter 6 summarises achievements, discusses limitations, and outlines future work.
