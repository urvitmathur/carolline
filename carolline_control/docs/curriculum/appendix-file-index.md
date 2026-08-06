# Appendix — Python File Index

**Previous:** [16-experiments-and-studies.md](16-experiments-and-studies.md) | **Back to:** [00-README.md](00-README.md)

Alphabetical index of every Python file. **Chapter** links to the curriculum chapter with the deepest coverage.

| File | ~Lines | Ch | Summary |
|------|--------|-----|---------|
| carolline_control/__init__.py | 3 | [01](01-project-overview.md) | Package version string. |
| carolline_control/carolline_controller.py | 576 | [06](06-control-stack-overview.md) | Top-level integrator: compute() runs mode manager, planners, controllers, mixer. |
| carolline_control/config_loader.py | 153 | [02](02-setup-and-config.md) | Loads config.yaml into ControllerConfig; builds default waypoints. |
| carolline_control/controllers/__init__.py | 23 | [01](01-project-overview.md) | Package version string. |
| carolline_control/controllers/attitude_controller.py | 43 | — | Project support module. |
| carolline_control/controllers/esc_mapper.py | 75 | — | Project support module. |
| carolline_control/controllers/flight_controller.py | 254 | — | Project support module. |
| carolline_control/controllers/ground_dynamics.py | 259 | — | Project support module. |
| carolline_control/controllers/lqr/__init__.py | 14 | [01](01-project-overview.md) | Package version string. |
| carolline_control/controllers/lqr/comparison_metrics.py | 160 | — | Project support module. |
| carolline_control/controllers/lqr/config_loader.py | 43 | [02](02-setup-and-config.md) | Loads config.yaml into ControllerConfig; builds default waypoints. |
| carolline_control/controllers/lqr/dare_solver.py | 45 | — | Project support module. |
| carolline_control/controllers/lqr/flight_lqr.py | 153 | — | Project support module. |
| carolline_control/controllers/lqr/rolling_lqr.py | 135 | — | Project support module. |
| carolline_control/controllers/mode_manager.py | 201 | — | Project support module. |
| carolline_control/controllers/motor_mixer.py | 70 | — | Project support module. |
| carolline_control/controllers/mpc/__init__.py | 11 | [01](01-project-overview.md) | Package version string. |
| carolline_control/controllers/mpc/config_loader.py | 48 | [02](02-setup-and-config.md) | Loads config.yaml into ControllerConfig; builds default waypoints. |
| carolline_control/controllers/mpc/finite_horizon.py | 105 | — | Project support module. |
| carolline_control/controllers/mpc/flight_mpc.py | 137 | — | Project support module. |
| carolline_control/controllers/mpc/rolling_mpc.py | 149 | — | Project support module. |
| carolline_control/controllers/orientation_generator.py | 91 | — | Project support module. |
| carolline_control/controllers/planner.py | 195 | — | Project support module. |
| carolline_control/controllers/pre_takeoff_controller.py | 72 | — | Project support module. |
| carolline_control/controllers/rolling_controller.py | 85 | — | Project support module. |
| carolline_control/controllers/state_estimator.py | 347 | — | Project support module. |
| carolline_control/controllers/torque_controller.py | 44 | — | Project support module. |
| carolline_control/logging/__init__.py | 0 | [01](01-project-overview.md) | Package version string. |
| carolline_control/logging/logger.py | 189 | — | Project support module. |
| carolline_control/logging/paths.py | 20 | — | Project support module. |
| carolline_control/logging/pipeline_tracer.py | 364 | — | Project support module. |
| carolline_control/main.py | 213 | [06](06-control-stack-overview.md) | Primary simulation entry: load model, run control loop, log and plot. |
| carolline_control/navigation/__init__.py | 17 | [01](01-project-overview.md) | Package version string. |
| carolline_control/navigation/course_analysis.py | 129 | — | Project support module. |
| carolline_control/navigation/course_layout.py | 208 | — | Project support module. |
| carolline_control/navigation/course_logger.py | 108 | — | Project support module. |
| carolline_control/navigation/course_scene.py | 245 | — | Project support module. |
| carolline_control/navigation/director.py | 520 | — | Project support module. |
| carolline_control/navigation/perception.py | 159 | — | Project support module. |
| carolline_control/navigation/rolling_follower.py | 110 | — | Project support module. |
| carolline_control/navigation/sensor_viz.py | 78 | — | Project support module. |
| carolline_control/navigation/sim_recorder.py | 89 | — | Project support module. |
| carolline_control/plots/__init__.py | 1 | [01](01-project-overview.md) | Package version string. |
| carolline_control/plots/trajectory_tracking.py | 128 | — | Project support module. |
| carolline_control/rolling_ramp/__init__.py | 15 | [01](01-project-overview.md) | Package version string. |
| carolline_control/rolling_ramp/controller.py | 41 | — | Project support module. |
| carolline_control/rolling_ramp/path.py | 245 | — | Project support module. |
| carolline_control/rolling_ramp/plots.py | 134 | — | Project support module. |
| carolline_control/rolling_ramp/scene.py | 176 | — | Project support module. |
| carolline_control/scripts/analyze_hover.py | 28 | — | Project support module. |
| carolline_control/scripts/analyze_log.py | 44 | — | Project support module. |
| carolline_control/scripts/compare_controllers.py | 48 | — | Project support module. |
| carolline_control/scripts/compare_flight_controllers.py | 240 | — | Project support module. |
| carolline_control/scripts/compare_rolling_controllers.py | 197 | — | Project support module. |
| carolline_control/scripts/debug_mission.py | 97 | — | Project support module. |
| carolline_control/scripts/debug_rolling.py | 46 | — | Project support module. |
| carolline_control/scripts/manual_teleop.py | 646 | — | Project support module. |
| carolline_control/scripts/quantitative_analysis.py | 269 | — | Project support module. |
| carolline_control/scripts/rc_motor_teleop.py | 394 | — | Project support module. |
| carolline_control/scripts/rolling_ramp_analysis.py | 223 | — | Project support module. |
| carolline_control/scripts/run_controller_study.py | 321 | — | Project support module. |
| carolline_control/scripts/run_lqr_study.py | 299 | — | Project support module. |
| carolline_control/scripts/simulate_30deg_ramp_hold.py | 277 | — | Project support module. |
| carolline_control/scripts/simulate_autonomous_course.py | 347 | — | Project support module. |
| carolline_control/scripts/simulate_hybrid_locomotion.py | 608 | — | Project support module. |
| carolline_control/scripts/simulate_ramp_platform.py | 837 | — | Project support module. |
| carolline_control/scripts/simulate_terrain_mobility.py | 382 | — | Project support module. |
| carolline_control/scripts/system_analysis.py | 100 | — | Project support module. |
| carolline_control/scripts/validate_energy_cot.py | 322 | — | Project support module. |
| carolline_control/scripts/validate_monte_carlo.py | 508 | — | Project support module. |
| carolline_control/scripts/validate_recovery.py | 378 | — | Project support module. |
| carolline_control/scripts/validate_takeoff.py | 431 | — | Project support module. |
| carolline_control/scripts/validate_upright_stabilization.py | 421 | — | Project support module. |
| carolline_control/sim_estimator.py | 51 | [05](05-state-estimation.md) | Builds StateEstimator; adds --sensor-only CLI to scripts. |
| carolline_control/terrain/__init__.py | 13 | [01](01-project-overview.md) | Package version string. |
| carolline_control/terrain/heightmap.py | 144 | — | Project support module. |
| carolline_control/terrain/scene.py | 153 | — | Project support module. |
| carolline_control/tests/test_director_transitions.py | 176 | — | Project support module. |
| carolline_control/tests/test_mpc_gains.py | 92 | — | Project support module. |
| carolline_control/tests/test_perception.py | 80 | — | Project support module. |
| carolline_control/utils/__init__.py | 43 | [01](01-project-overview.md) | Package version string. |
| carolline_control/utils/so3.py | 221 | — | Project support module. |
| carolline_control/utils/types.py | 176 | — | Project support module. |
| carolline_control/validation/__init__.py | 12 | [01](01-project-overview.md) | Package version string. |
| carolline_control/validation/__main__.py | 4 | — | Project support module. |
| carolline_control/validation/campaigns.py | 118 | — | Project support module. |
| carolline_control/validation/cli.py | 76 | — | Project support module. |
| carolline_control/validation/config.py | 144 | — | Project support module. |
| carolline_control/validation/disturbances.py | 45 | — | Project support module. |
| carolline_control/validation/failures.py | 73 | — | Project support module. |
| carolline_control/validation/metrics.py | 211 | — | Project support module. |
| carolline_control/validation/model_perturbation.py | 49 | — | Project support module. |
| carolline_control/validation/monte_carlo.py | 62 | — | Project support module. |
| carolline_control/validation/observer.py | 182 | — | Project support module. |
| carolline_control/validation/randomization.py | 162 | — | Project support module. |
| carolline_control/validation/report.py | 116 | — | Project support module. |
| carolline_control/validation/runner.py | 201 | — | Project support module. |
| carolline_control/validation/sensitivity.py | 75 | — | Project support module. |
| carolline_control/validation/statistics.py | 211 | — | Project support module. |
| carolline_control/validation/storage.py | 104 | — | Project support module. |
| carolline_control/validation/types.py | 15 | — | Project support module. |
| carolline_control/validation/wrappers.py | 88 | — | Project support module. |
| carolline_control/visualization/__init__.py | 1 | [01](01-project-overview.md) | Package version string. |
| carolline_control/visualization/markers.py | 85 | — | Project support module. |
| tests/test_esc_mapper.py | 70 | — | Project support module. |
| tests/test_ground_dynamics.py | 240 | — | Project support module. |
| tests/test_lqr_gains.py | 78 | — | Project support module. |
| tests/test_mpc_gains.py | 76 | — | Project support module. |
| tests/test_state_estimator.py | 90 | — | Project support module. |
