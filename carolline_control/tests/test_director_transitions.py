"""Unit tests for HybridCourseDirector FSM transitions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.navigation.course_layout import CourseLayout, load_course_layout
from carolline_control.navigation.director import CoursePhase, HybridCourseDirector
from carolline_control.navigation.perception import ObstacleScan
from carolline_control.navigation.rolling_follower import TerrainRollingFollower
from carolline_control.utils.types import ControlMode


def _scan(*, blocked: bool, flyable: bool = True) -> ObstacleScan:
    return ObstacleScan(
        forward_min_m=0.35 if blocked else 5.0,
        forward_argmin_angle=0.0,
        upward_clear_m=2.0,
        blocked=blocked,
        flyable=flyable,
        raw_ranges=np.array([0.35 if blocked else 5.0]),
    )


def _state(*, xy, vel=(0.0, 0.0), on_ground=True, z=0.5):
    state = MagicMock()
    state.position = np.array([xy[0], xy[1], z], dtype=float)
    state.velocity = np.array([vel[0], vel[1], 0.0], dtype=float)
    state.on_ground = on_ground
    state.rotation = np.eye(3)
    return state


def _build_director(layout: CourseLayout | None = None):
    layout = layout or load_course_layout()[0]
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    config.roll_target = layout.checkpoints[0].copy()
    config.landing_height = 0.5
    config.hover_height = 1.0
    config.spawn_xy = layout.spawn_xy.copy()
    config.upright_cos_threshold = 0.85
    config.cage_radius = layout.cage_radius

    controller = MagicMock()
    controller.config = config
    controller.mode_manager = MagicMock()
    controller.mode_manager.mode = ControlMode.ROLLING
    controller.planner = MagicMock()
    controller.planner.mission_complete = False
    controller.rolling = MagicMock()

    follower = TerrainRollingFollower(layout, config)
    director = HybridCourseDirector(controller, layout, follower, lambda x, y: 0.0)
    return director, controller, follower


def test_roll_to_blocked_when_wall_confirmed():
    layout, _ = load_course_layout()
    director, controller, follower = _build_director(layout)
    director.begin(0.0)
    follower.set_checkpoint_index(1)
    assert director.phase == CoursePhase.ROLL_TO_CHECKPOINT

    dt = 0.1
    t = 0.0
    for _ in range(5):
        t += dt
        running = director.update(
            _state(xy=(-4.3, 0.0), vel=(0.0, 0.0)),
            ControlMode.ROLLING,
            t,
            dt,
            _scan(blocked=True),
        )
        assert running

    assert director.phase == CoursePhase.BLOCKED_RECOVER
    controller.mode_manager.request_takeoff.assert_called()


def test_cleared_leg_does_not_stop_rolling_on_blocked_scan():
    layout, _ = load_course_layout()
    director, _, follower = _build_director(layout)
    director.begin(0.0)
    director._cleared_legs.add(2)
    follower.set_checkpoint_index(3)

    director.update(
        _state(xy=(6.5, 0.1), vel=(1.0, 0.0)),
        ControlMode.ROLLING,
        1.0,
        0.004,
        _scan(blocked=True),
    )
    assert not follower.respect_blocked_scan
    vel = follower.velocity_command_xy(_state(xy=(6.5, 0.1), vel=(1.0, 0.0)), _scan(blocked=True))
    assert float(np.linalg.norm(vel)) > 0.1


def test_done_when_final_goal_reached():
    layout, _ = load_course_layout()
    director, controller, follower = _build_director(layout)
    director.begin(0.0)
    follower.checkpoint_index = len(layout.checkpoints)
    follower._target_xy = layout.goal_xy.copy()

    running = director.update(
        _state(xy=layout.goal_xy, vel=(0.0, 0.0)),
        ControlMode.ROLLING,
        2.0,
        0.004,
        _scan(blocked=False),
    )
    assert not running
    assert director.phase == CoursePhase.DONE
    controller.mode_manager.request_idle.assert_called()


def test_near_goal_with_perimeter_blocked():
    from carolline_control.navigation.perception import filter_scan_for_mission

    layout, _ = load_course_layout()
    director, controller, follower = _build_director(layout)
    director.begin(0.0)
    follower.checkpoint_index = len(layout.checkpoints)
    follower._target_xy = layout.goal_xy.copy()
    overshoot = layout.goal_xy + np.array([1.0, 0.0])
    raw = _scan(blocked=True)
    filtered = filter_scan_for_mission(
        raw,
        _state(xy=overshoot),
        layout,
        checkpoint_index=follower.checkpoint_index,
        cleared_legs=director._cleared_legs,
    )
    assert filtered.blocked is False

    running = director.update(
        _state(xy=overshoot, vel=(0.0, 0.0)),
        ControlMode.ROLLING,
        2.0,
        0.004,
        raw,
    )
    assert not running
    assert director.phase == CoursePhase.DONE


def test_fly_over_waypoints_above_wall():
    layout, _ = load_course_layout()
    wall = layout.walls[0]

    def terrain_z_at(x, y):
        return 0.1

    pre, over, post = layout.fly_over_waypoints(wall, terrain_z_at, leg_index=wall.leg_index)
    assert over[2] > pre[2] - 0.01
    assert over[2] >= wall.height + layout.fly_clearance + 0.09


if __name__ == "__main__":
    test_roll_to_blocked_when_wall_confirmed()
    test_cleared_leg_does_not_stop_rolling_on_blocked_scan()
    test_done_when_final_goal_reached()
    test_near_goal_with_perimeter_blocked()
    test_fly_over_waypoints_above_wall()
    print("test_director_transitions: OK")
