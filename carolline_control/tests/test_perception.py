"""Unit tests for rangefinder obstacle perception."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.config_loader import load_config
from carolline_control.navigation.course_layout import load_course_layout
from carolline_control.navigation.course_scene import compile_course_scene
from carolline_control.navigation.perception import ObstacleScan, RangePerception
from carolline_control.sim_estimator import build_estimator


def test_obstacle_scan_blocked_and_flyable_flags():
    close = ObstacleScan(
        forward_min_m=0.40,
        forward_argmin_angle=0.0,
        upward_clear_m=2.0,
        blocked=True,
        flyable=True,
        raw_ranges=np.array([0.40, 2.0]),
    )
    assert close.blocked
    assert close.flyable

    far = ObstacleScan(
        forward_min_m=3.0,
        forward_argmin_angle=0.0,
        upward_clear_m=2.0,
        blocked=False,
        flyable=False,
        raw_ranges=np.array([3.0, 2.0]),
    )
    assert not far.blocked
    assert not far.flyable


def test_range_perception_scan_aggregates_sensor_readings():
    layout, raw = load_course_layout()
    config = load_config(str(REPO_ROOT / "carolline_control" / "config.yaml"))
    model_path = REPO_ROOT / raw.get("model_path", config.model_path)
    model, _, _, _ = compile_course_scene(model_path, layout, raw)
    data = mujoco.MjData(model)
    estimator = build_estimator(model, config)
    perception = RangePerception(
        model,
        stop_distance=layout.stop_distance,
        fly_clearance=layout.fly_clearance,
    )
    assert len(perception._sensor_ids) >= 7

    for sid, is_up in zip(perception._sensor_ids, perception._sensor_is_up):
        adr = int(model.sensor_adr[sid])
        data.sensordata[adr] = 2.0 if is_up else 0.35

    blocked_scan = perception.scan(data, estimator.estimate(data))
    assert blocked_scan.blocked
    assert blocked_scan.forward_min_m < layout.stop_distance
    assert blocked_scan.flyable

    for sid, is_up in zip(perception._sensor_ids, perception._sensor_is_up):
        adr = int(model.sensor_adr[sid])
        data.sensordata[adr] = 5.0

    open_scan = perception.scan(data, estimator.estimate(data))
    assert not open_scan.blocked


if __name__ == "__main__":
    test_obstacle_scan_blocked_and_flyable_flags()
    test_range_perception_scan_aggregates_sensor_readings()
    print("test_perception: OK")
