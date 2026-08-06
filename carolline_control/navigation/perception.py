"""Rangefinder-based obstacle perception for autonomous navigation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from carolline_control.navigation.course_scene import RANGE_SENSOR_PREFIX
from carolline_control.navigation.exploration_scene import EXPLORATION_SENSOR_PREFIX
from carolline_control.utils.types import RobotState


@dataclass
class ObstacleScan:
    forward_min_m: float
    forward_argmin_angle: float
    upward_clear_m: float
    blocked: bool
    flyable: bool
    raw_ranges: np.ndarray
    down_min_m: float = 8.0
    wall_block: bool = False
    floor_gap: bool = False
    climb_face: bool = False


class RangePerception:
    """Read MuJoCo rangefinder sensors mounted on the cage."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        stop_distance: float = 0.65,
        max_roll_height: float = 0.35,
        fly_clearance: float = 0.90,
        climb_clearance: float = 1.80,
        gap_down_threshold: float = 1.50,
        max_range: float = 8.0,
        forward_cone_deg: float = 60.0,
        min_valid_range: float = 0.18,
        sensor_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.stop_distance = float(stop_distance)
        self.max_roll_height = float(max_roll_height)
        self.fly_clearance = float(fly_clearance)
        self.climb_clearance = float(climb_clearance)
        self.gap_down_threshold = float(gap_down_threshold)
        self.max_range = float(max_range)
        self.forward_cone_deg = float(forward_cone_deg)
        self.min_valid_range = float(min_valid_range)
        self._sensor_ids: list[int] = []
        self._site_ids: list[int] = []
        self._sensor_is_up: list[bool] = []
        self._sensor_is_down: list[bool] = []
        self._sensor_angles_deg: list[float] = []
        prefixes = sensor_prefixes or (RANGE_SENSOR_PREFIX, EXPLORATION_SENSOR_PREFIX)
        entries: list[tuple[int, int, bool, bool, str]] = []
        for sid in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if not name or not any(name.startswith(prefix) for prefix in prefixes):
                continue
            is_up = name.endswith("_up")
            is_down = name.endswith("_down")
            entries.append((sid, int(model.sensor_objid[sid]), is_up, is_down, name))

        forward_count = sum(1 for _, _, is_up, is_down, _ in entries if not is_up and not is_down)
        forward_idx = 0
        for sid, site_id, is_up, is_down, name in entries:
            self._sensor_ids.append(sid)
            self._site_ids.append(site_id)
            self._sensor_is_up.append(is_up)
            self._sensor_is_down.append(is_down)
            if is_up or is_down:
                self._sensor_angles_deg.append(0.0)
            else:
                if forward_count > 7:
                    span = 150.0
                    angle = -span * 0.5 + span * (forward_idx / max(forward_count - 1, 1))
                else:
                    fan_angles = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
                    angle = fan_angles[forward_idx] if forward_idx < len(fan_angles) else 0.0
                self._sensor_angles_deg.append(angle)
                forward_idx += 1

    def _read_range(self, data: mujoco.MjData, sensor_id: int) -> float:
        adr = int(self.model.sensor_adr[sensor_id])
        value = float(data.sensordata[adr])
        if value < 0.0 or not np.isfinite(value):
            return self.max_range
        if value < self.min_valid_range:
            return self.max_range
        return min(value, self.max_range)

    def iter_ray_readings(self, data: mujoco.MjData):
        """Yield (site_id, is_upward, hit_distance) for each rangefinder."""
        for sid, site_id, is_up, is_down in zip(
            self._sensor_ids, self._site_ids, self._sensor_is_up, self._sensor_is_down
        ):
            yield site_id, is_up or is_down, self._read_range(data, sid)

    def scan(self, data: mujoco.MjData, state: RobotState) -> ObstacleScan:
        cone_pairs: list[tuple[float, float]] = []
        upward_clear = self.max_range
        down_min = self.max_range
        raw: list[float] = []

        for sid, is_up, is_down, angle_deg in zip(
            self._sensor_ids,
            self._sensor_is_up,
            self._sensor_is_down,
            self._sensor_angles_deg,
        ):
            dist = self._read_range(data, sid)
            raw.append(dist)
            if is_up:
                upward_clear = dist
            elif is_down:
                down_min = dist
            elif abs(angle_deg) <= self.forward_cone_deg:
                cone_pairs.append((dist, angle_deg))

        if not cone_pairs:
            return ObstacleScan(
                forward_min_m=self.max_range,
                forward_argmin_angle=0.0,
                upward_clear_m=upward_clear,
                blocked=False,
                flyable=False,
                down_min_m=down_min,
                raw_ranges=np.asarray(raw, dtype=float),
            )

        argmin_idx = int(np.argmin([pair[0] for pair in cone_pairs]))
        forward_min = float(cone_pairs[argmin_idx][0])
        argmin_angle = float(cone_pairs[argmin_idx][1])
        blocked = forward_min < self.stop_distance
        flyable = blocked and upward_clear > self.fly_clearance and forward_min > 0.05
        # Hop vs climb is resolved geographically in filter_scan_for_hybrid_maze.
        wall_block = flyable
        climb_face = flyable
        low_altitude = bool(state.on_ground) or float(state.position[2]) <= self.max_roll_height + 0.15
        floor_gap = low_altitude and down_min >= self.gap_down_threshold
        return ObstacleScan(
            forward_min_m=forward_min,
            forward_argmin_angle=argmin_angle,
            upward_clear_m=upward_clear,
            blocked=blocked,
            flyable=flyable,
            down_min_m=down_min,
            wall_block=wall_block,
            floor_gap=floor_gap,
            climb_face=climb_face,
            raw_ranges=np.asarray(raw, dtype=float),
        )

    def local_avoidance_xy(
        self,
        data: mujoco.MjData,
        *,
        clear_distance: float = 1.05,
        gain: float = 1.8,
        max_repulsion: float = 1.6,
    ) -> np.ndarray:
        """Repulsive XY velocity from nearby rangefinder hits (keep clear of walls)."""
        force = np.zeros(2, dtype=float)
        weight_sum = 0.0
        clear = float(clear_distance)
        for site_id, is_up, dist in self.iter_ray_readings(data):
            if is_up or dist >= clear:
                continue
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            # MuJoCo rangefinder looks along -site Z.
            ray_xy = -site_mat[:, 2][:2]
            norm = float(np.linalg.norm(ray_xy))
            if norm < 1e-6:
                continue
            ray_xy /= norm
            # Push away from the hit (opposite the ray).
            closeness = (clear - float(dist)) / clear
            w = closeness * closeness
            force -= ray_xy * w
            weight_sum += w

        if weight_sum < 1e-6:
            return np.zeros(2, dtype=float)
        force = (force / weight_sum) * float(gain)
        mag = float(np.linalg.norm(force))
        if mag > max_repulsion:
            force *= max_repulsion / mag
        return force


def _copy_scan(
    scan: ObstacleScan,
    *,
    blocked: bool | None = None,
    flyable: bool | None = None,
    wall_block: bool | None = None,
    floor_gap: bool | None = None,
    climb_face: bool | None = None,
) -> ObstacleScan:
    return ObstacleScan(
        forward_min_m=scan.forward_min_m,
        forward_argmin_angle=scan.forward_argmin_angle,
        upward_clear_m=scan.upward_clear_m,
        blocked=scan.blocked if blocked is None else blocked,
        flyable=scan.flyable if flyable is None else flyable,
        raw_ranges=scan.raw_ranges,
        down_min_m=scan.down_min_m,
        wall_block=scan.wall_block if wall_block is None else wall_block,
        floor_gap=scan.floor_gap if floor_gap is None else floor_gap,
        climb_face=scan.climb_face if climb_face is None else climb_face,
    )


def filter_scan_for_mission(
    scan: ObstacleScan,
    state,
    layout: "CourseLayout",
    *,
    checkpoint_index: int,
    cleared_legs: set[int],
) -> ObstacleScan:
    """Drop perimeter-only hits; keep internal wall detections for fly-over."""
    leg = checkpoint_index
    wall = layout.wall_for_leg(leg)
    has_internal_wall = wall is not None and leg not in cleared_legs

    blocked = scan.blocked and has_internal_wall
    flyable = scan.flyable and has_internal_wall
    wall_block = scan.wall_block and has_internal_wall
    climb_face = scan.climb_face and has_internal_wall

    if layout.on_final_approach(checkpoint_index):
        blocked = False
        flyable = False
        wall_block = False
        climb_face = False
    elif not has_internal_wall and scan.blocked:
        px, py = float(state.position[0]), float(state.position[1])
        near_boundary = (
            abs(px) > layout.terrain_half_x - 1.6
            or abs(py) > layout.terrain_half_y - 1.6
        )
        if near_boundary:
            blocked = False
            flyable = False
            wall_block = False
            climb_face = False

    if (
        blocked == scan.blocked
        and flyable == scan.flyable
        and wall_block == scan.wall_block
        and climb_face == scan.climb_face
    ):
        return scan
    return _copy_scan(
        scan,
        blocked=blocked,
        flyable=flyable,
        wall_block=wall_block,
        climb_face=climb_face,
        floor_gap=False,
    )


def filter_scan_for_exploration(
    scan: ObstacleScan,
    state,
    *,
    arena_half_x: float,
    arena_half_y: float,
    boundary_margin: float = 1.4,
) -> ObstacleScan:
    """Ignore perimeter-only hits so arena walls do not stop rolling."""
    px, py = float(state.position[0]), float(state.position[1])
    near_boundary = (
        abs(px) > arena_half_x - boundary_margin
        or abs(py) > arena_half_y - boundary_margin
    )
    if not near_boundary or not scan.blocked:
        return scan
    return _copy_scan(scan, blocked=False, flyable=False, wall_block=False, climb_face=False)


def filter_scan_for_hybrid_maze(
    scan: ObstacleScan,
    state,
    layout: "HybridMazeLayout",
    *,
    cleared: set[str],
    on_platform: bool,
) -> ObstacleScan:
    """Suppress arena perimeter hits and already-cleared obstacle classes."""
    px, py = float(state.position[0]), float(state.position[1])
    margin = 1.2
    near_boundary = (
        abs(px) > layout.arena_half_x - margin
        or abs(py) > layout.arena_half_y - margin
    )

    gap = layout.gap
    near_gap = layout.near_gap_xy(np.array([px, py]))
    near_low_wall = layout.near_low_wall_xy(np.array([px, py]))
    near_climb = layout.near_climb_xy(np.array([px, py]))

    wall_block = (
        scan.wall_block
        and "WALL" not in cleared
        and not near_boundary
        and near_low_wall
    )
    floor_gap = (
        scan.floor_gap
        and "GAP" not in cleared
        and not on_platform
        and near_gap
    )
    climb_face = (
        scan.climb_face
        and "CLIMB" not in cleared
        and not near_boundary
        and near_climb
    )
    blocked = scan.blocked and not near_boundary
    flyable = scan.flyable and not near_boundary
    if (
        wall_block == scan.wall_block
        and floor_gap == scan.floor_gap
        and climb_face == scan.climb_face
        and blocked == scan.blocked
        and flyable == scan.flyable
    ):
        return scan
    return _copy_scan(
        scan,
        blocked=blocked,
        flyable=flyable,
        wall_block=wall_block,
        floor_gap=floor_gap,
        climb_face=climb_face,
    )
