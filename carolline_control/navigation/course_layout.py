"""Course layout dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CAROLLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COURSE_CONFIG = CAROLLINE_ROOT / "navigation" / "course_config.yaml"


@dataclass
class WallSpec:
    """Vertical wall/obstacle blocking a checkpoint leg."""

    x: float
    y: float
    length: float = 0.25
    width: float = 3.0
    height: float = 0.55
    yaw_deg: float = 0.0
    leg_index: int = 0

    @property
    def top_z(self) -> float:
        return self.height

    @property
    def yaw_rad(self) -> float:
        return float(np.radians(self.yaw_deg))

    def face_normal_xy(self) -> np.ndarray:
        """Outward normal in the horizontal plane (along leg approach)."""
        yaw = self.yaw_rad
        return np.array([np.cos(yaw), np.sin(yaw)], dtype=float)

    def center_xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class CourseLayout:
    spawn_xy: np.ndarray
    checkpoints: list[np.ndarray]
    goal_xy: np.ndarray
    walls: list[WallSpec]
    fly_clearance: float = 0.90
    stop_distance: float = 0.65
    blocked_confirm_time: float = 0.30
    max_roll_height: float = 0.35
    patrol_arrival_radius: float = 0.45
    patrol_slowdown_radius: float = 1.2
    patrol_speed: float = 3.25
    goal_arrival_radius: float = 1.15
    terrain_half_x: float = 14.0
    terrain_half_y: float = 14.0
    terrain_fly_min_checkpoint: int = 3
    terrain_fly_after_recoveries: int = 2
    auto_fly_after_wall_leg: int = 2
    hover_height: float = 1.05
    pre_wall_offset: float = 0.55
    post_wall_offset: float = 0.65
    cage_radius: float = 0.40

    @property
    def all_targets(self) -> list[np.ndarray]:
        return [*self.checkpoints, self.goal_xy.copy()]

    def wall_for_leg(self, leg_index: int) -> WallSpec | None:
        for wall in self.walls:
            if wall.leg_index == leg_index:
                return wall
        return None

    def near_goal(self, xy: np.ndarray, tol: float | None = None) -> bool:
        tol = self.goal_arrival_radius if tol is None else tol
        return float(np.linalg.norm(np.asarray(xy[:2], dtype=float) - self.goal_xy)) < tol

    def on_final_approach(self, checkpoint_index: int) -> bool:
        return checkpoint_index >= len(self.checkpoints)

    def leg_bearing(self, leg_index: int) -> np.ndarray:
        """Unit vector from checkpoint leg_index toward leg_index+1."""
        if leg_index < 0 or leg_index >= len(self.checkpoints):
            start = self.spawn_xy
            end = self.checkpoints[0] if self.checkpoints else self.goal_xy
        elif leg_index >= len(self.checkpoints) - 1:
            start = self.checkpoints[-1]
            end = self.goal_xy
        else:
            start = self.checkpoints[leg_index]
            end = self.checkpoints[leg_index + 1]
        delta = np.asarray(end[:2], dtype=float) - np.asarray(start[:2], dtype=float)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return delta / norm

    def fly_over_waypoints(
        self,
        wall: WallSpec,
        terrain_z_at,
        *,
        leg_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pre, over, and post waypoints for a fly-over hop."""
        bearing = self.leg_bearing(leg_index)
        center = wall.center_xy()
        pre_xy = center - bearing * (0.5 * wall.length + self.pre_wall_offset)
        post_xy = center + bearing * (0.5 * wall.length + self.post_wall_offset)
        ground_z = max(
            float(terrain_z_at(pre_xy[0], pre_xy[1])),
            float(terrain_z_at(post_xy[0], post_xy[1])),
            float(terrain_z_at(center[0], center[1])),
        )
        over_z = ground_z + wall.height + self.fly_clearance
        hover_z = max(self.hover_height, over_z)
        pre = np.array([pre_xy[0], pre_xy[1], hover_z], dtype=float)
        over = np.array([center[0], center[1], over_z], dtype=float)
        post = np.array([post_xy[0], post_xy[1], hover_z], dtype=float)
        return pre, over, post

    def landing_xy_after_wall(self, wall: WallSpec, leg_index: int) -> np.ndarray:
        bearing = self.leg_bearing(leg_index)
        center = wall.center_xy()
        post_xy = center + bearing * (0.5 * wall.length + self.post_wall_offset)
        return post_xy

    def terrain_hop_waypoints(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        terrain_z_at,
        *,
        current_z: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lift, cruise, and approach waypoints to fly over unrollable terrain."""
        start = np.asarray(start_xy[:2], dtype=float)
        end = np.asarray(end_xy[:2], dtype=float)
        mid_xy = 0.5 * (start + end)
        samples = [start, mid_xy, end]
        max_ground = max(float(terrain_z_at(float(p[0]), float(p[1]))) for p in samples)
        cruise_z = max(
            max_ground + self.fly_clearance + self.cage_radius,
            self.hover_height,
        )
        z0 = float(terrain_z_at(float(start[0]), float(start[1])))
        lift_z = max(z0 + self.cage_radius + 0.55, self.hover_height * 0.85)
        if current_z is not None:
            lift_z = max(lift_z, float(current_z) + 0.70)
        lift = np.array([start[0], start[1], lift_z], dtype=float)
        cruise = np.array([mid_xy[0], mid_xy[1], cruise_z], dtype=float)
        approach = np.array([end[0], end[1], cruise_z], dtype=float)
        return lift, cruise, approach


def load_course_layout(
    path: Path | None = None,
    *,
    cage_radius: float = 0.40,
) -> tuple[CourseLayout, dict]:
    cfg_path = path or DEFAULT_COURSE_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    course = raw.get("course", {})
    walls = [
        WallSpec(
            x=float(w["x"]),
            y=float(w["y"]),
            length=float(w.get("length", 0.25)),
            width=float(w.get("width", 3.0)),
            height=float(w.get("height", 0.55)),
            yaw_deg=float(w.get("yaw_deg", w.get("yaw", 0.0))),
            leg_index=int(w.get("leg_index", idx)),
        )
        for idx, w in enumerate(course.get("walls", []))
    ]
    checkpoints = [np.array(pt[:2], dtype=float) for pt in course.get("checkpoints", [])]
    layout = CourseLayout(
        spawn_xy=np.array(course.get("spawn", [-10.0, 0.0])[:2], dtype=float),
        checkpoints=checkpoints,
        goal_xy=np.array(course.get("goal", [10.0, 0.0])[:2], dtype=float),
        walls=walls,
        fly_clearance=float(course.get("fly_clearance", 0.90)),
        stop_distance=float(course.get("stop_distance", 0.65)),
        blocked_confirm_time=float(course.get("blocked_confirm_time", 0.30)),
        max_roll_height=float(course.get("max_roll_height", 0.35)),
        patrol_arrival_radius=float(course.get("patrol_arrival_radius", 0.45)),
        patrol_slowdown_radius=float(course.get("patrol_slowdown_radius", 1.2)),
        patrol_speed=float(
            course.get("patrol_speed", raw.get("mobility", {}).get("patrol_speed", 3.25))
        ),
        goal_arrival_radius=float(course.get("goal_arrival_radius", 1.15)),
        terrain_half_x=float(course.get("terrain_half_x", raw.get("terrain", {}).get("half_x", 14.0))),
        terrain_half_y=float(course.get("terrain_half_y", raw.get("terrain", {}).get("half_y", 14.0))),
        terrain_fly_min_checkpoint=int(course.get("terrain_fly_min_checkpoint", 3)),
        terrain_fly_after_recoveries=int(course.get("terrain_fly_after_recoveries", 2)),
        auto_fly_after_wall_leg=int(course.get("auto_fly_after_wall_leg", 2)),
        hover_height=float(course.get("hover_height", 1.05)),
        pre_wall_offset=float(course.get("pre_wall_offset", 0.55)),
        post_wall_offset=float(course.get("post_wall_offset", 0.65)),
        cage_radius=cage_radius,
    )
    return layout, raw
