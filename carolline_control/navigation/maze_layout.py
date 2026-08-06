"""Fixed maze layouts for rolling SLAM (Gazebo-style orthogonal walls)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CAROLLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAZE_CONFIG = CAROLLINE_ROOT / "navigation" / "maze_config.yaml"


@dataclass
class MazeSegment:
    """Axis-aligned wall segment (center + half extents in X/Y)."""

    x: float
    y: float
    half_x: float
    half_y: float

    @property
    def half_size(self) -> list[float]:
        return [self.half_x, self.half_y, 0.0]


@dataclass
class MazeLayout:
    arena_half: float
    wall_height: float
    wall_thickness: float
    spawn_xy: np.ndarray
    goal_xy: np.ndarray
    segments: list[MazeSegment]
    entrance_side: str = "north"
    entrance_width: float = 1.4
    entrance_offset: float = 0.0

    @property
    def arena_half_x(self) -> float:
        return self.arena_half

    @property
    def arena_half_y(self) -> float:
        return self.arena_half

    def point_clear(
        self,
        xy: np.ndarray,
        *,
        clearance: float = 0.55,
    ) -> bool:
        """True if xy is inside the arena and away from wall boxes."""
        x, y = float(xy[0]), float(xy[1])
        margin = clearance + 0.5 * self.wall_thickness
        if abs(x) > self.arena_half - margin or abs(y) > self.arena_half - margin:
            return False
        for seg in self.segments:
            dx = abs(x - seg.x) - (seg.half_x + clearance)
            dy = abs(y - seg.y) - (seg.half_y + clearance)
            if dx < 0.0 and dy < 0.0:
                return False
        return True

    def sample_free_spawn(
        self,
        rng: np.random.Generator,
        *,
        clearance: float = 0.55,
        goal_xy: np.ndarray | None = None,
        min_goal_dist: float = 2.5,
        max_tries: int = 400,
    ) -> np.ndarray:
        """Sample a random free XY pose inside the maze."""
        goal = self.goal_xy if goal_xy is None else np.asarray(goal_xy[:2], dtype=float)
        half = self.arena_half - clearance - 0.2
        for _ in range(max_tries):
            xy = np.array([rng.uniform(-half, half), rng.uniform(-half, half)], dtype=float)
            if float(np.linalg.norm(xy - goal)) < min_goal_dist:
                continue
            if self.point_clear(xy, clearance=clearance):
                return xy
        # Fallback: keep configured spawn if sampling fails.
        if self.point_clear(self.spawn_xy, clearance=clearance * 0.7):
            return self.spawn_xy.copy()
        return np.array([-self.arena_half * 0.55, self.arena_half * 0.55], dtype=float)


def _default_gazebo_maze() -> MazeLayout:
    """Approximate the orthogonal wall pattern from a classic Gazebo maze arena."""
    t = 0.08
    return MazeLayout(
        arena_half=4.5,
        wall_height=0.42,
        wall_thickness=t,
        spawn_xy=np.array([-2.8, 3.0], dtype=float),
        goal_xy=np.array([3.2, -3.2], dtype=float),
        entrance_side="north",
        entrance_width=1.5,
        entrance_offset=-0.8,
        segments=[
            MazeSegment(-1.2, 2.0, 1.6, t),
            MazeSegment(1.8, 2.2, 1.0, t),
            MazeSegment(-0.2, 0.9, t, 1.3),
            MazeSegment(1.2, 0.2, 1.4, t),
            MazeSegment(-2.0, -0.3, t, 1.1),
            MazeSegment(-0.8, -1.2, 1.8, t),
            MazeSegment(1.6, -1.0, t, 1.4),
            MazeSegment(2.4, -2.6, 1.1, t),
            MazeSegment(-2.5, -2.8, 0.9, t),
        ],
    )


def load_maze_layout(path: Path | None = None) -> tuple[MazeLayout, dict]:
    cfg_path = path or DEFAULT_MAZE_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    maze_raw = raw.get("maze", {})
    if not maze_raw.get("segments"):
        layout = _default_gazebo_maze()
        if "spawn" in maze_raw:
            layout.spawn_xy = np.asarray(maze_raw["spawn"], dtype=float)
        if "goal" in maze_raw:
            layout.goal_xy = np.asarray(maze_raw["goal"], dtype=float)
        return layout, raw

    segments = [
        MazeSegment(
            float(s["x"]),
            float(s["y"]),
            float(s.get("half_x", s.get("length", 0.5) * 0.5)),
            float(s.get("half_y", s.get("thickness", 0.08))),
        )
        for s in maze_raw.get("segments", [])
    ]
    layout = MazeLayout(
        arena_half=float(maze_raw.get("arena_half", 4.5)),
        wall_height=float(maze_raw.get("wall_height", 0.42)),
        wall_thickness=float(maze_raw.get("wall_thickness", 0.08)),
        spawn_xy=np.asarray(maze_raw.get("spawn", [-2.8, 3.0]), dtype=float),
        goal_xy=np.asarray(maze_raw.get("goal", [3.2, -3.2]), dtype=float),
        segments=segments,
        entrance_side=str(maze_raw.get("entrance_side", "north")),
        entrance_width=float(maze_raw.get("entrance_width", 1.5)),
        entrance_offset=float(maze_raw.get("entrance_offset", -0.8)),
    )
    return layout, raw
