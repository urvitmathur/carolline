"""Layout helpers for the hybrid roll/fly maze course."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CAROLLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CAROLLINE_ROOT / "navigation" / "hybrid_maze_config.yaml"


@dataclass
class WallBox:
    x: float
    y: float
    half_x: float
    half_y: float
    height: float


@dataclass
class PlatformSpec:
    x: float
    y: float
    half_x: float
    half_y: float
    height: float

    @property
    def top_z(self) -> float:
        return float(self.height)

    def com_z(self, cage_radius: float) -> float:
        return self.top_z + float(cage_radius)

    def contains_xy(self, xy: np.ndarray, *, margin: float = 0.20) -> bool:
        return (
            abs(float(xy[0]) - self.x) <= self.half_x - margin
            and abs(float(xy[1]) - self.y) <= self.half_y - margin
        )


@dataclass
class GapSpec:
    x: float
    y: float
    half_x: float
    half_y: float


@dataclass
class HybridMazeLayout:
    arena_half_x: float
    arena_half_y: float
    wall_height: float
    wall_thickness: float
    spawn_xy: np.ndarray
    goal_xy: np.ndarray
    platform: PlatformSpec
    gap: GapSpec
    low_wall: WallBox
    walls: list[WallBox]
    checkpoints: dict[str, np.ndarray]
    cage_radius: float = 0.40
    hop_clearance: float = 0.55
    gap_clearance: float = 0.70
    shaft_clearance: float = 0.45
    hover_settle_s: float = 0.8
    land_hold_s: float = 0.9
    fly_xy_tol: float = 0.40
    fly_z_tol: float = 0.45
    arrival_radius: float = 0.45
    hover_height_nominal: float = 1.0

    def ground_z(self, xy: np.ndarray | None = None) -> float:
        _ = xy
        return 0.0

    def near_gap_xy(self, xy: np.ndarray, *, margin_y: float = 0.35, margin_x: float = 1.0) -> bool:
        px, py = float(xy[0]), float(xy[1])
        g = self.gap
        return (
            abs(py - g.y) <= g.half_y + margin_y
            and (g.x - g.half_x - margin_x) <= px <= (g.x + g.half_x + 0.4)
        )

    def near_low_wall_xy(self, xy: np.ndarray, *, margin: float = 1.2) -> bool:
        px, py = float(xy[0]), float(xy[1])
        lw = self.low_wall
        return abs(px - lw.x) <= lw.half_x + margin and abs(py - lw.y) <= lw.half_y + margin

    def near_climb_xy(self, xy: np.ndarray, *, margin_x: float = 1.4, margin_y: float = 0.8) -> bool:
        px, py = float(xy[0]), float(xy[1])
        plat = self.platform
        return (
            (plat.x - plat.half_x - margin_x) <= px <= (plat.x - plat.half_x + 0.5)
            and abs(py - plat.y) <= plat.half_y + margin_y
        )

    def hop_waypoints(self, *, hover_height: float) -> list[np.ndarray]:
        """Lift → over low wall → land side."""
        approach = self.checkpoints["hop_approach"]
        land = self.checkpoints["hop_land"]
        mid = 0.5 * (approach + land)
        wall_top = self.low_wall.height
        cruise_z = max(hover_height, wall_top + self.hop_clearance + self.cage_radius)
        return [
            np.array([approach[0], approach[1], cruise_z], dtype=float),
            np.array([mid[0], mid[1], cruise_z + 0.15], dtype=float),
            np.array([land[0], land[1], cruise_z], dtype=float),
        ]

    def gap_waypoints(self, *, hover_height: float) -> list[np.ndarray]:
        """Lift → cross trench → far edge."""
        approach = self.checkpoints["gap_approach"]
        land = self.checkpoints["gap_land"]
        mid = np.array([self.gap.x, self.gap.y], dtype=float)
        cruise_z = max(hover_height, self.gap_clearance + self.cage_radius + 0.35)
        return [
            np.array([approach[0], approach[1], cruise_z], dtype=float),
            np.array([mid[0], mid[1], cruise_z + 0.20], dtype=float),
            np.array([land[0], land[1], cruise_z], dtype=float),
        ]

    def shaft_waypoints(self, *, hover_height: float) -> list[np.ndarray]:
        """Climb from shaft base onto platform landing pad."""
        base = self.checkpoints["shaft_base"]
        land = self.checkpoints["platform_land"]
        return self.climb_waypoints(base, land, hover_height)

    def wall_hop_waypoints(
        self,
        pose_xy: np.ndarray,
        bearing: float,
        hit_range: float,
        hover_height: float,
    ) -> list[np.ndarray]:
        """Reactive hop over a low wall from current pose and forward hit."""
        forward = np.array([np.cos(bearing), np.sin(bearing)], dtype=float)
        cruise_z = max(
            hover_height,
            self.low_wall.height + self.hop_clearance + self.cage_radius,
        )
        pre = np.asarray(pose_xy[:2], dtype=float) - forward * 0.25
        over = np.asarray(pose_xy[:2], dtype=float) + forward * (float(hit_range) + 0.55)
        post = self.checkpoints["hop_land"].copy()
        if float(np.linalg.norm(post - over)) > 1.8:
            post = over + forward * 0.65
        return [
            np.array([pre[0], pre[1], cruise_z], dtype=float),
            np.array([over[0], over[1], cruise_z + 0.12], dtype=float),
            np.array([post[0], post[1], cruise_z], dtype=float),
        ]

    def gap_cross_waypoints(
        self,
        pose_xy: np.ndarray,
        bearing: float,
        hover_height: float,
    ) -> list[np.ndarray]:
        """Fly across floor gap trench from edge pose."""
        forward = np.array([np.cos(bearing), np.sin(bearing)], dtype=float)
        cruise_z = max(hover_height, self.gap_clearance + self.cage_radius + 0.35)
        start = np.asarray(pose_xy[:2], dtype=float)
        mid = np.array([self.gap.x, self.gap.y], dtype=float)
        end = self.checkpoints["gap_land"].copy()
        if float(np.dot(end - start, forward)) < 0.0:
            end = start + forward * (2.0 * self.gap.half_x + 0.90)
        return [
            np.array([start[0], start[1], cruise_z], dtype=float),
            np.array([mid[0], mid[1], cruise_z + 0.18], dtype=float),
            np.array([end[0], end[1], cruise_z], dtype=float),
        ]

    def climb_waypoints(
        self,
        pose_xy: np.ndarray,
        goal_xy: np.ndarray,
        hover_height: float,
    ) -> list[np.ndarray]:
        """Climb vertical face onto platform landing pad."""
        base = np.asarray(pose_xy[:2], dtype=float)
        land = np.asarray(goal_xy[:2], dtype=float)
        pad_z = self.platform.com_z(self.cage_radius)
        cruise_z = max(
            hover_height,
            self.platform.top_z + self.shaft_clearance + self.cage_radius,
            pad_z + 0.55,
        )
        return [
            np.array([base[0], base[1], cruise_z * 0.55], dtype=float),
            np.array([base[0], base[1], cruise_z], dtype=float),
            np.array([land[0], land[1], cruise_z], dtype=float),
            np.array([land[0], land[1], pad_z + 0.35], dtype=float),
        ]


def load_hybrid_maze_layout(
    path: Path | None = None,
    *,
    cage_radius: float = 0.40,
    hover_height: float = 1.0,
) -> tuple[HybridMazeLayout, dict]:
    cfg_path = path or DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    arena = raw.get("arena", {})
    plat = raw.get("platform", {})
    gap = raw.get("gap", {})
    low = raw.get("low_wall", {})
    flight = raw.get("flight", {})
    roll = raw.get("rolling", {})
    cps = raw.get("checkpoints", {})

    walls = [
        WallBox(
            float(w["x"]),
            float(w["y"]),
            float(w["half_x"]),
            float(w["half_y"]),
            float(w.get("height", arena.get("wall_height", 0.7))),
        )
        for w in raw.get("walls", [])
    ]
    checkpoints = {k: np.asarray(v, dtype=float) for k, v in cps.items()}

    layout = HybridMazeLayout(
        arena_half_x=float(arena.get("half_x", 6.0)),
        arena_half_y=float(arena.get("half_y", 4.0)),
        wall_height=float(arena.get("wall_height", 0.7)),
        wall_thickness=float(arena.get("wall_thickness", 0.2)),
        spawn_xy=np.asarray(raw.get("spawn", [-4.5, 2.5]), dtype=float),
        goal_xy=np.asarray(raw.get("goal", [5.0, 0.0]), dtype=float),
        platform=PlatformSpec(
            float(plat.get("x", 4.5)),
            float(plat.get("y", 0.0)),
            float(plat.get("half_x", 1.5)),
            float(plat.get("half_y", 1.7)),
            float(plat.get("height", 1.2)),
        ),
        gap=GapSpec(
            float(gap.get("x", 1.5)),
            float(gap.get("y", -1.5)),
            float(gap.get("half_x", 0.7)),
            float(gap.get("half_y", 0.65)),
        ),
        low_wall=WallBox(
            float(low.get("x", -2.0)),
            float(low.get("y", -1.0)),
            float(low.get("half_x", 0.7)),
            float(low.get("half_y", 0.1)),
            float(low.get("height", 0.5)),
        ),
        walls=walls,
        checkpoints=checkpoints,
        cage_radius=float(cage_radius),
        hop_clearance=float(flight.get("hop_clearance", 0.55)),
        gap_clearance=float(flight.get("gap_clearance", 0.70)),
        shaft_clearance=float(flight.get("shaft_clearance", 0.45)),
        hover_settle_s=float(flight.get("hover_settle_s", 0.8)),
        land_hold_s=float(flight.get("land_hold_s", 0.9)),
        fly_xy_tol=float(flight.get("fly_xy_tol", 0.40)),
        fly_z_tol=float(flight.get("fly_z_tol", 0.45)),
        arrival_radius=float(roll.get("arrival_radius", 0.45)),
        hover_height_nominal=float(hover_height),
    )
    return layout, raw
