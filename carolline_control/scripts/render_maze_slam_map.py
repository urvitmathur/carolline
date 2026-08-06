"""Re-render report SLAM figures from a saved log-odds cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carolline_control.logging.paths import resolve_repo_path
from carolline_control.navigation.map_viz import MapCropBounds, MapVisualizer
from carolline_control.navigation.mapping import OccupancyGridMapper
from carolline_control.navigation.maze_layout import load_maze_layout
from carolline_control.navigation.maze_map_viz import ground_truth_image, save_ground_truth_png
from carolline_control.navigation.slam_stack import load_slam_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render maze SLAM report figures")
    parser.add_argument(
        "--cache",
        default="carolline_control/plots/report_figures/maze_slam_log_odds.npz",
    )
    parser.add_argument(
        "--maze-config",
        default=str(REPO_ROOT / "carolline_control" / "navigation" / "maze_config.yaml"),
    )
    args = parser.parse_args()

    cache_path = resolve_repo_path(REPO_ROOT, args.cache)
    data = np.load(cache_path)
    maze_layout, maze_raw = load_maze_layout(Path(args.maze_config))
    slam_config, _ = load_slam_config(Path(args.maze_config))

    mapper = OccupancyGridMapper(
        origin_xy=data["origin"],
        width_m=float(data["width_m"]),
        height_m=float(data["height_m"]),
        resolution=float(data["resolution"]),
        log_odds_hit=slam_config.log_odds_hit,
        log_odds_miss=slam_config.log_odds_miss,
        inflation_radius=slam_config.inflation_radius,
        free_log_odds=slam_config.free_log_odds,
        occupied_log_odds=slam_config.occupied_log_odds,
        min_hit_range=slam_config.min_hit_range,
    )
    mapper._log_odds = data["log_odds"]

    spawn_xy = maze_layout.spawn_xy
    goal_xy = maze_layout.goal_xy
    origin = mapper.origin
    crop_bounds = MapCropBounds.around_center((0.0, 0.0), maze_layout.arena_half, margin=0.45)
    map_viz = MapVisualizer(mapper)

    report_dir = resolve_repo_path(REPO_ROOT, "carolline_control/plots/report_figures")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "fig_slam_map_report.png"
    legacy_path = report_dir / "fig_slam_map.png"
    compare_path = report_dir / "fig_slam_vs_ground_truth.png"
    gt_path = report_dir / "fig_maze_ground_truth.png"

    traj_xy = data["trajectory_xy"] if "trajectory_xy" in data else None
    map_viz.save_report_png(
        report_path,
        goal_xy=goal_xy,
        spawn_xy=spawn_xy,
        trajectory_xy=traj_xy,
        crop_bounds=crop_bounds,
        title="SLAM occupancy map (oracle pose)",
    )
    gt_img = ground_truth_image(
        maze_layout,
        origin_xy=origin,
        width_m=slam_config.map_width_m,
        height_m=slam_config.map_height_m,
        resolution=slam_config.map_resolution,
    )
    save_ground_truth_png(
        maze_layout,
        gt_path,
        origin_xy=origin,
        width_m=slam_config.map_width_m,
        height_m=slam_config.map_height_m,
        resolution=slam_config.map_resolution,
        goal_xy=goal_xy,
        spawn_xy=spawn_xy,
        crop_bounds=crop_bounds,
    )
    map_viz.save_comparison_png(
        compare_path,
        gt_img,
        goal_xy=goal_xy,
        spawn_xy=spawn_xy,
        trajectory_xy=traj_xy,
        crop_bounds=crop_bounds,
    )

    import shutil

    shutil.copy2(report_path, legacy_path)
    print(f"Report map: {report_path}")
    print(f"Comparison: {compare_path}")


if __name__ == "__main__":
    main()
