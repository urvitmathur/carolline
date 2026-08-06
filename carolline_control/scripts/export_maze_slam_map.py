"""Export report-quality SLAM maze maps without running the full viewer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use simulate_slam_maze main with fixed args
if __name__ == "__main__":
    sys.argv = [
        "simulate_slam_maze.py",
        "--no-viewer",
        "--oracle",
        "--report-map",
        "--no-save-map",
    ]
    from carolline_control.scripts.simulate_slam_maze import main

    main()
