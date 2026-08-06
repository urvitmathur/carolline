"""
Unified entry point for geometric vs LQR comparative studies.

Run from repo root:
    python carolline_control/scripts/compare_controllers.py
    python carolline_control/scripts/compare_controllers.py --domain flight
    python carolline_control/scripts/compare_controllers.py --domain rolling
    python carolline_control/scripts/compare_controllers.py --domain all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LQR vs geometric controller comparisons")
    parser.add_argument(
        "--domain",
        choices=["flight", "rolling", "all"],
        default="all",
        help="Which comparison to run",
    )
    args, extra = parser.parse_known_args()

    scripts = []
    if args.domain in ("flight", "all"):
        scripts.append(REPO_ROOT / "carolline_control" / "scripts" / "compare_flight_controllers.py")
    if args.domain in ("rolling", "all"):
        scripts.append(REPO_ROOT / "carolline_control" / "scripts" / "compare_rolling_controllers.py")

    for script in scripts:
        print(f"\n=== Running {script.name} ===")
        cmd = [sys.executable, str(script), *extra]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("\nComparative study complete. See carolline_control/plots/lqr_comparison/")


if __name__ == "__main__":
    main()
