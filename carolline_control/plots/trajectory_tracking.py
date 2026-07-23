"""Generate actual vs desired trajectory plots (CAROLLINE paper style)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_desired(
    log_path: str | Path,
    output_path: str | Path,
    title: str = "CAROLLINE Trajectory Tracking",
) -> Path:
    """Plot position tracking error in the style of the CAROLLINE paper."""
    log_path = Path(log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No data in log file: {log_path}")

    time = np.array([float(r["time"]) for r in rows])
    actual = np.column_stack(
        (
            [float(r["px"]) for r in rows],
            [float(r["py"]) for r in rows],
            [float(r["pz"]) for r in rows],
        )
    )
    if "des_px" in rows[0]:
        desired = np.column_stack(
            (
                [float(r["des_px"]) for r in rows],
                [float(r["des_py"]) for r in rows],
                [float(r["des_pz"]) for r in rows],
            )
        )
    else:
        desired = actual.copy()

    labels = ("X [m]", "Y [m]", "Z [m]")
    flight_idx = next((i for i, row in enumerate(rows) if row["mode"] == "FLIGHT"), 0)
    if flight_idx > 0:
        time = time[flight_idx:] - time[flight_idx]
        actual = actual[flight_idx:]
        desired = desired[flight_idx:]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for idx, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, desired[:, idx], color="#1f77b4", linewidth=1.8, label="Desired")
        ax.plot(time, actual[:, idx], color="#d62728", linewidth=1.2, linestyle="--", label="Actual")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)
        if idx == 2:
            ax.set_xlabel("Time [s]")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    rolling_rows = [r for r in rows if r["mode"] == "ROLLING"]
    if rolling_rows:
        roll_path = output_path.parent / "rolling_ground_track.png"
        rt = np.array([float(r["time"]) for r in rolling_rows])
        px = np.array([float(r["px"]) for r in rolling_rows])
        py = np.array([float(r["py"]) for r in rolling_rows])
        des_x = np.array([float(r["des_px"]) for r in rolling_rows])
        des_y = np.array([float(r["des_py"]) for r in rolling_rows])
        fig2, ax2 = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
        ax2.plot(des_x, des_y, "o", color="#1f77b4", markersize=8, label="Desired target")
        ax2.plot(px, py, "-", color="#d62728", linewidth=1.5, label="Actual path")
        ax2.plot(px[0], py[0], "o", color="#2ca02c", markersize=10, label="Start")
        ax2.plot(des_x[-1], des_y[-1], "s", color="#ff7f0e", markersize=9, label="Target")
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.set_title("Rolling Ground Track")
        ax2.axis("equal")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.savefig(roll_path, dpi=160)
        plt.close(fig2)

    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot actual vs desired trajectory from flight log")
    parser.add_argument(
        "--log",
        default=str(Path(__file__).resolve().parent.parent / "logs" / "flight_log.csv"),
        help="Path to flight_log.csv",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "actual_vs_desired.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    out = plot_actual_vs_desired(args.log, args.output)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
