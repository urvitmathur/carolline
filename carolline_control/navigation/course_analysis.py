"""Actual vs desired charts for autonomous course navigation logs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from carolline_control.plots.trajectory_tracking import plot_actual_vs_desired


def _load_rows(log_path: Path) -> list[dict[str, str]]:
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _phase_segments(rows: list[dict]) -> list[tuple[str, int, int]]:
    if not rows or "phase" not in rows[0]:
        return []
    segments: list[tuple[str, int, int]] = []
    start = 0
    phase = rows[0]["phase"]
    for idx, row in enumerate(rows[1:], start=1):
        if row["phase"] != phase:
            segments.append((phase, start, idx))
            phase = row["phase"]
            start = idx
    segments.append((phase, start, len(rows)))
    return segments


def _shade_phases(ax, rows: list[dict], alpha: float = 0.08) -> None:
    colors = {
        "ROLL_TO_CHECKPOINT": "#aec7e8",
        "ROLL_RECOVER": "#ffbb78",
        "BLOCKED_RECOVER": "#ff9896",
        "TAKEOFF_OVER": "#c5b0d5",
        "FLY_OVER": "#98df8a",
        "LAND_BEYOND": "#c49c94",
        "FLY_TERRAIN": "#17becf",
        "LAND_TERRAIN": "#bcbd22",
        "DONE": "#d9d9d9",
    }
    time = _f(rows, "time")
    for phase, i0, i1 in _phase_segments(rows):
        if i1 <= i0:
            continue
        ax.axvspan(time[i0], time[i1 - 1], color=colors.get(phase, "#eeeeee"), alpha=alpha, lw=0)


def plot_course_tracking(
    log_path: str | Path,
    output_dir: str | Path,
    *,
    title: str = "Autonomous Course Tracking",
) -> list[Path]:
    """Generate actual vs desired charts for a course flight log."""
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    rows = _load_rows(log_path)
    if not rows:
        raise ValueError(f"No data in log file: {log_path}")

    main_plot = output_dir / "course_actual_vs_desired.png"
    plot_actual_vs_desired(log_path, main_plot, title=title)
    saved.append(main_plot)

    time = _f(rows, "time")
    actual = np.column_stack((_f(rows, "px"), _f(rows, "py"), _f(rows, "pz")))
    desired = np.column_stack((_f(rows, "des_px"), _f(rows, "des_py"), _f(rows, "des_pz")))
    actual_speed = np.hypot(_f(rows, "vx"), _f(rows, "vy"))
    desired_speed = np.hypot(_f(rows, "des_vx"), _f(rows, "des_vy"))
    cmd_speed = _f(rows, "cmd_speed") if "cmd_speed" in rows[0] else desired_speed

    labels = ("X [m]", "Y [m]", "Z [m]")
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 9.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"{title} — position and speed", fontsize=12, fontweight="bold")

    for idx, (ax, label) in enumerate(zip(axes[:3], labels)):
        _shade_phases(ax, rows)
        ax.plot(time, desired[:, idx], color="#1f77b4", linewidth=1.6, label="Desired")
        ax.plot(time, actual[:, idx], color="#d62728", linewidth=1.1, linestyle="--", label="Actual")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    ax = axes[3]
    _shade_phases(ax, rows)
    ax.plot(time, desired_speed, color="#1f77b4", linewidth=1.6, label="Desired |v_xy|")
    ax.plot(time, actual_speed, color="#d62728", linewidth=1.1, linestyle="--", label="Actual |v_xy|")
    if "cmd_speed" in rows[0]:
        ax.plot(time, cmd_speed, color="#9467bd", linewidth=1.0, alpha=0.85, label="Rolling cmd speed")
    ax.set_ylabel("Speed [m/s]")
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    pos_path = output_dir / "course_position_speed.png"
    fig.savefig(pos_path, dpi=160)
    plt.close(fig)
    saved.append(pos_path)

    fig2, ax2 = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    ax2.plot(desired[:, 0], desired[:, 1], color="#1f77b4", linewidth=1.0, alpha=0.35, label="Desired path")
    ax2.plot(actual[:, 0], actual[:, 1], color="#d62728", linewidth=1.5, label="Actual path")
    ax2.plot(actual[0, 0], actual[0, 1], "o", color="#2ca02c", markersize=9, label="Start")
    ax2.plot(actual[-1, 0], actual[-1, 1], "s", color="#ff7f0e", markersize=8, label="End")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_title("Course ground track")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    track_path = output_dir / "course_ground_track.png"
    fig2.savefig(track_path, dpi=160)
    plt.close(fig2)
    saved.append(track_path)

    return saved
