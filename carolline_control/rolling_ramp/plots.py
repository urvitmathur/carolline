"""Actual vs desired charts for cage rolling on a ramp."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_log(log_path: Path) -> dict[str, np.ndarray]:
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {log_path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    return {
        "time": col("time"),
        "px": col("px"),
        "py": col("py"),
        "pz": col("pz"),
        "vx": col("vx"),
        "vy": col("vy"),
        "vz": col("vz"),
        "des_px": col("des_px"),
        "des_py": col("des_py"),
        "des_pz": col("des_pz"),
        "des_vx": col("des_vx"),
        "des_vy": col("des_vy"),
        "des_vz": col("des_vz"),
        "ex": col("ex"),
        "ey": col("ey"),
        "ez": col("ez"),
        "phase": np.array([r.get("phase", "") for r in rows]),
    }


def _plot_series(
    out_path: Path,
    time: np.ndarray,
    actual: np.ndarray,
    desired: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    ax.plot(time, desired, color="#1f77b4", linewidth=1.8, label="Desired")
    ax.plot(time, actual, color="#d62728", linewidth=1.2, linestyle="--", label="Actual")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling_ramp_analysis(log_path: str | Path, output_dir: str | Path) -> Path:
    """Generate separate actual-vs-desired charts for cage motion."""
    log_path = Path(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_log(log_path)
    t = data["time"]
    t0 = t[0]
    t = t - t0

    charts = [
        ("cage_position_x.png", data["px"], data["des_px"], "X [m]", "Cage X — Actual vs Desired"),
        ("cage_position_y.png", data["py"], data["des_py"], "Y [m]", "Cage Y — Actual vs Desired"),
        ("cage_position_z.png", data["pz"], data["des_pz"], "Z [m]", "Cage Z — Actual vs Desired"),
        ("cage_velocity_x.png", data["vx"], data["des_vx"], "Vx [m/s]", "Cage Vx — Actual vs Desired"),
        ("cage_velocity_y.png", data["vy"], data["des_vy"], "Vy [m/s]", "Cage Vy — Actual vs Desired"),
        ("cage_velocity_z.png", data["vz"], data["des_vz"], "Vz [m/s]", "Cage Vz — Actual vs Desired"),
    ]
    for fname, actual, desired, ylabel, title in charts:
        _plot_series(output_dir / fname, t, actual, desired, ylabel, title)

    # Tracking errors
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)
    fig.suptitle("Cage Tracking Errors", fontsize=12, fontweight="bold")
    for ax, key, label in zip(axes, ("ex", "ey", "ez"), ("X", "Y", "Z")):
        ax.plot(t, data[key], color="#9467bd", linewidth=1.2)
        ax.set_ylabel(f"e{label.lower()} [m]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(output_dir / "cage_tracking_errors.png", dpi=160)
    plt.close(fig)

    # Top-down ground track
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.plot(data["des_px"], data["des_py"], "-", color="#1f77b4", linewidth=1.5, label="Desired path")
    ax.plot(data["px"], data["py"], "--", color="#d62728", linewidth=1.2, label="Actual path")
    ax.plot(data["px"][0], data["py"][0], "o", color="#2ca02c", markersize=9, label="Start")
    ax.plot(data["px"][-1], data["py"][-1], "s", color="#ff7f0e", markersize=8, label="End")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Cage Ground Track (Top View)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "cage_path_xy.png", dpi=160)
    plt.close(fig)

    # Side profile (height vs distance along X)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.plot(data["des_px"], data["des_pz"], "-", color="#1f77b4", linewidth=1.5, label="Desired profile")
    ax.plot(data["px"], data["pz"], "--", color="#d62728", linewidth=1.2, label="Actual profile")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Cage Height Profile (Side View)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_dir / "cage_path_xz.png", dpi=160)
    plt.close(fig)

    # Speed magnitude
    v_act = np.sqrt(data["vx"] ** 2 + data["vy"] ** 2 + data["vz"] ** 2)
    v_des = np.sqrt(data["des_vx"] ** 2 + data["des_vy"] ** 2 + data["des_vz"] ** 2)
    _plot_series(
        output_dir / "cage_speed.png",
        t,
        v_act,
        v_des,
        "Speed [m/s]",
        "Cage Speed — Actual vs Desired",
    )

    return output_dir
