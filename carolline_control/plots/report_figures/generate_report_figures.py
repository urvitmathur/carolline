"""Generate ESC thrust–PWM and thrust–torque characterisation plots for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "carolline_control" / "plots" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from carolline_control.config_loader import load_config
from carolline_control.controllers.esc_mapper import EscThrustMapper


def plot_esc_characterization() -> Path:
    config = load_config(str(REPO / "carolline_control" / "config.yaml"))
    mapper = EscThrustMapper(
        thrust_forward_max=config.esc_thrust_forward_max,
        thrust_reverse_max=config.esc_thrust_reverse_max,
        reverse_efficiency=config.esc_reverse_efficiency,
    )
    yaw_coeff = config.yaw_drag_coeff

    esc = np.linspace(mapper.ESC_REVERSE, mapper.ESC_FORWARD, 400)
    thrust_from_esc = np.array([mapper.esc_to_thrust(s) for s in esc])

    thrust = np.linspace(-config.esc_thrust_reverse_max, config.esc_thrust_forward_max, 300)
    esc_from_thrust = np.array([mapper.thrust_to_esc(t) for t in thrust])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax = axes[0]
    pos = esc >= mapper.ESC_MID_HIGH
    neg = esc <= mapper.ESC_MID_LOW
    mid = ~(pos | neg)
    ax.plot(thrust_from_esc[pos], esc[pos], "b-", linewidth=2.0, label="Forward region")
    ax.plot(thrust_from_esc[neg], esc[neg], "r-", linewidth=2.0, label="Reverse region")
    ax.plot(thrust_from_esc[mid], esc[mid], "g-", linewidth=2.0, label="Mid deadband")
    ax.axhline(mapper.ESC_CENTER, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Thrust [N]")
    ax.set_ylabel("ESC signal [µs]")
    ax.set_title("(a) Thrust to ESC mapping")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    torque_pos = yaw_coeff[0] * thrust
    torque_neg = yaw_coeff[1] * thrust
    ax.plot(thrust[thrust >= 0], torque_pos[thrust >= 0], "b-", linewidth=2.0, label=r"Motors 1,3 ($\mu>0$)")
    ax.plot(thrust[thrust <= 0], torque_neg[thrust <= 0], "C1-", linewidth=2.0, label=r"Motors 2,4 ($\mu<0$)")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.axvline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Thrust [N]")
    ax.set_ylabel("Yaw torque [Nm]")
    ax.set_title("(b) Thrust to reaction torque")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    out = OUT_DIR / "fig_esc_characterization.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_mode_fsm() -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = {
        "ROLLING": (1.0, 4.5),
        "PRETAKEOFF": (3.5, 4.5),
        "UPRIGHT": (6.0, 4.5),
        "TAKEOFF": (8.0, 4.5),
        "HOVER": (8.0, 2.5),
        "FLIGHT": (6.0, 1.0),
        "LANDING": (3.5, 1.0),
        "IDLE": (1.0, 1.0),
    }
    for name, (x, y) in boxes.items():
        ax.add_patch(plt.Rectangle((x - 0.7, y - 0.35), 1.4, 0.7, fill=False, linewidth=1.5))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold")

    arrows = [
        ("ROLLING", "PRETAKEOFF", "goal reached"),
        ("PRETAKEOFF", "UPRIGHT", "upright"),
        ("UPRIGHT", "TAKEOFF", "settled"),
        ("UPRIGHT", "ROLLING", "resume roll"),
        ("TAKEOFF", "HOVER", "altitude OK"),
        ("HOVER", "FLIGHT", "dwell"),
        ("FLIGHT", "LANDING", "mission done"),
        ("LANDING", "IDLE", "settled"),
        ("ROLLING", "PRETAKEOFF", "contact loss"),
    ]
    for a, b, label in arrows:
        x1, y1 = boxes[a]
        x2, y2 = boxes[b]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"),
        )
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, fontsize=7, ha="center", color="0.35")

    ax.set_title("Mode manager finite-state machine (simplified)")
    out = OUT_DIR / "fig_mode_fsm.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    layers = [
        ("MuJoCo simulation", 5.0, ["Physics", "Terrain", "Rangefinders"]),
        ("State estimation", 3.6, ["Oracle / IMU / SLAM odom"]),
        ("Mission / navigation", 2.2, ["Director", "SLAM stack", "Planner"]),
        ("Control stack", 0.8, ["Mode manager", "Rolling / Flight", "Motor mixer"]),
    ]
    for title, y, items in layers:
        ax.add_patch(plt.Rectangle((0.5, y - 0.45), 11.0, 0.9, fill=True, facecolor="0.93", edgecolor="0.4"))
        ax.text(0.7, y, title, fontsize=10, fontweight="bold", va="center")
        ax.text(3.2, y, " | ".join(items), fontsize=9, va="center")

    ax.annotate("", xy=(6, 4.55), xytext=(6, 4.05), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.annotate("", xy=(6, 3.15), xytext=(6, 2.65), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.annotate("", xy=(6, 1.75), xytext=(6, 1.25), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(11.2, 5.0, "sensors", fontsize=8, rotation=90, va="center")
    ax.text(11.2, 0.8, "motor thrusts", fontsize=8, rotation=90, va="center")
    ax.set_title("Software architecture layers")
    out = OUT_DIR / "fig_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    paths = [plot_esc_characterization(), plot_mode_fsm(), plot_architecture()]
    traj = REPO / "carolline_control" / "plots" / "actual_vs_desired.png"
    if traj.exists():
        import shutil

        shutil.copy(traj, OUT_DIR / "fig_actual_vs_desired.png")
    maze = REPO / "carolline_control" / "logs" / "maze_slam_map_test.png"
    if maze.exists():
        import shutil

        shutil.copy(maze, OUT_DIR / "fig_slam_map.png")
    for p in paths:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
