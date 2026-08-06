"""Timeseries logging and report charts for the hybrid roll/fly maze."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from carolline_control.navigation.hybrid_maze_layout import HybridMazeLayout
from carolline_control.utils.types import ControlMode

TIMESERIES_FIELDS = (
    "time",
    "px",
    "py",
    "pz",
    "des_px",
    "des_py",
    "des_pz",
    "vx",
    "vy",
    "vz",
    "speed",
    "mode",
    "phase",
    "dist_to_target",
    "dist_to_goal",
)

DESIRED_FIELDS = ("des_px", "des_py", "des_pz")

FLY_MODES = frozenset(
    {
        ControlMode.TAKEOFF,
        ControlMode.HOVER,
        ControlMode.FLIGHT,
        ControlMode.LANDING,
    }
)

MODE_COLORS = {
    "ROLLING": "#2ca02c",
    "TAKEOFF": "#ff7f0e",
    "HOVER": "#9467bd",
    "FLIGHT": "#d62728",
    "LANDING": "#8c564b",
    "PRETAKEOFF": "#bcbd22",
    "UPRIGHT": "#17becf",
    "IDLE": "#7f7f7f",
}

PHASE_GROUPS = {
    "flight": ("RECOVER", "FLY", "LAND"),
}


def _is_flight_phase(phase: str) -> bool:
    return phase in ("RECOVER", "FLY", "LAND")


def _phase_group(phase: str) -> str:
    if phase in PHASE_GROUPS["flight"]:
        return "flight"
    return "roll"


def _phase_target_xy(phase: str, layout: HybridMazeLayout) -> np.ndarray:
    """Mission checkpoint XY for a hybrid-maze phase."""
    cps = layout.checkpoints
    mapping = {
        "ROLL": cps["deadend"],
        "RECOVER": cps["hop_approach"],
        "FLY": cps["platform_land"],
        "LAND": cps["platform_land"],
        "DONE": layout.goal_xy,
    }
    return mapping.get(phase, layout.goal_xy)


def _scripted_desired_z(mode: str, phase: str, layout: HybridMazeLayout) -> float:
    """Nominal Z reference when controller targets were not logged."""
    radius = layout.cage_radius
    hover = layout.hover_height_nominal
    platform_z = layout.platform.com_z(radius)
    if mode == "ROLLING":
        if phase == "DONE":
            return platform_z
        return radius
    if mode in ("TAKEOFF", "HOVER"):
        return hover
    if mode == "LANDING":
        if phase in ("LAND", "DONE", "FLY"):
            return platform_z
        return radius
    if mode == "FLIGHT":
        if phase == "FLY":
            return float(layout.hop_waypoints(hover_height=hover)[0][2])
    return radius


def _scripted_checkpoint_polyline(layout: HybridMazeLayout) -> np.ndarray:
    """Ordered checkpoint XY polyline for the scripted mission route."""
    cps = layout.checkpoints
    keys = (
        "deadend",
        "junction",
        "hop_approach",
        "hop_land",
        "gap_approach",
        "gap_land",
        "shaft_base",
        "platform_land",
    )
    points = [layout.spawn_xy.copy()]
    for key in keys:
        points.append(cps[key])
    points.append(layout.goal_xy.copy())
    return np.vstack(points)


def _reconstruct_scripted_desired(
    data: dict[str, np.ndarray | list[str]],
    layout: HybridMazeLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = data["phase"]
    modes = data["mode"]
    n = len(phases)
    des_px = np.zeros(n)
    des_py = np.zeros(n)
    des_pz = np.zeros(n)
    for i in range(n):
        xy = _phase_target_xy(str(phases[i]), layout)
        des_px[i] = float(xy[0])
        des_py[i] = float(xy[1])
        des_pz[i] = _scripted_desired_z(str(modes[i]), str(phases[i]), layout)
    return des_px, des_py, des_pz


def _ensure_desired_columns(
    data: dict[str, np.ndarray | list[str]],
    layout: HybridMazeLayout,
) -> tuple[dict[str, np.ndarray | list[str]], str]:
    """Attach desired position columns; reconstruct from phase if missing."""
    if data.get("has_logged_desired"):
        return data, "controller target (logged)"
    des_px, des_py, des_pz = _reconstruct_scripted_desired(data, layout)
    enriched = dict(data)
    enriched["des_px"] = des_px
    enriched["des_py"] = des_py
    enriched["des_pz"] = des_pz
    return enriched, "scripted reference (reconstructed from checkpoints)"


@dataclass
class HybridMazeRecorder:
    """Append-only per-control-step log for hybrid maze runs."""

    layout: HybridMazeLayout
    rows: list[dict[str, float | str]] = field(default_factory=list)
    phase_transitions: list[tuple[float, str, str]] = field(default_factory=list)
    _last_phase: str | None = None

    def record(
        self,
        *,
        time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        mode: ControlMode,
        phase: str,
        desired_position: np.ndarray | None = None,
        scan=None,
        takeoff_reason: str | None = None,
    ) -> None:
        _ = scan, takeoff_reason
        if desired_position is None:
            xy = _phase_target_xy(phase, self.layout)
            z = _scripted_desired_z(mode.name, phase, self.layout)
            desired_position = np.array([xy[0], xy[1], z], dtype=float)
        tgt = desired_position[:2]
        dist_target = float(np.linalg.norm(position[:2] - tgt))
        dist_goal = float(np.linalg.norm(position[:2] - self.layout.goal_xy))
        speed = float(np.linalg.norm(velocity))

        if self._last_phase is not None and phase != self._last_phase:
            self.phase_transitions.append((time, self._last_phase, phase))
        self._last_phase = phase

        self.rows.append(
            {
                "time": float(time),
                "px": float(position[0]),
                "py": float(position[1]),
                "pz": float(position[2]),
                "des_px": float(desired_position[0]),
                "des_py": float(desired_position[1]),
                "des_pz": float(desired_position[2]),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "speed": speed,
                "mode": mode.name,
                "phase": phase,
                "dist_to_target": dist_target,
                "dist_to_goal": dist_goal,
            }
        )

    def _current_target_xy(self, phase: str) -> np.ndarray:
        return _phase_target_xy(phase, self.layout)

    def save_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def save_phase_transitions(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "from_phase", "to_phase"])
            for t, prev, nxt in self.phase_transitions:
                writer.writerow([f"{t:.4f}", prev, nxt])
        return path


def _load_timeseries(csv_path: Path) -> dict[str, np.ndarray | list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows])

    has_logged_desired = all(field in rows[0] for field in DESIRED_FIELDS)
    result: dict[str, np.ndarray | list[str] | bool] = {
        "time": col("time"),
        "px": col("px"),
        "py": col("py"),
        "pz": col("pz"),
        "speed": col("speed"),
        "dist_to_target": col("dist_to_target"),
        "dist_to_goal": col("dist_to_goal"),
        "mode": [r["mode"] for r in rows],
        "phase": [r["phase"] for r in rows],
        "has_logged_desired": has_logged_desired,
    }
    if has_logged_desired:
        result["des_px"] = col("des_px")
        result["des_py"] = col("des_py")
        result["des_pz"] = col("des_pz")
    return result



def _load_phase_transitions(path: Path) -> list[tuple[float, str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (float(row["time_s"]), row["from_phase"], row["to_phase"])
            for row in reader
        ]


def _parse_success_from_summary(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("success="):
            return line.split("=", 1)[1].strip().lower() == "true"
    return False


def _save_figure(fig, path: Path, *, dpi: int = 160) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write plot: {path}")
    return path


def _segment_colors(values: list[str], palette: dict[str, str], default: str) -> list[str]:
    return [palette.get(v, default) for v in values]


def _colored_line_segments(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    palette: dict[str, str],
    *,
    default: str = "#1f77b4",
    linewidth: float = 2.0,
) -> None:
    if len(x) < 2:
        return
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = _segment_colors(labels[:-1], palette, default)
    lc = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round")
    ax.add_collection(lc)


def _draw_layout_annotations(ax, layout: HybridMazeLayout) -> None:
    plat = layout.platform
    ax.add_patch(
        Rectangle(
            (plat.x - plat.half_x, plat.y - plat.half_y),
            2 * plat.half_x,
            2 * plat.half_y,
            fill=False,
            edgecolor="#555555",
            linewidth=1.2,
            linestyle="--",
            label="Platform",
        )
    )
    gap = layout.gap
    ax.add_patch(
        Rectangle(
            (gap.x - gap.half_x, gap.y - gap.half_y),
            2 * gap.half_x,
            2 * gap.half_y,
            fill=True,
            facecolor="#cce5ff",
            edgecolor="#1f77b4",
            alpha=0.35,
            linewidth=1.0,
            label="Gap",
        )
    )
    lw = layout.low_wall
    ax.add_patch(
        Rectangle(
            (lw.x - lw.half_x, lw.y - lw.half_y),
            2 * lw.half_x,
            2 * lw.half_y,
            fill=True,
            facecolor="#ffe0b2",
            edgecolor="#ff7f0e",
            alpha=0.55,
            linewidth=1.0,
            label="Low wall",
        )
    )
    ax.plot(
        layout.spawn_xy[0],
        layout.spawn_xy[1],
        "o",
        color="#2ca02c",
        markersize=10,
        markeredgecolor="black",
        label="Start",
        zorder=5,
    )
    ax.plot(
        layout.goal_xy[0],
        layout.goal_xy[1],
        "*",
        color="#d62728",
        markersize=14,
        markeredgecolor="black",
        label="Goal",
        zorder=5,
    )


def _mode_mobility(mode: str) -> str:
    return "fly" if mode in {m.name for m in FLY_MODES} else "roll"


def _write_hybrid_maze_pngs(
    data: dict[str, np.ndarray | list[str]],
    layout: HybridMazeLayout,
    output_dir: Path,
    phase_transitions: list[tuple[float, str, str]],
    *,
    success: bool,
) -> list[Path]:
    """Write hybrid maze PNG charts; return paths (PNG only)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = data["time"]
    t0 = float(t[0]) if len(t) else 0.0
    t_rel = t - t0
    png_paths: list[Path] = []

    # --- 1. XY trajectory colored by mobility (roll vs fly) ---
    roll_fly_palette = {"roll": "#2ca02c", "fly": "#d62728"}
    mobility = [_mode_mobility(m) for m in data["mode"]]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    _draw_layout_annotations(ax, layout)
    _colored_line_segments(
        ax,
        data["px"],
        data["py"],
        mobility,
        roll_fly_palette,
        linewidth=2.2,
    )
    ax.plot(data["px"][0], data["py"][0], "o", color="#2ca02c", markersize=8, zorder=6)
    ax.plot(data["px"][-1], data["py"][-1], "s", color="#ff7f0e", markersize=8, zorder=6)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Hybrid Maze Trajectory (roll vs fly) — success={success}")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            Line2D([0], [0], color=roll_fly_palette["roll"], linewidth=2.5, label="Rolling"),
            Line2D([0], [0], color=roll_fly_palette["fly"], linewidth=2.5, label="Flying"),
        ],
        loc="upper left",
    )
    xy_path = output_dir / "hybrid_maze_trajectory_xy.png"
    _save_figure(fig, xy_path)
    png_paths.append(xy_path)

    # --- 2. Altitude vs time ---
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax.plot(t_rel, data["pz"], color="#1f77b4", linewidth=1.5, label="Altitude z")
    ax.axhline(layout.platform.top_z, color="#555555", linestyle="--", linewidth=1.0, label="Platform top")
    for name, members in PHASE_GROUPS.items():
        for phase in members:
            if phase in ("RECOVER", "FLY"):
                idx = [i for i, p in enumerate(data["phase"]) if p == phase]
                if idx:
                    ax.axvline(t_rel[idx[0]], color="#ff7f0e", alpha=0.25, linewidth=0.8)
    for t_ev, prev, nxt in phase_transitions:
        if _is_flight_phase(nxt) and nxt in ("RECOVER", "FLY"):
            ax.axvline(t_ev - t0, color="#d62728", alpha=0.35, linewidth=0.9, linestyle=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Altitude vs Time (flight segments marked)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    alt_path = output_dir / "hybrid_maze_altitude.png"
    _save_figure(fig, alt_path)
    png_paths.append(alt_path)

    # --- 3. Mode timeline ---
    unique_modes = sorted(set(data["mode"]), key=lambda m: list(MODE_COLORS).index(m) if m in MODE_COLORS else 99)
    mode_to_y = {m: i for i, m in enumerate(unique_modes)}
    fig, ax = plt.subplots(figsize=(9.0, max(2.8, 0.45 * len(unique_modes) + 1.5)), constrained_layout=True)
    for i in range(len(t_rel) - 1):
        mode = data["mode"][i]
        ax.barh(
            mode_to_y[mode],
            t_rel[i + 1] - t_rel[i],
            left=t_rel[i],
            height=0.72,
            color=MODE_COLORS.get(mode, "#7f7f7f"),
            edgecolor="none",
        )
    ax.set_yticks(list(mode_to_y.values()))
    ax.set_yticklabels(list(mode_to_y.keys()))
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Mode Timeline")
    ax.grid(True, axis="x", alpha=0.3)
    mode_path = output_dir / "hybrid_maze_mode_timeline.png"
    _save_figure(fig, mode_path)
    png_paths.append(mode_path)

    # --- 4. Distance to target / goal ---
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax.plot(t_rel, data["dist_to_target"], color="#9467bd", linewidth=1.4, label="Dist to checkpoint")
    ax.plot(t_rel, data["dist_to_goal"], color="#2ca02c", linewidth=1.4, label="Dist to goal")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.set_title("Progress — Distance to Checkpoint / Goal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    dist_path = output_dir / "hybrid_maze_progress.png"
    _save_figure(fig, dist_path)
    png_paths.append(dist_path)

    # --- 5. Speed vs time ---
    fig, ax = plt.subplots(figsize=(9.0, 3.2), constrained_layout=True)
    ax.plot(t_rel, data["speed"], color="#d62728", linewidth=1.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed vs Time")
    ax.grid(True, alpha=0.3)
    speed_path = output_dir / "hybrid_maze_speed.png"
    _save_figure(fig, speed_path)
    png_paths.append(speed_path)

    data, desired_label = _ensure_desired_columns(data, layout)
    png_paths.extend(
        _write_actual_vs_desired_pngs(
            data,
            layout,
            output_dir,
            success=success,
            desired_label=desired_label,
        )
    )

    return png_paths


def _write_actual_vs_desired_pngs(
    data: dict[str, np.ndarray | list[str]],
    layout: HybridMazeLayout,
    output_dir: Path,
    *,
    success: bool,
    desired_label: str,
) -> list[Path]:
    """XY trajectory and XYZ time-series: actual vs desired reference."""
    output_dir = Path(output_dir)
    t = data["time"]
    t0 = float(t[0]) if len(t) else 0.0
    t_rel = t - t0
    png_paths: list[Path] = []
    checkpoint_poly = _scripted_checkpoint_polyline(layout)

    # --- Actual vs desired XY ---
    roll_fly_palette = {"roll": "#2ca02c", "fly": "#d62728"}
    mobility = [_mode_mobility(m) for m in data["mode"]]
    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    _draw_layout_annotations(ax, layout)
    ax.plot(
        checkpoint_poly[:, 0],
        checkpoint_poly[:, 1],
        "--",
        color="#888888",
        linewidth=1.4,
        alpha=0.85,
        label="Scripted checkpoints",
        zorder=2,
    )
    ax.plot(
        checkpoint_poly[:, 0],
        checkpoint_poly[:, 1],
        "s",
        color="#888888",
        markersize=4,
        alpha=0.7,
        zorder=3,
    )
    ax.plot(
        data["des_px"],
        data["des_py"],
        "--",
        color="#ff7f0e",
        linewidth=1.6,
        alpha=0.9,
        label=f"Desired ({desired_label})",
        zorder=4,
    )
    _colored_line_segments(
        ax,
        data["px"],
        data["py"],
        mobility,
        roll_fly_palette,
        linewidth=2.2,
    )
    ax.plot(data["px"][0], data["py"][0], "o", color="#2ca02c", markersize=8, zorder=6)
    ax.plot(data["px"][-1], data["py"][-1], "s", color="#1f77b4", markersize=8, zorder=6)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Actual vs Desired Path — success={success}")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    xy_path = output_dir / "hybrid_maze_actual_vs_desired_xy.png"
    _save_figure(fig, xy_path)
    png_paths.append(xy_path)

    # --- Actual vs desired X/Y/Z vs time ---
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    labels = ("X [m]", "Y [m]", "Z [m]")
    actual_cols = ("px", "py", "pz")
    desired_cols = ("des_px", "des_py", "des_pz")
    for ax, label, act_key, des_key in zip(axes, labels, actual_cols, desired_cols):
        ax.plot(t_rel, data[act_key], color="#1f77b4", linewidth=1.4, label="Actual")
        ax.plot(
            t_rel,
            data[des_key],
            "--",
            color="#ff7f0e",
            linewidth=1.2,
            alpha=0.9,
            label=f"Desired ({desired_label})",
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"Actual vs Desired Position — success={success}")
    xyz_path = output_dir / "hybrid_maze_actual_vs_desired_xyz.png"
    _save_figure(fig, xyz_path)
    png_paths.append(xyz_path)

    return png_paths


def generate_hybrid_maze_report(
    recorder: HybridMazeRecorder,
    layout: HybridMazeLayout,
    output_dir: str | Path,
    *,
    success: bool,
) -> list[Path]:
    """Write CSV logs and PNG charts; return paths of generated artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = recorder.save_csv(output_dir / "hybrid_maze_timeseries.csv")
    trans_path = recorder.save_phase_transitions(output_dir / "hybrid_maze_phase_transitions.csv")
    data = _load_timeseries(csv_path)
    png_paths = _write_hybrid_maze_pngs(
        data,
        layout,
        output_dir,
        recorder.phase_transitions,
        success=success,
    )
    generated: list[Path] = [csv_path, trans_path, *png_paths]

    # --- Summary text ---
    summary_path = output_dir / "hybrid_maze_summary.txt"
    lines = [
        f"Hybrid roll/fly maze run summary",
        f"success={success}",
        f"samples={len(recorder.rows)}",
        f"duration_s={float(data["time"][-1] - data["time"][0]) if len(data["time"]) else 0.0:.2f}",
        f"phase_transitions={len(recorder.phase_transitions)}",
        "",
        "Phase transitions:",
    ]
    for t_ev, prev, nxt in recorder.phase_transitions:
        lines.append(f"  t={t_ev:7.2f}s  {prev} -> {nxt}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    generated.append(summary_path)

    return generated

def regenerate_hybrid_maze_report_from_dir(
    log_dir: str | Path,
    *,
    maze_config: str | Path | None = None,
    success: bool | None = None,
) -> list[Path]:
    """Rebuild PNG charts from existing hybrid_maze log CSVs."""
    log_dir = Path(log_dir)
    csv_path = log_dir / "hybrid_maze_timeseries.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing timeseries CSV: {csv_path}")

    if maze_config is None:
        maze_config = Path(__file__).resolve().parent / "hybrid_maze_config.yaml"
    from carolline_control.navigation.hybrid_maze_layout import load_hybrid_maze_layout

    layout, _ = load_hybrid_maze_layout(Path(maze_config))
    data = _load_timeseries(csv_path)
    phase_transitions = _load_phase_transitions(log_dir / "hybrid_maze_phase_transitions.csv")
    if success is None:
        success = _parse_success_from_summary(log_dir / "hybrid_maze_summary.txt")
    return _write_hybrid_maze_pngs(
        data,
        layout,
        log_dir,
        phase_transitions,
        success=success,
    )


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid maze report charts")
    parser.add_argument(
        "--from-dir",
        type=Path,
        required=True,
        help="Log directory containing hybrid_maze_timeseries.csv",
    )
    parser.add_argument(
        "--maze-config",
        type=Path,
        default=None,
        help="Optional hybrid_maze_config.yaml for layout annotations",
    )
    parser.add_argument(
        "--success",
        choices=("true", "false"),
        default=None,
        help="Override success flag in chart titles (default: read summary.txt)",
    )
    args = parser.parse_args()
    success = None if args.success is None else args.success == "true"
    paths = regenerate_hybrid_maze_report_from_dir(
        args.from_dir,
        maze_config=args.maze_config,
        success=success,
    )
    for path in paths:
        size = path.stat().st_size
        print(f"{path.resolve()}  ({size} bytes)")


if __name__ == "__main__":
    _cli_main()
