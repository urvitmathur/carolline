"""
Quantitative control analysis from enriched flight_log.csv.

Generates summary statistics and diagnostic plots:
  - motor thrusts, attitude, position/velocity errors
  - control moments, attitude error, saturation events
  - waypoint segment transitions
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def _mode_segments(rows: list[dict]) -> list[tuple[str, int, int]]:
    if not rows:
        return []
    segments: list[tuple[str, int, int]] = []
    start = 0
    mode = rows[0]["mode"]
    for i, row in enumerate(rows[1:], start=1):
        if row["mode"] != mode:
            segments.append((mode, start, i))
            mode = row["mode"]
            start = i
    segments.append((mode, start, len(rows)))
    return segments


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def summarize(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"Samples: {len(rows)}  Duration: {float(rows[-1]['time'])-float(rows[0]['time']):.1f}s")
    lines.append(f"Modes: {dict(Counter(r['mode'] for r in rows))}")
    lines.append("")

    for mode, i0, i1 in _mode_segments(rows):
        seg = rows[i0:i1]
        if len(seg) < 5:
            continue
        ex = _f(seg, "ex")
        ey = _f(seg, "ey")
        ez = _f(seg, "ez")
        pos_rms = _rms(np.sqrt(ex**2 + ey**2 + ez**2))
        spread = _f(seg, "motor_spread")
        thrust = _f(seg, "thrust")
        tilt = _f(seg, "tilt_deg")
        moment = np.column_stack((_f(seg, "mx"), _f(seg, "my"), _f(seg, "mz")))
        sat_frac = float(np.mean(_f(seg, "motor_saturated")))
        lines.append(
            f"[{mode}] n={len(seg)} dt={float(seg[-1]['time'])-float(seg[0]['time']):.2f}s"
        )
        lines.append(
            f"  pos RMS={pos_rms:.3f}m  |e| max={np.max(np.sqrt(ex**2+ey**2+ez**2)):.3f}m"
        )
        lines.append(
            f"  thrust: mean={thrust.mean():.2f} std={thrust.std():.3f} "
            f"range=[{thrust.min():.2f},{thrust.max():.2f}] N"
        )
        lines.append(
            f"  motor spread: mean={spread.mean():.3f} max={spread.max():.3f} N  "
            f"sat={100*sat_frac:.1f}%"
        )
        lines.append(
            f"  tilt: mean={tilt.mean():.2f} max={tilt.max():.2f} deg  "
            f"|moment| mean={np.linalg.norm(moment, axis=1).mean():.3f} Nm"
        )
        if mode == "FLIGHT":
            roll_err = _f(seg, "roll_deg") - _f(seg, "des_roll_deg")
            pitch_err = _f(seg, "pitch_deg") - _f(seg, "des_pitch_deg")
            lines.append(
                f"  roll err RMS={_rms(roll_err):.2f} deg  "
                f"pitch err RMS={_rms(pitch_err):.2f} deg"
            )
        lines.append("")

    return "\n".join(lines)


def _flight_segment_boundaries(rows: list[dict], seg_dur: float = 6.0) -> list[int]:
    flight_idx = [i for i, r in enumerate(rows) if r["mode"] == "FLIGHT"]
    if not flight_idx:
        return []
    i0 = flight_idx[0]
    t0 = float(rows[i0]["time"])
    bounds = [i0]
    t = t0 + seg_dur
    while True:
        idx = next((i for i in flight_idx if float(rows[i]["time"]) >= t), None)
        if idx is None:
            break
        bounds.append(idx)
        t += seg_dur
    return bounds


def plot_analysis(rows: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    flight = [r for r in rows if r["mode"] == "FLIGHT"]
    if not flight:
        return saved

    t = _f(flight, "time") - float(flight[0]["time"])

    # 1. Motor thrusts (raw Newtons, not normalized)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, label in enumerate(("m1", "m2", "m3", "m4"), start=1):
        ax.plot(t, _f(flight, f"m{i}"), linewidth=0.9, label=f"Motor {i}")
    ax.plot(t, _f(flight, "thrust") / 4.0, "k--", linewidth=1.0, alpha=0.5, label="thrust/4")
    ax.set_ylabel("Thrust [N]")
    ax.set_xlabel("Time [s]")
    ax.set_title("FLIGHT: per-motor thrust (raw)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    p = out_dir / "motors_flight.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 2. Roll / pitch actual vs desired
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    axes[0].plot(t, _f(flight, "roll_deg"), label="actual")
    axes[0].plot(t, _f(flight, "des_roll_deg"), "--", label="desired")
    axes[0].set_ylabel("Roll [deg]")
    axes[1].plot(t, _f(flight, "pitch_deg"), label="actual")
    axes[1].plot(t, _f(flight, "des_pitch_deg"), "--", label="desired")
    axes[1].set_ylabel("Pitch [deg]")
    axes[1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("FLIGHT: attitude tracking")
    p = out_dir / "attitude_flight.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 3. Position error + velocity
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, keys, lbl in zip(
        axes,
        (("ex", "ey", "ez"), ("evx", "evy", "evz"), ("vx", "vy", "vz")),
        ("Position error [m]", "Velocity error [m/s]", "Velocity [m/s]"),
    ):
        for k in keys:
            ax.plot(t, _f(flight, k), linewidth=0.9, label=k)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("FLIGHT: tracking errors")
    p = out_dir / "tracking_flight.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 4. Control effort: moment, motor spread, collective thrust
    moment_norm = np.linalg.norm(
        np.column_stack((_f(flight, "mx"), _f(flight, "my"), _f(flight, "mz"))),
        axis=1,
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t, moment_norm, color="#9467bd", linewidth=0.9)
    axes[0].set_ylabel("|moment| [Nm]")
    axes[1].plot(t, _f(flight, "motor_spread"), color="#ff7f0e", linewidth=0.9)
    axes[1].set_ylabel("Motor spread [N]")
    axes[2].plot(t, _f(flight, "thrust"), color="#2ca02c", linewidth=0.9)
    axes[2].set_ylabel("Collective thrust [N]")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("FLIGHT: control effort")
    p = out_dir / "control_effort_flight.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 5. Waypoint transition zoom (first 3 segment boundaries)
    bounds = _flight_segment_boundaries(rows)
    for bi, start_idx in enumerate(bounds[:3]):
        i_end = bounds[bi + 1] if bi + 1 < len(bounds) else start_idx + int(2.0 / 0.004)
        i_end = min(i_end, len(rows))
        seg = rows[start_idx:i_end]
        if len(seg) < 20:
            continue
        ts = _f(seg, "time") - float(seg[0]["time"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
        axes[0].plot(ts, _f(seg, "m1"), label="m1")
        axes[0].plot(ts, _f(seg, "m2"), label="m2")
        axes[0].plot(ts, _f(seg, "m3"), label="m3")
        axes[0].plot(ts, _f(seg, "m4"), label="m4")
        axes[0].set_ylabel("Motor [N]")
        axes[0].legend(fontsize=8)
        axes[1].plot(ts, np.sqrt(_f(seg, "ex") ** 2 + _f(seg, "ey") ** 2), label="|e_xy|")
        axes[1].plot(ts, _f(seg, "pitch_deg") - _f(seg, "des_pitch_deg"), label="pitch err")
        axes[1].set_ylabel("Error")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Waypoint transition {bi} (first {ts[-1]:.1f}s of segment)")
        p = out_dir / f"waypoint_transition_{bi}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative CAROLLINE control analysis")
    parser.add_argument(
        "--log",
        default=str(Path(__file__).resolve().parent.parent / "logs" / "flight_log.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "plots" / "control_analysis"),
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = load_log(log_path)
    if not rows:
        raise SystemExit(f"Empty log: {log_path}")

    required = {"m1", "roll_deg", "ex", "motor_spread", "mx"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(
            f"Log missing diagnostic columns {missing}. Re-run simulation with updated logger."
        )

    summary = summarize(rows)
    print(summary)

    out_dir = Path(args.output_dir)
    saved = plot_analysis(rows, out_dir)
    print(f"\nSaved {len(saved)} plots to {out_dir}:")
    for p in saved:
        print(f"  {p}")

    summary_path = out_dir / "analysis_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
