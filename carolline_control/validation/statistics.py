"""Statistical aggregation and plotting for validation results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.metrics import RunMetrics
from carolline_control.validation.types import SensitivityPoint


def _values(metrics: list[RunMetrics], key: str) -> np.ndarray:
    return np.array([getattr(m, key) for m in metrics], dtype=float)


def compute_statistics(metrics: list[RunMetrics]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}

    def stat_vec(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        n = arr.size
        ci_half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        return {
            "mean": mean,
            "median": float(np.median(arr)),
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ci95_low": mean - ci_half,
            "ci95_high": mean + ci_half,
        }

    success = sum(1 for m in metrics if m.mission_success)
    failures = Counter(m.failure_cause for m in metrics if not m.mission_success)

    numeric_keys = [
        "mission_time",
        "max_position_error",
        "mean_position_error",
        "max_attitude_error",
        "max_tilt_deg",
        "motor_saturation_pct",
        "total_energy_j",
        "recovery_time_s",
        "oscillation_metric",
        "max_force_residual",
    ]
    stats = {k: stat_vec(_values(metrics, k)) for k in numeric_keys}
    stats["count"] = len(metrics)
    stats["success_rate"] = success / len(metrics)
    stats["failure_rate"] = 1.0 - stats["success_rate"]
    stats["failure_histogram"] = dict(failures)
    return stats


def _save_histogram(out_dir: Path, values: np.ndarray, title: str, fname: str) -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(5, values.size // 5)), color="#2563eb", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)


def _save_boxplot(out_dir: Path, data: dict[str, np.ndarray], title: str, fname: str) -> None:
    labels = list(data.keys())
    arrays = [data[k] for k in labels]
    if not any(a.size for a in arrays):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(arrays, tick_labels=labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)


def _save_scatter(out_dir: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, fname: str) -> None:
    if x.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.5, s=12, c="#0d9488")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)


def generate_plots(out_dir: Path, metrics: list[RunMetrics]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _save_histogram(plot_dir, _values(metrics, "mission_time"), "Mission Time", "hist_mission_time.png")
    _save_histogram(plot_dir, _values(metrics, "max_position_error"), "Max Position Error", "hist_position_error.png")
    _save_histogram(plot_dir, _values(metrics, "max_tilt_deg"), "Max Tilt", "hist_tilt.png")
    _save_histogram(plot_dir, _values(metrics, "peak_motor_utilization"), "Peak Motor Utilization", "hist_motor_thrust.png")
    _save_histogram(plot_dir, _values(metrics, "recovery_time_s"), "Recovery Time", "hist_recovery_time.png")
    _save_histogram(plot_dir, _values(metrics, "total_energy_j"), "Control Energy", "hist_energy.png")
    _save_histogram(plot_dir, _values(metrics, "wind_magnitude"), "Wind Magnitude", "hist_wind.png")

    _save_boxplot(
        plot_dir,
        {
            "pos_err": _values(metrics, "max_position_error"),
            "att_err": _values(metrics, "max_attitude_error"),
            "energy": _values(metrics, "total_energy_j"),
            "motor": _values(metrics, "peak_motor_utilization"),
            "recovery": _values(metrics, "recovery_time_s"),
        },
        "Performance Distributions",
        "box_performance.png",
    )

    _save_scatter(
        plot_dir,
        _values(metrics, "wind_magnitude"),
        _values(metrics, "max_position_error"),
        "Wind Magnitude [m/s]",
        "Max Position Error [m]",
        "scatter_wind_vs_error.png",
    )
    _save_scatter(
        plot_dir,
        _values(metrics, "wind_magnitude"),
        _values(metrics, "recovery_time_s"),
        "Wind Magnitude [m/s]",
        "Recovery Time [s]",
        "scatter_wind_vs_recovery.png",
    )
    _save_scatter(
        plot_dir,
        _values(metrics, "mass_scale"),
        _values(metrics, "total_energy_j"),
        "Mass Scale",
        "Total Energy [J]",
        "scatter_mass_vs_energy.png",
    )

    success = np.array([1.0 if m.mission_success else 0.0 for m in metrics])
    motor_eff = _values(metrics, "motor_effectiveness")
    if motor_eff.size:
        bins = np.linspace(motor_eff.min(), motor_eff.max(), 8)
        bin_success = []
        bin_centers = []
        for i in range(len(bins) - 1):
            mask = (motor_eff >= bins[i]) & (motor_eff < bins[i + 1])
            if np.any(mask):
                bin_success.append(float(np.mean(success[mask])))
                bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
        if bin_centers:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(bin_centers, bin_success, "o-")
            ax.set_xlabel("Motor Effectiveness")
            ax.set_ylabel("Mission Success Rate")
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(plot_dir / "scatter_motor_eff_vs_success.png", dpi=120)
            plt.close(fig)


def plot_sensitivity_curves(out_dir: Path, points: list[SensitivityPoint]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gains = sorted({p.gain for p in points})

    for gain in gains:
        pts = sorted([p for p in points if p.gain == gain], key=lambda p: p.fraction)
        scales = [p.scale for p in pts]
        track = [float(np.mean([m.max_position_error for m in p.metrics])) for p in pts]
        recovery = [float(np.mean([m.recovery_time_s for m in p.metrics])) for p in pts]
        energy = [float(np.mean([m.total_energy_j for m in p.metrics])) for p in pts]
        succ = [float(np.mean([1.0 if m.mission_success else 0.0 for m in p.metrics])) for p in pts]

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        axes[0, 0].plot(scales, track, "o-")
        axes[0, 0].set_title(f"{gain}: Tracking Error")
        axes[0, 1].plot(scales, recovery, "o-")
        axes[0, 1].set_title(f"{gain}: Recovery Time")
        axes[1, 0].plot(scales, energy, "o-")
        axes[1, 0].set_title(f"{gain}: Energy")
        axes[1, 1].plot(scales, succ, "o-")
        axes[1, 1].set_title(f"{gain}: Success Rate")
        axes[1, 1].set_ylim(-0.05, 1.05)
        for ax in axes.flat:
            ax.set_xlabel("Gain scale")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Sensitivity: {gain}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"sensitivity_{gain}.png", dpi=120)
        plt.close(fig)


def save_statistics_json(out_dir: Path, stats: dict[str, Any]) -> Path:
    path = out_dir / "statistics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return path
