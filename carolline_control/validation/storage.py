"""Persist validation outputs (CSV, NumPy, JSON)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.metrics import RunMetrics
from carolline_control.validation.types import SensitivityPoint
from carolline_control.validation.statistics import compute_statistics, generate_plots, save_statistics_json


def _metrics_rows(metrics: list[RunMetrics]) -> list[dict[str, Any]]:
    rows = []
    for m in metrics:
        row = m.to_dict()
        row.pop("params", None)
        rows.append(row)
    return rows


def save_campaign_results(
    out_dir: Path,
    campaign: str,
    metrics: list[RunMetrics],
    val_cfg: ValidationConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _metrics_rows(metrics)
    if not rows:
        return

    csv_path = out_dir / "run_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_dir / "run_metrics.npz",
        **{k: np.array([r[k] for r in rows]) for k in rows[0] if k != "failure_cause"},
        failure_cause=np.array([r["failure_cause"] for r in rows]),
    )

    stats = compute_statistics(metrics)
    save_statistics_json(out_dir, stats)
    generate_plots(out_dir, metrics)

    meta = {
        "campaign": campaign,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(metrics),
        "validation_config": {
            "num_runs": val_cfg.num_runs,
            "base_seed": val_cfg.base_seed,
            "sim_duration": val_cfg.sim_duration,
            "mission_timeout": val_cfg.mission_timeout,
            "controller_config": val_cfg.controller_config,
        },
        "statistics_summary": {
            "success_rate": stats.get("success_rate"),
            "failure_histogram": stats.get("failure_histogram"),
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    params_path = out_dir / "run_parameters.json"
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump([m.params for m in metrics], handle, indent=2)


def save_sensitivity_results(out_dir: Path, points: list[SensitivityPoint], val_cfg: ValidationConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for pt in points:
        payload.append(
            {
                "gain": pt.gain,
                "fraction": pt.fraction,
                "scale": pt.scale,
                "success_rate": float(np.mean([1.0 if m.mission_success else 0.0 for m in pt.metrics])),
                "mean_position_error": float(np.mean([m.max_position_error for m in pt.metrics])),
                "mean_recovery_time": float(np.mean([m.recovery_time_s for m in pt.metrics])),
                "mean_energy": float(np.mean([m.total_energy_j for m in pt.metrics])),
            }
        )
    with (out_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gains": val_cfg.sensitivity_gains,
        "sweep_fractions": val_cfg.sensitivity_fractions,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
