"""PDF validation report generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.metrics import RunMetrics
from carolline_control.validation.statistics import compute_statistics


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.94
    for line in lines:
        ax.text(0.0, y, line, fontsize=10, va="top", family="monospace", transform=ax.transAxes)
        y -= 0.035
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.exists():
        return
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.97, title, ha="center", fontsize=14, fontweight="bold")
    img = plt.imread(image_path)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(
    out_dir: Path,
    campaign: str,
    metrics: list[RunMetrics],
    val_cfg: ValidationConfig,
    controller_gains: dict[str, float] | None = None,
) -> Path:
    stats = compute_statistics(metrics)
    report_path = out_dir / "validation_report.pdf"
    plot_dir = out_dir / "plots"

    header = [
        f"Campaign: {campaign}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Runs: {len(metrics)}",
        f"Success rate: {100 * stats.get('success_rate', 0):.1f}%",
        f"Failure rate: {100 * stats.get('failure_rate', 0):.1f}%",
        "",
        "Simulation settings:",
        f"  duration={val_cfg.sim_duration}s  timeout={val_cfg.mission_timeout}s",
        f"  base_seed={val_cfg.base_seed}",
        f"  controller_config={val_cfg.controller_config}",
    ]
    if controller_gains:
        header.append("")
        header.append("Controller gains (nominal):")
        for k, v in sorted(controller_gains.items()):
            header.append(f"  {k}: {v}")

    summary_lines = header + ["", "Statistical summaries:"]
    for key in ("mission_time", "max_position_error", "max_tilt_deg", "total_energy_j", "recovery_time_s"):
        block = stats.get(key, {})
        if block:
            summary_lines.append(
                f"  {key}: mean={block.get('mean', 0):.4f}  "
                f"std={block.get('std', 0):.4f}  "
                f"95%CI=[{block.get('ci95_low', 0):.3f}, {block.get('ci95_high', 0):.3f}]"
            )

    fail_hist: dict[str, Any] = stats.get("failure_histogram", {})
    if fail_hist:
        summary_lines.append("")
        summary_lines.append("Failure cause histogram:")
        for cause, count in sorted(fail_hist.items(), key=lambda x: -x[1]):
            summary_lines.append(f"  {cause}: {count}")

    summary_lines.extend(
        [
            "",
            "Conclusions:",
            "  Observer-only validation; controller and state machine unmodified.",
            f"  Mission success under {campaign} disturbances: {100 * stats.get('success_rate', 0):.1f}%.",
        ]
    )

    with PdfPages(report_path) as pdf:
        _text_page(pdf, "CAROLLINE Validation Report", summary_lines)

        plot_titles = [
            ("hist_mission_time.png", "Mission Time Distribution"),
            ("hist_position_error.png", "Position Error Distribution"),
            ("hist_tilt.png", "Tilt Distribution"),
            ("hist_energy.png", "Energy Distribution"),
            ("box_performance.png", "Performance Box Plots"),
            ("scatter_wind_vs_error.png", "Wind vs Tracking Error"),
            ("scatter_wind_vs_recovery.png", "Wind vs Recovery Time"),
            ("scatter_mass_vs_energy.png", "Mass vs Energy"),
        ]
        for fname, title in plot_titles:
            _image_page(pdf, plot_dir / fname, title)

    return report_path
