"""CLI entry point for the validation suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from carolline_control.config_loader import load_config
from carolline_control.validation.config import load_validation_config
from carolline_control.validation.monte_carlo import MonteCarloValidator
from carolline_control.validation.report import generate_pdf_report
from carolline_control.validation.sensitivity import SensitivityAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="CAROLLINE research validation suite")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "validation.yaml"),
        help="Validation config YAML",
    )
    parser.add_argument("--campaign", default="nominal", help="Disturbance campaign name")
    parser.add_argument("--runs", type=int, default=None, help="Override number of Monte Carlo runs")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration [s]")
    parser.add_argument("--timeout", type=float, default=None, help="Override mission timeout [s]")
    parser.add_argument("--sensitivity", action="store_true", help="Run gain sensitivity sweep")
    parser.add_argument("--sensitivity-runs", type=int, default=10, help="Runs per sensitivity point")
    parser.add_argument("--no-report", action="store_true", help="Skip PDF report generation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    val_cfg = load_validation_config(repo_root / args.config if not Path(args.config).is_absolute() else args.config)
    if args.runs is not None:
        val_cfg.num_runs = args.runs
    if args.duration is not None:
        val_cfg.sim_duration = args.duration
    if args.timeout is not None:
        val_cfg.mission_timeout = args.timeout

    if args.sensitivity:
        print("Running parameter sensitivity analysis...")
        analyzer = SensitivityAnalyzer(val_cfg, repo_root)
        analyzer.run(runs_per_point=args.sensitivity_runs)
        print("Sensitivity analysis complete.")
        return

    print(f"Running Monte Carlo campaign '{args.campaign}' ({val_cfg.num_runs} runs)...")

    def progress(done: int, total: int, m) -> None:
        if done % max(1, total // 20) == 0 or done == total:
            print(f"  [{done}/{total}] success={m.mission_success}  mode={m.final_mode}  cause={m.failure_cause}")

    mc = MonteCarloValidator(val_cfg, repo_root)
    result = mc.run_campaign(args.campaign, num_runs=val_cfg.num_runs, progress_cb=progress)

    success_rate = sum(1 for m in result.metrics if m.mission_success) / max(len(result.metrics), 1)
    print(f"Campaign complete. Success rate: {100 * success_rate:.1f}%")
    print(f"Results: {result.output_dir}")

    if not args.no_report:
        ctrl = load_config(repo_root / val_cfg.controller_config)
        gains = {
            k: float(getattr(ctrl, k))
            for k in val_cfg.sensitivity_gains
            if hasattr(ctrl, k)
        }
        pdf = generate_pdf_report(result.output_dir, args.campaign, result.metrics, val_cfg, gains)
        print(f"PDF report: {pdf}")


if __name__ == "__main__":
    main()
