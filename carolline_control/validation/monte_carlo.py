"""Monte Carlo orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from carolline_control.validation.campaigns import CampaignRunSpec, iter_campaign_runs
from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.metrics import RunMetrics
from carolline_control.validation.runner import RunResult, ValidationRunner
from carolline_control.validation.storage import save_campaign_results


@dataclass
class MonteCarloResult:
    campaign: str
    metrics: list[RunMetrics]
    output_dir: Path


class MonteCarloValidator:
    def __init__(
        self,
        val_cfg: ValidationConfig,
        repo_root: Path | None = None,
        *,
        sensor_only: bool = False,
    ) -> None:
        self.val_cfg = val_cfg
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.runner = ValidationRunner(val_cfg, self.repo_root, sensor_only=sensor_only)

    def run_campaign(
        self,
        campaign_name: str = "nominal",
        *,
        num_runs: int | None = None,
        progress_cb: Callable[[int, int, RunMetrics], None] | None = None,
    ) -> MonteCarloResult:
        from carolline_control.config_loader import load_raw_config

        raw = load_raw_config(self.repo_root / self.val_cfg.controller_config)
        specs: list[CampaignRunSpec] = list(
            iter_campaign_runs(campaign_name, self.val_cfg, raw, num_runs=num_runs)
        )
        results: list[RunMetrics] = []
        total = len(specs)

        for idx, spec in enumerate(specs):
            run_result: RunResult = self.runner.run_single(
                spec.run_params,
                campaign_mode=spec.campaign_mode,
            )
            results.append(run_result.metrics)
            if progress_cb:
                progress_cb(idx + 1, total, run_result.metrics)

        out_dir = self.repo_root / self.val_cfg.output_dir / campaign_name
        save_campaign_results(out_dir, campaign_name, results, self.val_cfg)
        return MonteCarloResult(campaign=campaign_name, metrics=results, output_dir=out_dir)
