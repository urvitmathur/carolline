"""Controller gain sensitivity sweeps."""

from __future__ import annotations

from pathlib import Path

from carolline_control.validation.config import ValidationConfig
from carolline_control.validation.metrics import RunMetrics
from carolline_control.validation.randomization import sample_run_parameters
from carolline_control.validation.runner import ValidationRunner
from carolline_control.validation.types import SensitivityPoint


class SensitivityAnalyzer:
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

    def run(
        self,
        *,
        runs_per_point: int = 10,
        gains: list[str] | None = None,
    ) -> list[SensitivityPoint]:
        from carolline_control.config_loader import load_raw_config

        raw = load_raw_config(self.repo_root / self.val_cfg.controller_config)
        mission = raw.get("mission", {})
        spawn = mission.get("spawn_xy", [0.0, 0.0])
        ground_z = float(raw.get("ground_z", 0.40))
        gain_list = gains or self.val_cfg.sensitivity_gains
        fractions = self.val_cfg.sensitivity_fractions
        points: list[SensitivityPoint] = []

        for gain in gain_list:
            for frac in fractions:
                scale = 1.0 + float(frac)
                batch: list[RunMetrics] = []
                for i in range(runs_per_point):
                    run_id = len(points) * runs_per_point + i
                    params = sample_run_parameters(
                        run_id=run_id,
                        seed=self.val_cfg.base_seed + run_id,
                        bounds=self.val_cfg.randomization,
                        spawn_xy=spawn,
                        ground_z=ground_z,
                        sim_duration=self.val_cfg.sim_duration,
                        campaign=f"sensitivity_{gain}",
                        overrides={
                            "randomization": {
                                "wind_magnitude": [0.0, 3.0],
                            }
                        },
                        gain_overrides={gain: scale},
                    )
                    result = self.runner.run_single(params, campaign_mode="nominal")
                    batch.append(result.metrics)
                pt = SensitivityPoint(gain=gain, fraction=frac, scale=scale, metrics=batch)
                points.append(pt)
                print(f"  {gain} {frac:+.0%}: success={sum(m.mission_success for m in batch)/len(batch):.0%}")

        out_dir = self.repo_root / self.val_cfg.output_dir / "sensitivity"
        from carolline_control.validation.statistics import plot_sensitivity_curves
        from carolline_control.validation.storage import save_sensitivity_results

        save_sensitivity_results(out_dir, points, self.val_cfg)
        plot_sensitivity_curves(out_dir, points)
        return points
