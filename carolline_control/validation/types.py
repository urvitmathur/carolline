"""Shared validation datatypes."""

from __future__ import annotations

from dataclasses import dataclass

from carolline_control.validation.metrics import RunMetrics


@dataclass
class SensitivityPoint:
    gain: str
    fraction: float
    scale: float
    metrics: list[RunMetrics]
