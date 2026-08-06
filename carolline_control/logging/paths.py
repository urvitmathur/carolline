"""Resolve repository-relative output paths for logs and plots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def resolve_repo_path(repo_root: Path, configured: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(configured)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def fallback_log_path(path: Path) -> Path:
    """Timestamped fallback when the primary log file cannot be opened."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")
