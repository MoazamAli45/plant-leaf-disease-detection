"""Configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _list(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.environ.get(name, default).split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LEAFDX_MODEL", _ROOT / "model" / "inference.keras")
        )
    )
    classes_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LEAFDX_CLASSES", _ROOT / "model" / "classes.json")
        )
    )
    max_upload_bytes: int = int(os.environ.get("LEAFDX_MAX_UPLOAD_MB", "10")) * 1024 * 1024
    max_pixels: int = int(os.environ.get("LEAFDX_MAX_PIXELS", str(40_000_000)))
    rate_limit_per_minute: int = int(os.environ.get("LEAFDX_RATE_LIMIT", "30"))
    cors_origins: list[str] = field(
        default_factory=lambda: _list("LEAFDX_CORS_ORIGINS", "http://localhost:3000")
    )
    # Same thresholds as the web app (web/src/lib/verdict.ts)
    confident: float = 0.7
    not_sure: float = 0.4
    min_plant_pixels: float = 0.12
