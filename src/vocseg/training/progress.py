"""Progress-state helpers for resumable training and suites."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from vocseg.utils.io import load_json, save_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def load_history_if_exists(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def write_progress_file(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(payload, path)


def compute_percent(completed_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 0.0
    return 100.0 * float(completed_steps) / float(total_steps)
