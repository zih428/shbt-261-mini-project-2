"""Suite-level progress state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vocseg.training.progress import compute_percent, load_history_if_exists, read_json_if_exists, utc_now_iso, write_progress_file
from vocseg.utils.io import load_json


def build_initial_suite_state(suite_name: str, config_paths: list[str], artifact_root: str | Path) -> dict[str, Any]:
    runs = []
    for index, config_path in enumerate(config_paths, start=1):
        config = load_json(config_path) if config_path.endswith(".json") else None
        run_name = config["experiment_name"] if config is not None else Path(config_path).stem
        runs.append(
            {
                "index": index,
                "config_path": str(Path(config_path).resolve()),
                "run_name": run_name,
                "status": "pending",
                "current_epoch": 0,
                "max_epochs": None,
                "percent_complete": 0.0,
                "run_dir": None,
                "best_dev_mIoU": None,
                "last_update": None,
            }
        )
    return {
        "suite_name": suite_name,
        "artifact_root": str(Path(artifact_root).resolve()),
        "status": "pending",
        "total_runs": len(runs),
        "completed_runs": 0,
        "current_run_index": None,
        "current_run_name": None,
        "overall_percent": 0.0,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "runs": runs,
    }


def update_suite_overall_progress(state: dict[str, Any]) -> dict[str, Any]:
    total_runs = int(state["total_runs"])
    completed = 0
    in_progress_fraction = 0.0
    current_run_index = None
    current_run_name = None
    for run in state["runs"]:
        if run["status"] == "completed":
            completed += 1
        elif run["status"] in {"running", "resuming", "interrupted"} and current_run_index is None:
            current_run_index = run["index"]
            current_run_name = run["run_name"]
            in_progress_fraction = float(run.get("percent_complete", 0.0)) / 100.0
    state["completed_runs"] = completed
    state["current_run_index"] = current_run_index
    state["current_run_name"] = current_run_name
    state["overall_percent"] = compute_percent(completed + in_progress_fraction, total_runs)
    state["updated_at"] = utc_now_iso()
    return state


def load_suite_state(path: str | Path) -> dict[str, Any] | None:
    return read_json_if_exists(path)


def save_suite_state(path: str | Path, state: dict[str, Any]) -> None:
    update_suite_overall_progress(state)
    write_progress_file(path, state)
