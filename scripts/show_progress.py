#!/usr/bin/env python3
"""Print saved run or suite progress."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vocseg.training.progress import read_json_if_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Run directory, suite directory, or direct progress.json path.")
    return parser.parse_args()


def resolve_progress_path(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if path.is_dir():
        return path / "progress.json"
    return path


def print_suite(progress: dict) -> None:
    current = progress.get("current_run_name") or "-"
    current_index = progress.get("current_run_index") or "-"
    print(
        f"suite={progress['suite_name']} status={progress['status']} "
        f"completed={progress['completed_runs']}/{progress['total_runs']} "
        f"overall={progress['overall_percent']:.1f}% current={current_index}:{current}"
    )
    for run in progress["runs"]:
        max_epochs = run.get("max_epochs") or "-"
        print(
            f"  [{run['index']}/{progress['total_runs']}] {run['run_name']} "
            f"status={run['status']} epoch={run.get('current_epoch', 0)}/{max_epochs} "
            f"percent={float(run.get('percent_complete', 0.0)):.1f}%"
        )


def print_run(progress: dict) -> None:
    suite = progress.get("suite", {})
    suite_prefix = ""
    if suite.get("run_index") and suite.get("total_runs"):
        suite_prefix = f"model={suite['run_index']}/{suite['total_runs']} "
    latest = progress.get("latest_epoch_metrics", {})
    max_epochs = progress.get("max_epochs", "-")
    print(
        f"{suite_prefix}run={progress['run_name']} status={progress['status']} "
        f"epoch={progress.get('current_epoch', 0)}/{max_epochs} "
        f"percent={float(progress.get('percent_complete', 0.0)):.1f}% "
        f"best_mIoU={progress.get('best_dev_metrics', {}).get('mIoU')}"
    )
    if latest:
        print(
            f"  latest: train_loss={latest.get('train_loss')} "
            f"dev_mIoU={latest.get('dev_mIoU')} eta_s={latest.get('eta_seconds')}"
        )


def main() -> None:
    path = resolve_progress_path(parse_args().path)
    progress = read_json_if_exists(path)
    if progress is None:
        raise FileNotFoundError(f"No progress file found at {path}")
    if "runs" in progress:
        print_suite(progress)
    else:
        print_run(progress)


if __name__ == "__main__":
    main()
