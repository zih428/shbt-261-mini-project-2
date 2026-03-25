#!/usr/bin/env python3
"""Run a sequence of experiment configs with persistent suite progress and resume."""

from __future__ import annotations

import argparse
from pathlib import Path

from vocseg.config import load_config
from vocseg.training.progress import read_json_if_exists, write_progress_file
from vocseg.training.runner import fit
from vocseg.training.suite import build_initial_suite_state, load_suite_state, save_suite_state, update_suite_overall_progress
from vocseg.utils.distributed import cleanup_distributed, init_distributed, is_main_process
from vocseg.utils.repro import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", required=True, help="Logical name for this suite.")
    parser.add_argument("--configs", nargs="+", required=True, help="Ordered config paths to run.")
    parser.add_argument("--data-root", required=True, help="VOC dataset root.")
    parser.add_argument("--artifact-root", default="./artifacts", help="Artifact output root.")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic suite/model resume.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one run fails.")
    return parser.parse_args()


def _resolve_metadata_dir(config: dict, artifact_root: Path) -> str:
    metadata_dir = Path(config["data"].get("metadata_dir", "dataset"))
    if not metadata_dir.is_absolute():
        metadata_dir = artifact_root / metadata_dir
    return str(metadata_dir.resolve())


def _mark_suite_status(suite_state_path: Path, status: str, run_index: int | None = None) -> None:
    state = load_suite_state(suite_state_path)
    if state is None:
        return
    state["status"] = status
    if run_index is not None:
        for run in state["runs"]:
            if int(run["index"]) == int(run_index):
                run["status"] = status
                break
    save_suite_state(suite_state_path, state)


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    suite_dir = artifact_root / "suites" / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_state_path = suite_dir / "progress.json"

    distributed_state = init_distributed()
    try:
        if suite_state_path.exists() and not args.no_resume:
            suite_state = load_suite_state(suite_state_path)
        else:
            suite_state = build_initial_suite_state(args.suite_name, args.configs, artifact_root=artifact_root)
            # Resolve run names from YAML configs instead of file stems.
            for run, config_path in zip(suite_state["runs"], args.configs):
                config = load_config(config_path)
                run["run_name"] = config["experiment_name"]
            save_suite_state(suite_state_path, suite_state)

        total_runs = len(args.configs)
        for index, config_path in enumerate(args.configs, start=1):
            suite_state = load_suite_state(suite_state_path) or suite_state
            run_state = suite_state["runs"][index - 1]
            if run_state["status"] == "completed" and not args.no_resume:
                print(f"[model {index}/{total_runs}] {run_state['run_name']}: already completed, skipping.")
                continue

            config = load_config(config_path)
            config["data"]["metadata_dir"] = _resolve_metadata_dir(config, artifact_root)
            seed_everything(int(config["training"].get("seed", 42)), deterministic=bool(config["training"].get("deterministic", False)))

            run_state["status"] = "running"
            suite_state["status"] = "running"
            suite_state["current_run_index"] = index
            suite_state["current_run_name"] = config["experiment_name"]
            save_suite_state(suite_state_path, suite_state)

            try:
                fit(
                    config=config,
                    data_root=args.data_root,
                    artifact_root=artifact_root,
                    distributed_state=distributed_state,
                    resume=not args.no_resume,
                    progress_context={
                        "suite_state_path": str(suite_state_path),
                        "suite_name": args.suite_name,
                        "run_index": index,
                        "total_runs": total_runs,
                    },
                    print_progress=is_main_process(distributed_state),
                )
                suite_state = load_suite_state(suite_state_path) or suite_state
                suite_state["runs"][index - 1]["status"] = "completed"
                save_suite_state(suite_state_path, suite_state)
            except KeyboardInterrupt:
                _mark_suite_status(suite_state_path, "interrupted", run_index=index)
                raise
            except Exception:
                _mark_suite_status(suite_state_path, "failed", run_index=index)
                if args.stop_on_error:
                    raise

        suite_state = load_suite_state(suite_state_path) or suite_state
        suite_state["status"] = "completed"
        save_suite_state(suite_state_path, suite_state)
        if is_main_process(distributed_state):
            print(f"Suite complete. Progress saved to {suite_state_path}")
    finally:
        cleanup_distributed(distributed_state)


if __name__ == "__main__":
    main()
