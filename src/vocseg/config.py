"""YAML config loading with recursive inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_single_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a mapping.")

    base_ref = data.pop("_base_", None)
    if not base_ref:
        return data

    if isinstance(base_ref, str):
        base_paths = [base_ref]
    elif isinstance(base_ref, list):
        base_paths = base_ref
    else:
        raise TypeError(f"_base_ in {path} must be a string or list.")

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = (path.parent / base_item).resolve()
        merged = _deep_merge(merged, _load_single_config(base_path))

    return _deep_merge(merged, data)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_single_config(config_path)
    config["_config_path"] = str(config_path)
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
