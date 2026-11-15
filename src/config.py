"""Configuration loading and shared runtime helpers.

Configs are YAML files (stored in folder ./configs). A config may inherit from one or more other configs via an ``extends`` key (a string or a list of strings). Paths are resolved relative to the repository root, which is located by walking up to ``pyproject.toml``.

Example
-------
``configs/experiments/E1_baseline.yaml``:

    extends:
      - ../base.yaml
      - ../models/lightgbm.yaml

    experiment: E1_baseline

The values from the extended files are merged first, then the child values override them. Nested mappings are merged recursively; lists are replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PathLike = str | os.PathLike[str]

_MISSING = object()


def find_project_root(start: PathLike | None = None) -> Path:
    """Return the nearest ancestor directory containing ``pyproject.toml``.

    Parameters
    ----------
    start:
        File or directory to start from. Defaults to the location of this module.

    Raises
    ------
    FileNotFoundError
        If no ancestor contains ``pyproject.toml``.
    """
    current = Path(start or __file__).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find a pyproject.toml above {current!s}; pass project_root explicitly."
    )


def _expand_env(value: Any) -> Any:
    """Recursively expand ``~`` and environment variables in string values."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings, returning a new dictionary."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], MutableMapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping (an empty file becomes ``{}``)."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return payload


def _load_config_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    """Load one config file and recursively merge its ``extends`` files."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Circular config inheritance detected: {chain}")

    payload = _load_yaml(resolved)
    extends = payload.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise TypeError(f"'extends' must be a string or list in {resolved}")

    merged: dict[str, Any] = {}
    for parent in extends:
        parent_path = Path(_expand_env(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        merged = _deep_merge(merged, _load_config_file(parent_path, (*stack, resolved)))

    return _deep_merge(merged, _expand_env(payload))


def load_config(path: PathLike, *, project_root: PathLike | None = None) -> dict[str, Any]:
    """Load a YAML config, resolve inheritance, and attach metadata.

    Parameters
    ----------
    path:
        Config file to load.
    project_root:
        Optional repository root. If omitted, it is discovered from ``path``.

    Returns
    -------
    dict
        Merged config with a ``_meta`` section containing ``config_path`` and
        ``project_root``. The ``_meta`` section is informational only and is
        excluded from :func:`config_digest`.
    """
    config_path = Path(path).expanduser().resolve()
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        root = find_project_root(config_path.parent)

    config = _load_config_file(config_path, stack=())
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(root),
    }
    return config


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = _MISSING,
) -> Any:
    """Return a nested value using a dotted key, e.g. ``paths.splits``.

    Raises
    ------
    KeyError
        If the key is missing and no default was supplied.
    """
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is not _MISSING:
                return default
            raise KeyError(f"Missing config key: {dotted_key}")
        current = current[part]
    return current


def resolve_path(
    config: Mapping[str, Any],
    dotted_key: str,
    *,
    default: Any = _MISSING,
) -> Path:
    """Resolve a path-valued config entry against the repository root.

    Absolute paths are returned unchanged. Relative paths are anchored at
    ``_meta.project_root`` (or the config file's project root when metadata is
    absent). This keeps configs portable between a local clone and Kaggle.
    """
    value = get_nested(config, dotted_key, default)
    if value is _MISSING:
        raise KeyError(f"Missing path config key: {dotted_key}")
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"Expected a path string at {dotted_key}, got {type(value)!r}")

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    meta = config.get("_meta", {})
    root_value = meta.get("project_root")
    if root_value:
        root = Path(root_value).expanduser().resolve()
    else:
        root = find_project_root(config["_meta"]["config_path"])

    return (root / path).resolve()


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the config without runtime ``_meta`` metadata."""
    payload = deepcopy(dict(config))
    payload.pop("_meta", None)
    return payload


def config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of the semantic config content."""
    serialized = json.dumps(public_config(config), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def set_seeds(seed: int) -> int:
    """Seed Python, NumPy, and the hash seed for reproducibility.

    Note that ``PYTHONHASHSEED`` only fully affects processes started after it
    is set; scripts still set it so child processes inherit the same value.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
