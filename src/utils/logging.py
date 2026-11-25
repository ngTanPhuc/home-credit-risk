"""Logging and MLflow helpers.

MLflow is optional at runtime: if tracking fails, callers can still persist a
JSON run record via :func:`write_json`. This module never writes to git.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.config import get_nested, public_config, resolve_path


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Configure root logging once with a readable console formatter."""
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level!r}")

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)

    root.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)


def git_commit(repo_root: str | Path) -> str:
    """Read the current git commit hash without changing repository state."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON document, creating parent directories as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return output


def resolve_tracking_uri(config: Mapping[str, Any]) -> str:
    """Resolve the MLflow tracking URI from config.

    Plain paths (for example ``mlruns``) are anchored at the project root.
    URIs such as ``sqlite:///...`` or ``http://...`` are passed through.
    """
    value = str(get_nested(config, "mlflow.tracking_uri", None) or "mlruns")
    if "://" in value or value.startswith("file:"):
        return value
    return str(resolve_path(config, "mlflow.tracking_uri", default="mlruns"))


def start_mlflow_run(
    config: Mapping[str, Any],
    experiment_name: str,
    run_name: str,
    *,
    tags: Mapping[str, Any] | None = None,
):
    """Configure MLflow and return an active run context manager.

    The caller is responsible for entering/exiting the returned context. If
    MLflow is unavailable, the import error propagates so callers can fall back
    to JSON logging.
    """
    # MLflow 3.x requires an explicit opt-in for the filesystem backend.
    # INIT.md specifies ./mlruns, so opt in unless the environment says otherwise.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    import mlflow

    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    mlflow.set_experiment(experiment_name)
    clean_tags = {str(key): str(value) for key, value in (tags or {}).items()}
    return mlflow.start_run(run_name=run_name, tags=clean_tags)


def log_config(config: Mapping[str, Any]) -> None:
    """Log the resolved config as a JSON artifact in the active MLflow run."""
    import mlflow

    mlflow.log_dict(public_config(config), "config.json")


def log_params(params: Mapping[str, Any]) -> None:
    """Log flat parameters, stringifying non-scalar values."""
    import mlflow

    clean: dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[str(key)] = str(value)
        else:
            clean[str(key)] = json.dumps(value, default=_json_default, sort_keys=True)
    mlflow.log_params(clean)


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    """Log numeric metrics, silently skipping non-numeric values."""
    import mlflow

    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            clean[str(key)] = float(value)

    if clean:
        if step is None:
            mlflow.log_metrics(clean)
        else:
            mlflow.log_metrics(clean, step=int(step))


def log_artifacts(paths: Iterable[str | Path]) -> None:
    """Log existing files as MLflow artifacts."""
    import mlflow

    for path in paths:
        artifact = Path(path)
        if artifact.exists():
            mlflow.log_artifact(str(artifact))
