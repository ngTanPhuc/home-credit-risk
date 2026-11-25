"""E7: cross-model explanation agreement (RQ4).

Do LightGBM, XGBoost and CatBoost rank the same features at the top? This module
compares the gain-based feature importance each model saved in E5 (the
per-fold mean ``importance_mean`` column) and reports, for every pair:

* Spearman rank correlation over all 180 features, and
* top-k Jaccard overlap for k in the configured list.

Reading the saved importance files keeps the step cheap and keeps E7 from
refitting models just to answer a ranking question. Gain importance answers "did
the models split on the same features", which is RQ4's question; it is not a
per-applicant attribution.

Usage
-----
``uv run python -m src.explain.agreement --config configs/experiments/E7_explain.yaml``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_nested, load_config, resolve_path
from src.utils.logging import setup_logging, write_json

LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_K = (10, 20, 50, 100)


def load_importance(
    path: str | Path,
    *,
    name: str | None = None,
    value_column: str = "importance_mean",
    feature_column: str = "feature",
) -> pd.Series:
    """Read one importance table into a descending Series indexed by feature."""
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"Importance table not found: {table_path}")
    frame = pd.read_csv(table_path, usecols=[feature_column, value_column])
    if frame[feature_column].duplicated().any():
        raise ValueError(f"{table_path} lists duplicated features")
    series = (
        frame.set_index(feature_column)[value_column].astype("float64").sort_values(ascending=False)
    )
    series.name = str(name or table_path.stem)
    return series


def top_k_jaccard(left: pd.Series, right: pd.Series, k: int) -> float:
    """Overlap of the two top-k feature sets: |A ∩ B| / |A ∪ B|."""
    if k < 1:
        raise ValueError("k must be positive")
    set_left = set(left.index[:k])
    set_right = set(right.index[:k])
    union = set_left | set_right
    if not union:
        return 0.0
    return len(set_left & set_right) / len(union)


def spearman_rank(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation of the two rankings over the union of their features.

    A feature missing from one ranking gets the lowest rank there, so a model
    that ignores a feature is penalised instead of silently dropped.
    """
    index = left.index.union(right.index)
    left_aligned = left.reindex(index).fillna(0.0)
    right_aligned = right.reindex(index).fillna(0.0)
    left_rank = left_aligned.rank(method="average", ascending=True)
    right_rank = right_aligned.rank(method="average", ascending=True)
    if left_rank.std(ddof=0) == 0 or right_rank.std(ddof=0) == 0:
        return float("nan")
    return float(left_rank.corr(right_rank, method="pearson"))


def pairwise_agreement(
    importances: Mapping[str, pd.Series],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Return one row per model pair with Spearman plus the top-k Jaccards."""
    if len(importances) < 2:
        raise ValueError("Agreement needs at least two models")
    ks = sorted({int(k) for k in top_k if int(k) > 0})
    if not ks:
        raise ValueError("top_k must contain at least one positive value")

    rows: list[dict[str, Any]] = []
    for name_a, name_b in combinations(importances, 2):
        left, right = importances[name_a], importances[name_b]
        index = left.index.union(right.index)
        row: dict[str, Any] = {
            "model_a": str(name_a),
            "model_b": str(name_b),
            "n_features": int(index.size),
            "n_shared_features": int(len(left.index.intersection(right.index))),
            "spearman_rank": spearman_rank(left, right),
        }
        for k in ks:
            row[f"jaccard_top{k}"] = top_k_jaccard(left, right, k)
        rows.append(row)
    return pd.DataFrame(rows)


def _output_path(config: Mapping[str, Any], key: str) -> Path:
    rendered = str(resolve_path(config, f"agreement.{key}")).format(
        exp=get_nested(config, "outputs.exp", "E7_explain")
    )
    return Path(rendered)


def configured_importances(config: Mapping[str, Any]) -> dict[str, pd.Series]:
    """Load every ``agreement.importance`` entry (a mapping of arm to CSV path)."""
    declared = get_nested(config, "agreement.importance", {}) or {}
    if not isinstance(declared, Mapping) or len(declared) < 2:
        raise ValueError("agreement.importance must map at least two model names to CSV paths")
    root = Path(str(get_nested(config, "_meta.project_root", ".")))
    importances: dict[str, pd.Series] = {}
    for name, value in declared.items():
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        importances[str(name)] = load_importance(path, name=str(name))
    return importances


def run_agreement(config: Mapping[str, Any], *, force: bool = False) -> dict[str, str]:
    """Build the pairwise agreement table and its record."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    importances = configured_importances(config)
    top_k = get_nested(config, "agreement.top_k", DEFAULT_TOP_K)

    table_path = _output_path(config, "table")
    record_path = _output_path(config, "record")
    if not force:
        existing = [str(path) for path in (table_path, record_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Agreement artifacts exist (use --force): {existing}")

    table = pairwise_agreement(importances, top_k=top_k)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    record = {
        "experiment": get_nested(config, "experiment", "E7_explain"),
        "models": list(importances),
        "n_features": {name: int(series.size) for name, series in importances.items()},
        "top_k": [int(k) for k in top_k],
        "source": "E5 gain importance (importance_mean)",
        "pairs": table.to_dict("records"),
        "artifacts": {"table": str(table_path)},
        "note": (
            "Spearman is computed over the union of features; top-k Jaccard uses each "
            "model's own top-k list. Gain importance is not a SHAP value: it describes "
            "split usage, not per-applicant attribution."
        ),
    }
    write_json(record_path, record)
    LOGGER.info("Agreement table: %s", table_path)
    return {"table": str(table_path), "record": str(record_path)}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare model feature rankings (RQ4).")
    parser.add_argument("--config", required=True, help="E7 explanation YAML config")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = run_agreement(config, force=args.force)
    except Exception:
        setup_logging()
        LOGGER.exception("E7 feature agreement failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("E7 agreement %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
