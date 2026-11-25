"""Cumulative feature-group ablation (E2) and the B1 table.

Each feature set of the ladder declared in ``configs/experiments/E2_ablation.yaml``
runs the standard cross-validation with the same folds, seed and LightGBM
defaults as E1; only ``features.use_set`` changes. The B1 table is then built
from the per-fold validation AUCs.

Usage
-----
``uv run python -m src.evaluation.ablation --config configs/experiments/E2_ablation.yaml``

``--sample N`` writes sample-suffixed outputs, ``--force`` reruns finished
feature sets, and ``--table-only`` rebuilds B1 from the fold CSVs already on disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_nested, load_config, resolve_path
from src.cv.dataset import load_contract, resolve_feature_columns
from src.cv.runner import render_output_path, run_from_config
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def feature_set_config(config: Mapping[str, Any], feature_set: str) -> dict[str, Any]:
    """Return a copy of the E2 config selecting ``feature_set`` as the feature set."""
    sets_cfg = get_nested(config, "feature_sets", {}) or {}
    if feature_set not in sets_cfg:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; known feature sets: {sorted(sets_cfg)}"
        )

    cfg = deepcopy(dict(config))
    cfg["features"] = {**cfg.get("features", {}), "use_set": feature_set}
    cfg["variant"] = feature_set
    return cfg


def _fold_aucs(path: Path, n_folds: int) -> list[float]:
    """Read validation AUCs from one feature set's fold metrics, ordered by fold id."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics for this feature set: {path}")
    frame = pd.read_csv(path)
    for column in ("fold", "roc_auc"):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing the {column!r} column")
    if sorted(frame["fold"].tolist()) != list(range(n_folds)):
        raise ValueError(f"{path} must hold exactly folds 0..{n_folds - 1}")
    return frame.sort_values("fold")["roc_auc"].astype(float).tolist()


def build_b1_table(
    config: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None = None,
) -> pd.DataFrame:
    """Build the B1 ablation table from the per-feature-set fold metrics.

    Columns: ``set``, ``n_features``, ``n_features_added``, ``group_added``,
    ``auc_fold_<k>``, ``auc_mean``, ``auc_std``, ``delta_auc_vs_previous`` and
    ``gain_per_group_auc``. The gain is the cumulative AUC delta attributed to the
    group entering at that feature set, not a leave-one-group-out marginal; the
    first feature set leaves the delta and the gain blank.
    """
    feature_sets = [str(name) for name in get_nested(config, "ablation.sets", [])]
    if not feature_sets:
        raise ValueError(
            "ablation.sets is empty; declare the ordered feature sets in the E2 config"
        )

    sets_cfg = get_nested(config, "feature_sets", {}) or {}
    n_folds = int(get_nested(config, "split.n_folds", 5))
    contract = load_contract(resolve_path(config, "dataset.contract"))

    rows: list[dict[str, Any]] = []
    previous_mean: float | None = None
    previous_features = 0
    previous_groups: set[str] = set()

    for feature_set in feature_sets:
        cfg = feature_set_config(config, feature_set)
        aucs = _fold_aucs(render_output_path(cfg, "folds_csv", seed=seed, sample=sample), n_folds)
        n_features = len(resolve_feature_columns(cfg, contract))

        groups = {str(group) for group in sets_cfg[feature_set]}
        added = sorted(groups - previous_groups)
        if rows and not added:
            raise ValueError(
                f"Feature set {feature_set!r} adds no feature group; check ablation.sets order"
            )
        if rows and n_features <= previous_features:
            raise ValueError(
                f"Feature set {feature_set!r} adds no features; check ablation.sets order"
            )

        mean = float(pd.Series(aucs).mean())
        std = float(pd.Series(aucs).std(ddof=0))
        delta = None if previous_mean is None else mean - previous_mean
        gain = None if delta is None else delta / len(added)

        rows.append(
            {
                "set": feature_set,
                "n_features": n_features,
                "n_features_added": n_features - previous_features,
                "group_added": "+".join(added),
                **{f"auc_fold_{fold}": value for fold, value in enumerate(aucs)},
                "auc_mean": mean,
                "auc_std": std,
                "delta_auc_vs_previous": delta,
                "gain_per_group_auc": gain,
            }
        )
        previous_mean = mean
        previous_features = n_features
        previous_groups = groups

    return pd.DataFrame(rows)


def write_b1_table(
    config: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None = None,
) -> Path:
    """Write ``B1_ablation.csv`` (or its sample variant) and return the path."""
    table = build_b1_table(config, seed=seed, sample=sample)

    path = resolve_path(config, "ablation.table")
    if sample is not None:
        path = path.with_name(f"{path.stem}_sample{sample}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _single_seed(config: Mapping[str, Any]) -> int:
    """Return the one model seed E2 uses; B1 cannot merge seeds."""
    seeds = get_nested(config, "seeds.model", [42])
    seeds = [int(seeds)] if isinstance(seeds, int) else [int(seed) for seed in seeds]
    if len(seeds) != 1:
        raise ValueError("E2 expects exactly one model seed")
    return seeds[0]


def run_ablation(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    force: bool = False,
    table_only: bool = False,
) -> Path:
    """Run every feature set with missing outputs, then write B1."""
    seed = _single_seed(config)
    if not table_only:
        for feature_set in [str(item) for item in get_nested(config, "ablation.sets", [])]:
            LOGGER.info("E2 feature set: %s", feature_set)
            run_from_config(feature_set_config(config, feature_set), sample=sample, force=force)
    return write_b1_table(config, seed=seed, sample=sample)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cumulative feature ablation (E2).")
    parser.add_argument("--config", required=True, help="E2 experiment YAML config")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Smoke run on N development rows; outputs get a _sampleN suffix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun feature sets whose outputs already exist.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Rebuild B1 from the fold CSVs already on disk.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        setup_logging(get_nested(config, "logging.level", "INFO"))
        path = run_ablation(
            config,
            sample=args.sample,
            force=args.force,
            table_only=args.table_only,
        )
    except Exception:
        setup_logging()
        LOGGER.exception("E2 ablation failed")
        return 1
    LOGGER.info("B1 table: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
