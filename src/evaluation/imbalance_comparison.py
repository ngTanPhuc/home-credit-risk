"""Imbalance-handling comparison (E3) and the E3 table.

Each variant declared in ``configs/experiments/E3_imbalance.yaml`` runs the standard
cross-validation on the same features, folds and seed; only
``imbalance.strategy`` changes. The table is built from the per-fold validation
metrics, with the ``none`` variant as the reference for ``delta_auc_vs_none``.

Usage
-----
``uv run python -m src.evaluation.imbalance_comparison --config configs/experiments/E3_imbalance.yaml``

``--sample N`` writes sample-suffixed outputs, ``--force`` reruns finished variants,
and ``--table-only`` rebuilds the table from the fold CSVs already on disk.
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

_SUMMARY_COLUMNS = (
    "ks",
    "pr_auc",
    "brier",
    "ece",
    "precision",
    "recall",
    "fit_seconds",
)


def variants(config: Mapping[str, Any]) -> list[str]:
    """Return the E3 variant order, refusing a list without the reference variant first."""
    declared = [str(item) for item in get_nested(config, "imbalance.variants", [])]
    if not declared:
        raise ValueError("imbalance.variants is empty; declare the variants in the E3 config")
    if declared[0] != "none":
        raise ValueError("imbalance.variants must start with 'none', the reference variant")
    return declared


def variant_config(config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Return a copy of the E3 config running ``variant`` as the imbalance strategy."""
    if variant not in variants(config):
        raise ValueError(f"Unknown variant {variant!r}; known variants: {variants(config)}")

    cfg = deepcopy(dict(config))
    cfg["imbalance"] = {**dict(get_nested(config, "imbalance", {})), "strategy": variant}
    cfg["variant"] = variant
    return cfg


def _fold_frame(path: Path, n_folds: int) -> pd.DataFrame:
    """Read one variant's fold metrics and require the columns this table needs."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold metrics for this variant: {path}")
    frame = pd.read_csv(path)
    for column in ("fold", "roc_auc", "n_train_fit"):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing the {column!r} column")
    if sorted(frame["fold"].tolist()) != list(range(n_folds)):
        raise ValueError(f"{path} must hold exactly folds 0..{n_folds - 1}")
    return frame.sort_values("fold")


def build_e3_table(
    config: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None = None,
) -> pd.DataFrame:
    """Build the E3 comparison table from the per-variant fold metrics.

    Columns: ``variant``, ``n_features``, ``n_train_fit_mean``, ``auc_fold_<k>``,
    ``auc_mean``, ``auc_std``, ``delta_auc_vs_none`` and the mean of the
    remaining metrics. AUC is rank-based, so the informative movement is in
    Brier, ECE, precision and recall rather than in ``auc_mean``.
    """
    variant_names = variants(config)
    n_folds = int(get_nested(config, "split.n_folds", 5))
    contract = load_contract(resolve_path(config, "dataset.contract"))

    rows: list[dict[str, Any]] = []
    baseline_auc: float | None = None

    for variant in variant_names:
        cfg = variant_config(config, variant)
        frame = _fold_frame(render_output_path(cfg, "folds_csv", seed=seed, sample=sample), n_folds)
        aucs = frame["roc_auc"].astype(float)
        auc_mean = float(aucs.mean())

        row: dict[str, Any] = {
            "variant": variant,
            "n_features": len(resolve_feature_columns(cfg, contract)),
            "n_train_fit_mean": float(frame["n_train_fit"].mean()),
            **{f"auc_fold_{fold}": value for fold, value in enumerate(aucs.tolist())},
            "auc_mean": auc_mean,
            "auc_std": float(aucs.std(ddof=0)),
            "delta_auc_vs_none": None if baseline_auc is None else auc_mean - baseline_auc,
        }
        for column in _SUMMARY_COLUMNS:
            if column not in frame.columns:
                raise ValueError(f"Fold metrics for variant {variant!r} are missing {column!r}")
            row[f"{column}_mean"] = float(frame[column].astype(float).mean())

        rows.append(row)
        if baseline_auc is None:
            baseline_auc = auc_mean

    return pd.DataFrame(rows)


def write_e3_table(
    config: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None = None,
) -> Path:
    """Write ``E3_imbalance_comparison.csv`` (or its sample variant) and return the path."""
    table = build_e3_table(config, seed=seed, sample=sample)
    path = resolve_path(config, "imbalance.table")
    if sample is not None:
        path = path.with_name(f"{path.stem}_sample{sample}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _single_seed(config: Mapping[str, Any]) -> int:
    """Return the one model seed E3 uses; the table cannot merge seeds."""
    seeds = get_nested(config, "seeds.model", [42])
    seeds = [int(seeds)] if isinstance(seeds, int) else [int(seed) for seed in seeds]
    if len(seeds) != 1:
        raise ValueError("E3 expects exactly one model seed")
    return seeds[0]


def run_comparison(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    force: bool = False,
    table_only: bool = False,
) -> Path:
    """Run every variant with missing outputs, then write the E3 table."""
    seed = _single_seed(config)
    if not table_only:
        for variant in variants(config):
            LOGGER.info("E3 variant: %s", variant)
            run_from_config(variant_config(config, variant), sample=sample, force=force)
    return write_e3_table(config, seed=seed, sample=sample)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the imbalance-handling comparison (E3).")
    parser.add_argument("--config", required=True, help="E3 experiment YAML config")
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
        help="Rerun variants whose outputs already exist.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Rebuild the E3 table from the fold CSVs already on disk.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        setup_logging(get_nested(config, "logging.level", "INFO"))
        path = run_comparison(
            config,
            sample=args.sample,
            force=args.force,
            table_only=args.table_only,
        )
    except Exception:
        setup_logging()
        LOGGER.exception("E3 imbalance comparison failed")
        return 1
    LOGGER.info("E3 table: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
