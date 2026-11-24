"""E6 calibration: cross-fitted Platt and isotonic on the E5 OOF predictions.

Every arm that E5 compared (the three tuned GBDTs and the cross-fitted
stacking / weighted-average ensembles) is recalibrated with each declared
calibrator. A fold's calibrated probability always comes from a calibrator
fitted on the *other* folds' OOF rows, so the Brier/ECE before vs after are
out-of-sample; no model is refitted and the hold-out is never loaded.

The output ``E6_calibration.csv`` is the Brier/ECE/log-loss comparison that
Phase 8 needs to freeze a model and a calibrator before the single hold-out
run. The uncalibrated row (``none``) stays in the table as the reference.

Usage
-----
``uv run python -m src.evaluation.calibration_eval --config configs/experiments/E6_calibration.yaml``

``--sample N`` subsamples N OOF rows per fold for a smoke run; outputs get a
``_sampleN`` suffix.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import (
    CALIBRATORS,
    build_calibrator,
    calibration_by_arm,
    cross_fitted_predictions,
    recommend_calibrator,
)
from src.config import get_nested, load_config, resolve_path
from src.cv.dataset import load_contract
from src.evaluation.model_comparison import (
    load_seed_oof,
    model_seeds,
    seed_arms,
)
from src.utils.logging import setup_logging, write_json

LOGGER = logging.getLogger(__name__)

SCORE_METRICS = ("roc_auc", "ks", "pr_auc", "brier", "ece", "log_loss")
_TABLE_METRICS = SCORE_METRICS


def declared_calibrators(config: Mapping[str, Any]) -> list[str]:
    """Return the calibrators E6 compares, validated against the known set."""
    declared = [str(name) for name in get_nested(config, "calibration.calibrators", CALIBRATORS)]
    unknown = [name for name in declared if name not in CALIBRATORS]
    if unknown:
        raise ValueError(f"Unknown calibrators {unknown}; known: {list(CALIBRATORS)}")
    if "none" not in declared:
        raise ValueError("calibration.calibrators must include 'none' as the reference row")
    return declared


def _with_sample(path, sample: int | None):
    if sample is None:
        return path
    return path.with_name(f"{path.stem}_sample{sample}{path.suffix}")


def _output_path(config: Mapping[str, Any], key: str, sample: int | None):
    """Render one ``calibration.<key>`` path template and apply the sample suffix."""
    rendered = str(resolve_path(config, f"calibration.{key}")).format(
        exp=get_nested(config, "outputs.exp", "E6")
    )
    return _with_sample(Path(rendered), sample)


def _sample_positions(folds: np.ndarray, sample: int | None, *, seed: int) -> np.ndarray:
    """Subsample row positions per fold for a smoke run; every fold stays represented."""
    if sample is None or sample >= len(folds):
        return np.arange(len(folds))
    per_fold = max(1, sample // max(1, np.unique(folds).size))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for fold in np.unique(folds):
        positions = np.flatnonzero(folds == fold)
        take = min(per_fold, positions.size)
        chosen.append(rng.choice(positions, size=take, replace=False))
    selected = np.sort(np.concatenate(chosen))
    LOGGER.warning(
        "Smoke mode: calibrating on %d OOF rows sampled across folds (hold-out untouched)",
        selected.size,
    )
    return selected


def seed_arm_arrays(
    config: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, dict[str, Any]]:
    """Load one seed's OOF, rebuild the ensemble arms and return arrays per arm."""
    contract = load_contract(resolve_path(config, "dataset.contract"))
    identifier_column = str(contract["identifier_column"])
    target_column = str(contract["target_column"])
    fold_column = str(contract["fold_column"])

    model_names = [str(name) for name in get_nested(config, "comparison.models", {}) or {}]
    if not model_names:
        raise ValueError("comparison.models is empty; E6 scores the arms E5 compared")

    # The E5 OOF files are always full-length; ``--sample`` subsamples their rows
    # below instead of looking for sample-suffixed predictions that never exist.
    matrix = load_seed_oof(
        config,
        model_names,
        seed=int(seed),
        sample=None,
        identifier_column=identifier_column,
        target_column=target_column,
        fold_column=fold_column,
    )
    arms = seed_arms(config, matrix, seed=int(seed))
    target = matrix.meta[target_column].to_numpy(dtype="int8")
    all_folds = matrix.meta[fold_column].to_numpy()
    positions = _sample_positions(all_folds, sample, seed=int(seed))

    arrays = {
        name: {"target": target[positions], "y_prob": arm.y_prob[positions]}
        for name, arm in arms.items()
    }
    folds = all_folds[positions]
    info = {
        "n_rows": int(positions.size),
        "n_folds": int(np.unique(folds).size),
        "identifier_column": identifier_column,
        "target_column": target_column,
        "fold_column": fold_column,
        "arms": list(arms),
    }
    return arrays, folds, info


def build_calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed metrics: mean over folds, then mean/std over seeds."""
    required = {
        "arm",
        "calibrator",
        "seed",
        "n_rows",
        "fit_rows_min",
        *(f"{metric}_mean" for metric in _TABLE_METRICS),
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Calibration rows are missing columns {missing}")

    aggregated: list[dict[str, Any]] = []
    for (arm, calibrator), group in rows.groupby(["arm", "calibrator"], sort=False):
        row: dict[str, Any] = {
            "arm": str(arm),
            "calibrator": str(calibrator),
            "n_seeds": int(group["seed"].nunique()),
            "n_rows": int(group["n_rows"].iloc[0]),
            "fit_rows_min": int(group["fit_rows_min"].min()),
        }
        for metric in _TABLE_METRICS:
            values = group[f"{metric}_mean"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        aggregated.append(row)

    table = pd.DataFrame(aggregated)
    reference = (
        table.loc[table["calibrator"] == "none", ["arm", "brier_mean", "ece_mean", "log_loss_mean"]]
        .set_index("arm")
        .rename(columns={"brier_mean": "_b", "ece_mean": "_e", "log_loss_mean": "_l"})
    )
    table = table.join(reference, on="arm")
    table["delta_brier_vs_none"] = table["brier_mean"] - table["_b"]
    table["delta_ece_vs_none"] = table["ece_mean"] - table["_e"]
    table["delta_log_loss_vs_none"] = table["log_loss_mean"] - table["_l"]
    return table.drop(columns=["_b", "_e", "_l"])


def write_reliability_figure(
    arms: Mapping[str, Mapping[str, np.ndarray]],
    folds: np.ndarray,
    choices: Mapping[str, str],
    *,
    seed: int,
    path,
    min_rows: int,
    ece_bins: int,
) -> Any:
    """Plot a reliability curve for every arm, uncalibrated vs its chosen calibrator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(arms)
    columns = min(3, max(1, len(names)))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)

    for axis, name in zip(axes.ravel(), names, strict=False):
        payload = arms[name]
        series = [("none", payload["y_prob"])]
        chosen = str(choices.get(name, "none"))
        if chosen != "none":
            calibrated, _ = cross_fitted_predictions(
                payload["target"],
                payload["y_prob"],
                folds,
                build_calibrator(chosen),
                min_rows=min_rows,
            )
            series.append((chosen, calibrated))

        for label, probabilities in series:
            order = np.argsort(probabilities)
            for chunk in np.array_split(order, ece_bins):
                predicted = float(np.mean(probabilities[chunk]))
                observed = float(np.mean(payload["target"][chunk]))
                axis.plot(predicted, observed, "o", markersize=3.5, label=None)
            axis.plot([], [], label=label)

        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=0.8, color="grey")
        axis.set_title(f"{name}\nchosen: {chosen}")
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Observed default rate")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    for axis in axes.ravel()[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(f"E6 reliability (seed {seed}, development OOF, {ece_bins} equal-count bins)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_calibration(config: Mapping[str, Any], *, sample: int | None = None) -> dict[str, Any]:
    """Score every arm under every calibrator and write the E6 tables and figure."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    seeds = model_seeds(config)
    calibrators = declared_calibrators(config)
    min_rows = int(get_nested(config, "calibration.min_rows", 200))
    ece_bins = int(get_nested(config, "metrics.ece_bins", 15))
    thresholds = tuple(get_nested(config, "metrics.thresholds", (0.5,)))

    rows: list[dict[str, Any]] = []
    figure_inputs: tuple[dict[str, dict[str, np.ndarray]], np.ndarray, int] | None = None
    seed_details: dict[str, Any] = {}

    for seed in seeds:
        arms, folds, info = seed_arm_arrays(config, seed=int(seed), sample=sample)
        scored = calibration_by_arm(
            arms,
            folds,
            calibrators=calibrators,
            min_rows=min_rows,
            thresholds=thresholds,
            ece_bins=ece_bins,
        )
        for row in scored:
            rows.append({**row, "seed": int(seed)})
        if figure_inputs is None:
            figure_inputs = (arms, folds, int(seed))
        seed_details[str(int(seed))] = info
        LOGGER.info("seed=%d: scored %d arm/calibrator pairs", seed, len(scored))

    if figure_inputs is None:
        raise ValueError("No seeds to calibrate; seeds.model is empty")

    frame = pd.DataFrame(rows)
    table = build_calibration_table(frame)
    recommended = recommend_calibrator(table.to_dict("records"))
    choices = {arm: str(row["calibrator"]) for arm, row in recommended.items()}

    table = table.sort_values(["arm", "brier_mean"], kind="stable").reset_index(drop=True)
    table_path = _output_path(config, "table", sample)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    folds_path = _output_path(config, "fold_table", sample)
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folds_path, index=False)

    figure_path = _output_path(config, "figure", sample)
    write_reliability_figure(
        figure_inputs[0],
        figure_inputs[1],
        choices,
        seed=figure_inputs[2],
        path=figure_path,
        min_rows=min_rows,
        ece_bins=ece_bins,
    )

    final_choice = {
        "arm": str(get_nested(config, "final.arm", "")),
        "calibrator": str(get_nested(config, "final.calibrator", "")),
    }
    record = {
        "experiment": get_nested(config, "experiment", "E6_calibration"),
        "seeds": [int(seed) for seed in seeds],
        "calibrators": calibrators,
        "min_rows": min_rows,
        "ece_bins": ece_bins,
        "seeds_detail": seed_details,
        "recommended": {
            arm: {
                "calibrator": row["calibrator"],
                "brier_mean": row["brier_mean"],
                "ece_mean": row["ece_mean"],
                "log_loss_mean": row["log_loss_mean"],
                "roc_auc_mean": row["roc_auc_mean"],
            }
            for arm, row in recommended.items()
        },
        "final_choice": final_choice,
        "final_choice_matches_recommendation": (
            final_choice["calibrator"] == choices.get(final_choice["arm"])
        ),
    }
    record_path = write_json(_output_path(config, "record", sample), record)

    return {
        "table": table_path,
        "fold_table": folds_path,
        "figure": figure_path,
        "record": record_path,
        "recommended": {arm: row["calibrator"] for arm, row in recommended.items()},
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the E6 calibration comparison.")
    parser.add_argument("--config", required=True, help="E6 experiment YAML config")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Smoke run on N OOF rows per fold; outputs get a _sampleN suffix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = run_calibration(config, sample=args.sample)
    except Exception:
        setup_logging()
        LOGGER.exception("E6 calibration failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("E6 %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
