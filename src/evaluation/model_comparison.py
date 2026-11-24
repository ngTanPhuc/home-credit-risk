"""E5 model comparison and ensembling, plus B2 and the pairwise DeLong table.

Every model declared in ``configs/experiments/E5_models.yaml`` runs the standard
cross-validation with the tuned parameters saved by E4 (``final_params``: fixed
iteration count, no early stopping) for every model seed, on the same folds,
features, imbalance variant and metrics. The OOF predictions are then blended
with a cross-fitted weighted average and a cross-fitted logistic stack, and
``B2_model_comparison.csv`` reports mean ± std over seeds.

Usage
-----
``uv run python -m src.evaluation.model_comparison --config configs/experiments/E5_models.yaml``

``--sample N`` writes sample-suffixed outputs, ``--force`` reruns finished
models, ``--models lightgbm,xgboost`` restricts the models, and ``--table-only``
rebuilds the tables and the figure from the OOF files already on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.config import get_nested, load_config, resolve_path
from src.cv.dataset import load_contract
from src.cv.runner import render_output_path, run_from_config
from src.evaluation.delong import delong_test
from src.evaluation.metrics import compute_metrics
from src.models.ensemble import (
    OOFMatrix,
    align_oof,
    cross_fitted_average,
    cross_fitted_stacking,
)
from src.utils.logging import setup_logging, write_json

LOGGER = logging.getLogger(__name__)

SUPPORTED_ENSEMBLES = ("weighted_average", "stacking")
_TABLE_METRICS = ("roc_auc", "ks", "pr_auc", "brier", "ece", "precision", "recall")


@dataclass(frozen=True)
class _Arm:
    """One comparison arm: a base model or an ensemble, with its OOF scores."""

    name: str
    y_prob: np.ndarray
    n_models: int
    weights: Mapping[str, float] | None = None
    fold_weights: Mapping[int, Mapping[str, float]] | None = None


def model_seeds(config: Mapping[str, Any]) -> list[int]:
    """Return the model seeds E5 repeats over."""
    seeds = get_nested(config, "seeds.model", [])
    seeds = [int(seeds)] if isinstance(seeds, int) else [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds.model is empty; E5 needs at least one model seed")
    return seeds


def ensembles(config: Mapping[str, Any]) -> list[str]:
    """Return the declared ensemble arms."""
    declared = [
        str(name) for name in get_nested(config, "comparison.ensembles", list(SUPPORTED_ENSEMBLES))
    ]
    unknown = [name for name in declared if name not in SUPPORTED_ENSEMBLES]
    if unknown:
        raise ValueError(f"Unknown ensembles {unknown}; supported: {list(SUPPORTED_ENSEMBLES)}")
    return declared


def best_params_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the E4 ``final_params`` payload E5 refits from."""
    path = resolve_path(config, "comparison.best_params")
    if not path.is_file():
        raise FileNotFoundError(f"Missing E4 best parameters: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("models"), Mapping) or not payload["models"]:
        raise ValueError(f"{path} has no models block")
    return payload


def model_definitions(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load each ``comparison.models`` entry and apply its E4 ``final_params``."""
    declared = get_nested(config, "comparison.models", {}) or {}
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("comparison.models is empty; declare model name -> YAML path")
    tuned = best_params_payload(config)["models"]
    root = Path(str(get_nested(config, "_meta.project_root", "."))).expanduser().resolve()

    definitions: dict[str, dict[str, Any]] = {}
    for name, value in declared.items():
        if isinstance(value, Mapping):
            model_block = deepcopy(dict(value))
        else:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = root / path
            model_block = dict(load_config(path)["model"])
        model_block.setdefault("name", str(name))
        if str(model_block["name"]).lower() != str(name).lower():
            raise ValueError(
                f"comparison.models[{name!r}] declares model name {model_block['name']!r}"
            )

        saved = tuned.get(str(name))
        if not isinstance(saved, Mapping) or "final_params" not in saved:
            raise KeyError(f"E4 parameters for model {name!r} are missing final_params")
        model_block["params"] = dict(saved["final_params"])
        definitions[str(name)] = model_block
    return definitions


def model_config(
    config: Mapping[str, Any],
    name: str,
    model_block: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the runner config for one model, with the model name as the variant."""
    cfg = deepcopy(dict(config))
    cfg["model"] = {key: value for key, value in model_block.items() if key != "tuning"}
    cfg["variant"] = str(name)
    return cfg


def _with_sample(path: Path, sample: int | None) -> Path:
    if sample is None:
        return path
    return path.with_name(f"{path.stem}_sample{sample}{path.suffix}")


def _record_path(config: Mapping[str, Any], *, sample: int | None) -> Path:
    rendered = str(resolve_path(config, "comparison.record")).format(
        exp=get_nested(config, "outputs.exp", "E5")
    )
    return _with_sample(Path(rendered), sample)


def load_seed_oof(
    config: Mapping[str, Any],
    model_names: Sequence[str],
    *,
    seed: int,
    sample: int | None,
    identifier_column: str,
    target_column: str,
    fold_column: str,
) -> OOFMatrix:
    """Read every model's OOF parquet for one seed and align it."""
    frames: dict[str, pd.DataFrame] = {}
    for name in model_names:
        cfg = model_config(config, name, {"name": name})
        path = render_output_path(cfg, "oof", seed=seed, sample=sample)
        if not path.is_file():
            raise FileNotFoundError(f"Missing OOF predictions for model {name!r}: {path}")
        frames[name] = pd.read_parquet(path)

    return align_oof(
        frames,
        identifier_column=identifier_column,
        target_column=target_column,
        fold_column=fold_column,
    )


def seed_arms(
    config: Mapping[str, Any],
    matrix: OOFMatrix,
    *,
    seed: int,
) -> dict[str, _Arm]:
    """Return every arm's OOF probabilities: base models first, then ensembles."""
    arms = {
        str(name): _Arm(
            name=str(name),
            y_prob=matrix.probabilities[name].to_numpy(dtype="float64"),
            n_models=1,
            weights={str(name): 1.0},
        )
        for name in matrix.probabilities.columns
    }

    for kind in ensembles(config):
        if kind == "weighted_average":
            prediction = cross_fitted_average(
                matrix,
                step=float(get_nested(config, "comparison.weight_step", 0.05)),
            )
        else:
            prediction = cross_fitted_stacking(matrix, seed=seed)
        arms[kind] = _Arm(
            name=kind,
            y_prob=prediction.y_prob,
            n_models=int(matrix.probabilities.shape[1]),
            weights=prediction.weights,
            fold_weights=prediction.fold_weights,
        )
    return arms


def _fold_rows(
    config: Mapping[str, Any],
    matrix: OOFMatrix,
    arms: Mapping[str, _Arm],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Per-arm, per-fold metrics for one seed."""
    thresholds = tuple(get_nested(config, "metrics.thresholds", (0.5,)))
    ece_bins = int(get_nested(config, "metrics.ece_bins", 15))
    y = matrix.meta[matrix.target_column].to_numpy()
    folds = matrix.meta[matrix.fold_column].to_numpy()

    rows: list[dict[str, Any]] = []
    for name, arm in arms.items():
        for fold in np.unique(folds):
            mask = folds == fold
            metrics = compute_metrics(
                y[mask], arm.y_prob[mask], thresholds=thresholds, ece_bins=ece_bins
            )
            rows.append(
                {
                    "arm": name,
                    "n_models": arm.n_models,
                    "seed": int(seed),
                    "fold": int(fold),
                    **{metric: metrics[metric] for metric in _TABLE_METRICS},
                }
            )
    return rows


def build_b2_table(fold_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics: mean over folds per seed, then mean/std over seeds."""
    for column in ("arm", "seed", "fold", *_TABLE_METRICS):
        if column not in fold_frame.columns:
            raise ValueError(f"Fold metrics are missing the {column!r} column")

    rows: list[dict[str, Any]] = []
    for arm, group in fold_frame.groupby("arm", sort=False):
        per_seed = group.groupby("seed")[list(_TABLE_METRICS)].mean()
        row: dict[str, Any] = {
            "arm": str(arm),
            "n_models": int(group["n_models"].iloc[0]),
            "n_seeds": int(per_seed.shape[0]),
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in _TABLE_METRICS:
            row[f"{metric}_mean"] = float(per_seed[metric].mean())
            row[f"{metric}_std"] = float(per_seed[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_delong_table(
    matrix: OOFMatrix,
    arms: Mapping[str, _Arm],
    *,
    seed: int,
) -> pd.DataFrame:
    """Pairwise DeLong tests across every arm on one seed's pooled OOF rows."""
    y = matrix.meta[matrix.target_column].to_numpy()
    names = list(arms)
    rows: list[dict[str, Any]] = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            result = delong_test(y, arms[name_a].y_prob, arms[name_b].y_prob)
            rows.append(
                {
                    "seed": int(seed),
                    "model_a": name_a,
                    "model_b": name_b,
                    "auc_a": result.auc_a,
                    "auc_b": result.auc_b,
                    "delta_auc": result.difference,
                    "z": result.z,
                    "p_value": result.p_value,
                    "significant_at_0_05": bool(result.p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def write_e5_figure(
    matrix: OOFMatrix,
    arms: Mapping[str, _Arm],
    *,
    seed: int,
    path: Path,
) -> Path:
    """Plot ROC and precision/recall curves for every arm on one seed's OOF rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = matrix.meta[matrix.target_column].to_numpy()
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for name, arm in arms.items():
        false_positive, true_positive, _ = roc_curve(y, arm.y_prob)
        auc = float(roc_auc_score(y, arm.y_prob))
        roc_axis.plot(false_positive, true_positive, label=f"{name} (AUC {auc:.4f})")
        precision, recall, _ = precision_recall_curve(y, arm.y_prob)
        pr_axis.plot(recall, precision, label=name)

    roc_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=0.8, color="grey")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.set_title(f"E5 ROC (seed {seed}, development OOF)")
    roc_axis.legend(fontsize=8)
    roc_axis.grid(alpha=0.3)

    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title(f"E5 precision/recall (seed {seed})")
    pr_axis.legend(fontsize=8)
    pr_axis.grid(alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_outputs(
    config: Mapping[str, Any],
    *,
    model_names: Sequence[str],
    seeds: Sequence[int],
    sample: int | None = None,
) -> dict[str, Path]:
    """Build B2, the DeLong pairs, the figure and the ensemble record from OOF files."""
    contract = load_contract(resolve_path(config, "dataset.contract"))
    identifier_column = str(contract["identifier_column"])
    target_column = str(contract["target_column"])
    fold_column = str(contract["fold_column"])

    fold_rows: list[dict[str, Any]] = []
    delong_rows: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "experiment": get_nested(config, "experiment", "E5_models"),
        "seeds": [int(seed) for seed in seeds],
        "models": [str(name) for name in model_names],
        "ensembles": ensembles(config),
        "weight_step": float(get_nested(config, "comparison.weight_step", 0.05)),
        "seeds_detail": {},
    }
    figure_inputs: tuple[OOFMatrix, dict[str, _Arm], int] | None = None

    for seed in seeds:
        matrix = load_seed_oof(
            config,
            model_names,
            seed=int(seed),
            sample=sample,
            identifier_column=identifier_column,
            target_column=target_column,
            fold_column=fold_column,
        )
        arms = seed_arms(config, matrix, seed=int(seed))
        fold_rows.extend(_fold_rows(config, matrix, arms, seed=int(seed)))
        delong_rows.extend(build_delong_table(matrix, arms, seed=int(seed)).to_dict("records"))
        average_arm = arms.get("weighted_average")
        record["seeds_detail"][str(int(seed))] = {
            "average_weights_all_rows": (
                None if average_arm is None else dict(average_arm.weights or {})
            ),
            "average_weights_by_fold": (
                None
                if average_arm is None
                else {
                    str(fold): dict(weights)
                    for fold, weights in (average_arm.fold_weights or {}).items()
                }
            ),
        }
        if figure_inputs is None:
            figure_inputs = (matrix, arms, int(seed))

    if figure_inputs is None:
        raise ValueError("No seeds to compare; seeds.model is empty")

    fold_frame = pd.DataFrame(fold_rows)
    table = build_b2_table(fold_frame)

    table_path = _with_sample(resolve_path(config, "comparison.table"), sample)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    delong_path = _with_sample(resolve_path(config, "comparison.delong_table"), sample)
    delong_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(delong_rows).to_csv(delong_path, index=False)

    figure_path = _with_sample(resolve_path(config, "comparison.figure"), sample)
    write_e5_figure(figure_inputs[0], figure_inputs[1], seed=figure_inputs[2], path=figure_path)

    record_path = write_json(_record_path(config, sample=sample), record)
    return {
        "table": table_path,
        "delong_table": delong_path,
        "figure": figure_path,
        "record": record_path,
    }


def run_comparison(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    force: bool = False,
    table_only: bool = False,
    models: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Run E5: tuned CV per model and seed, then the B2 tables and figure."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    seeds = model_seeds(config)
    definitions = model_definitions(config)
    selected = [str(name) for name in (models or definitions)]
    unknown = [name for name in selected if name not in definitions]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; known: {sorted(definitions)}")

    if not table_only:
        for name in selected:
            LOGGER.info("E5 model: %s (seeds=%s)", name, seeds)
            run_from_config(
                model_config(config, name, definitions[name]), sample=sample, force=force
            )

    return write_outputs(config, model_names=selected, seeds=seeds, sample=sample)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the E5 model comparison and ensemble.")
    parser.add_argument("--config", required=True, help="E5 experiment YAML config")
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
        help="Rerun models whose outputs already exist.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to run (default: every declared model).",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Rebuild B2, DeLong pairs and the figure from the OOF files on disk.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        selected = args.models.split(",") if args.models else None
        paths = run_comparison(
            config,
            sample=args.sample,
            force=args.force,
            table_only=args.table_only,
            models=selected,
        )
    except Exception:
        setup_logging()
        LOGGER.exception("E5 model comparison failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("E5 %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
