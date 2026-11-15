"""Cross-validation runner for Person B experiments.

The runner is deliberately strict about the shared protocol:

* it reads the frozen dataset (``data/processed/datasets/train.parquet``) and never re-splits;
* fold membership comes from the ``fold`` column, which mirrors ``data/splits/train_val_metadata.json``;
* the hold-out split is never loaded here, so it cannot enter ``fit``;
* each fold's model is fitted only on that fold's training rows;
* OOF predictions are produced only for development rows.

Usage
-----
``uv run python -m src.cv.runner --config configs/experiments/E1_baseline.yaml``

Add ``--sample N`` for a quick smoke run (writes sample-suffixed outputs) and
``--force`` to overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    config_digest,
    get_nested,
    load_config,
    public_config,
    resolve_path,
    set_seeds,
)
from src.cv.dataset import (
    dataset_path,
    load_contract,
    load_split_frame,
    resolve_feature_columns,
)
from src.cv.splits import (
    load_split_metadata,
    parquet_row_count,
    validate_split_metadata,
)
from src.evaluation.metrics import compute_metrics
from src.models.base import BaseModel
from src.utils.logging import (
    git_commit,
    log_artifacts,
    log_config,
    log_metrics,
    log_params,
    setup_logging,
    start_mlflow_run,
    write_json,
)
from src.utils.memory import frame_memory_mb, peak_rss_mb

LOGGER = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Output of one seed's cross-validation run."""

    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict[str, float]
    feature_importance: pd.DataFrame | None = None
    feature_names: list[str] = field(default_factory=list)


def _prepare_development(
    train_frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    id_col: str,
    target_col: str,
    sample: int | None = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """Select predictors and optionally subsample, keeping fold membership.

    ``train_frame`` comes from ``train.parquet``, which holds development rows
    only, so the hold-out cannot enter this frame.
    """
    required = [id_col, target_col, "fold"]
    missing = [column for column in required if column not in train_frame.columns]
    if missing:
        raise ValueError(f"train.parquet is missing required columns: {missing}")

    development = train_frame.loc[:, [*required, *columns]].copy()
    if sample is not None:
        development = _sample_development(development, sample, sample_seed, id_col=id_col)

    if development["fold"].isna().any():
        raise ValueError("Development frame has missing fold values")
    if development.empty:
        raise ValueError("Development frame is empty")
    return development.sort_values(id_col).reset_index(drop=True)


def _sample_development(
    development: pd.DataFrame,
    sample: int,
    seed: int,
    *,
    id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Sample rows per fold for a smoke run, keeping every fold represented."""
    if sample < 2:
        raise ValueError("--sample must be at least 2")
    if sample >= len(development):
        return development

    folds = sorted(development["fold"].unique().tolist())
    per_fold = max(1, sample // len(folds))
    selected: list[pd.DataFrame] = []

    for fold_id in folds:
        fold_rows = development.loc[development["fold"] == fold_id]
        take = min(per_fold, len(fold_rows))
        selected.append(fold_rows.sample(n=take, random_state=seed + int(fold_id)))

    chosen = pd.concat(selected, ignore_index=True)
    if len(chosen) > sample:
        chosen = chosen.sample(n=sample, random_state=seed)

    remaining = sample - len(chosen)
    if remaining > 0:
        pool = development.loc[~development[id_col].isin(chosen[id_col])]
        extra = pool.sample(n=min(remaining, len(pool)), random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=True)

    LOGGER.warning(
        "Smoke mode: using %d development rows sampled across folds (hold-out untouched)",
        len(chosen),
    )
    return chosen


def _resolve_categorical_features(
    features: pd.DataFrame,
    specification: Any,
) -> list[str]:
    """Resolve the ``input.categorical_features`` config value."""
    if specification is None or specification is False:
        return []
    if isinstance(specification, str):
        normalized = specification.strip().lower()
        if normalized in {"none", "false", ""}:
            return []
        if normalized == "auto":
            return list(features.select_dtypes(include=["object", "string", "category"]).columns)
        raise ValueError(
            "input.categorical_features must be 'auto', 'none', a list of column names, "
            f"or null; got {specification!r}"
        )
    if isinstance(specification, Sequence):
        names = [str(name) for name in specification]
        missing = [name for name in names if name not in features.columns]
        if missing:
            raise KeyError(f"Categorical columns not found: {missing}")
        return names
    raise TypeError(f"Unsupported categorical_features specification: {specification!r}")


def run_cv(
    development: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    model_factory: Callable[[int], BaseModel],
    seed: int,
    n_folds: int,
    categorical_features: Any = "auto",
    metrics_config: Mapping[str, Any] | None = None,
) -> CVResult:
    """Run one seed of stratified 5-fold CV on development rows only.

    Parameters
    ----------
    development:
        Output of :func:`_prepare_development`; must contain ``fold`` and no
        hold-out rows.
    model_factory:
        Callable returning a fresh model for the given seed.
    metrics_config:
        ``thresholds`` and ``ece_bins`` settings.

    Notes
    -----
    The hold-out is not present in ``development``, so it cannot enter ``fit``.
    """
    if "fold" not in development.columns:
        raise ValueError("development frame must contain a 'fold' column")
    if development.empty:
        raise ValueError("development frame is empty")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    drop_columns = [id_col, target_col, "fold"]
    features = development.drop(columns=drop_columns)
    target = development[target_col].to_numpy(dtype="int8")
    fold_ids = development["fold"].to_numpy(dtype="int8")
    ids = development[id_col].to_numpy()

    if target_col in features.columns:
        raise AssertionError("TARGET leaked into the feature matrix")

    categorical = _resolve_categorical_features(features, categorical_features)
    metric_settings = dict(metrics_config or {})
    thresholds = tuple(metric_settings.get("thresholds", (0.5,)))
    ece_bins = int(metric_settings.get("ece_bins", 15))

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    importance_parts: list[pd.Series] = []
    model_name = "model"

    for fold_id in range(n_folds):
        train_mask = fold_ids != fold_id
        validation_mask = fold_ids == fold_id
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Fold {fold_id} has no training or validation rows")

        train_ids = set(ids[train_mask].tolist())
        validation_ids = set(ids[validation_mask].tolist())
        if train_ids & validation_ids:
            raise RuntimeError(f"Train/validation overlap in fold {fold_id}")

        model = model_factory(seed)
        model_name = model.name

        fit_start = time.perf_counter()
        model.fit(
            features.loc[train_mask],
            target[train_mask],
            categorical_features=categorical,
        )
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        probabilities = model.predict_proba(features.loc[validation_mask])
        predict_seconds = time.perf_counter() - predict_start

        metrics = compute_metrics(
            target[validation_mask],
            probabilities,
            thresholds=thresholds,
            ece_bins=ece_bins,
        )
        row: dict[str, Any] = {
            "model": model_name,
            "seed": int(seed),
            "fold": int(fold_id),
            "n_train": int(train_mask.sum()),
            "n_valid": int(validation_mask.sum()),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        }
        row.update(metrics)
        fold_rows.append(row)

        oof_parts.append(
            pd.DataFrame(
                {
                    id_col: ids[validation_mask],
                    target_col: target[validation_mask],
                    "fold": int(fold_id),
                    "seed": int(seed),
                    "model": model_name,
                    "y_prob": np.asarray(probabilities, dtype="float64"),
                }
            )
        )

        importance = model.feature_importance()
        if not importance.empty:
            importance_parts.append(importance.rename(f"fold_{fold_id}"))

        LOGGER.info(
            "seed=%d fold=%d: AUC=%.5f KS=%.5f Brier=%.5f (train=%d, valid=%d)",
            seed,
            fold_id,
            metrics["roc_auc"],
            metrics["ks"],
            metrics["brier"],
            int(train_mask.sum()),
            int(validation_mask.sum()),
        )

    oof = pd.concat(oof_parts, ignore_index=True).sort_values([id_col, "fold"])
    fold_metrics = pd.DataFrame(fold_rows)
    summary = _summarize_folds(fold_metrics)

    importance_frame: pd.DataFrame | None = None
    if importance_parts:
        importance_frame = (
            pd.concat(importance_parts, axis=1)
            .mean(axis=1, skipna=True)
            .sort_values(ascending=False)
            .rename("importance_mean")
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    return CVResult(
        oof=oof,
        fold_metrics=fold_metrics,
        summary=summary,
        feature_importance=importance_frame,
        feature_names=list(features.columns),
    )


def _summarize_folds(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """Aggregate numeric fold metrics into mean/std summary values."""
    ignored = {
        "model",
        "seed",
        "fold",
        "n_train",
        "n_valid",
    }
    summary: dict[str, float] = {
        "seed": float(fold_metrics["seed"].iloc[0]),
        "n_folds": float(len(fold_metrics)),
    }
    for column in fold_metrics.columns:
        if column in ignored:
            continue
        values = pd.to_numeric(fold_metrics[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_std"] = float(values.std(ddof=0))
    return summary


def _render_output_path(
    config: Mapping[str, Any],
    key: str,
    *,
    seed: int,
    sample: int | None,
) -> Path:
    path = resolve_path(config, f"outputs.{key}")
    rendered = Path(str(path).format(seed=seed))
    if sample is not None:
        rendered = rendered.with_name(f"{rendered.stem}_sample{sample}{rendered.suffix}")
    return rendered


def _model_factory(config: Mapping[str, Any], *, seed: int) -> BaseModel:
    model_config = get_nested(config, "model")
    if not isinstance(model_config, Mapping):
        raise TypeError("model must be a mapping")
    model_name = str(model_config.get("name", "lightgbm")).lower()
    if model_name != "lightgbm":
        raise ValueError(f"Unsupported model for Phase 2: {model_name!r}")

    from src.models.lgbm import LGBMModel

    runtime = dict(model_config.get("runtime") or {})
    params = dict(model_config.get("params") or {})
    return LGBMModel(
        params=params,
        seed=seed,
        n_jobs=int(runtime.get("n_jobs", -1)),
        deterministic=bool(runtime.get("deterministic", True)),
        force_col_wise=bool(runtime.get("force_col_wise", True)),
    )


def _model_params(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    model_config = get_nested(config, "model")
    params = dict(model_config.get("params") or {})
    runtime = dict(model_config.get("runtime") or {})
    flat: dict[str, Any] = {"seed": seed, "model.name": model_config.get("name")}
    for key, value in {**runtime, **params}.items():
        flat[f"model.{key}"] = value
    return flat


def _save_outputs(
    config: Mapping[str, Any],
    *,
    result: CVResult,
    seed: int,
    sample: int | None,
    n_holdout: int,
    elapsed_seconds: float,
) -> dict[str, Path]:
    paths = {
        "folds": _render_output_path(config, "folds_csv", seed=seed, sample=sample),
        "summary": _render_output_path(config, "summary_csv", seed=seed, sample=sample),
        "oof": _render_output_path(config, "oof", seed=seed, sample=sample),
        "run_json": _render_output_path(config, "run_json", seed=seed, sample=sample),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    result.fold_metrics.to_csv(paths["folds"], index=False)
    pd.DataFrame([{"seed": seed, **result.summary}]).to_csv(paths["summary"], index=False)
    result.oof.to_parquet(paths["oof"], index=False, engine="pyarrow")

    if result.feature_importance is not None:
        importance_path = _render_output_path(config, "importance_csv", seed=seed, sample=sample)
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        result.feature_importance.to_csv(importance_path, index=False)
        paths["importance"] = importance_path

    git_hash = git_commit(get_nested(config, "_meta.project_root", "."))
    run_payload = {
        "experiment": get_nested(config, "experiment", "unknown"),
        "variant": get_nested(config, "variant", "unknown"),
        "model": result.oof["model"].iloc[0] if not result.oof.empty else "unknown",
        "seed": int(seed),
        "git_commit": git_hash,
        "data_version": get_nested(config, "data_version", "unknown"),
        "config_digest": config_digest(config),
        "config": public_config(config),
        "summary": result.summary,
        "fold_metrics": result.fold_metrics.to_dict(orient="records"),
        "n_features": len(result.feature_names),
        "feature_names": result.feature_names,
        "n_development_rows": int(len(result.oof)),
        "n_holdout_rows": int(n_holdout),
        "elapsed_seconds": float(elapsed_seconds),
        "peak_memory_mb": float(peak_rss_mb()),
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["run_json"], run_payload)
    return paths


def _log_to_mlflow(
    config: Mapping[str, Any],
    *,
    result: CVResult,
    seed: int,
    paths: Mapping[str, Path],
) -> None:
    experiment = str(get_nested(config, "mlflow.experiment", "dev"))
    model_name = str(result.oof["model"].iloc[0])
    variant = str(get_nested(config, "variant", "default"))
    run_name = f"{model_name}_{variant}_seed{seed}"
    tags = {
        "git_commit": git_commit(get_nested(config, "_meta.project_root", ".")),
        "data_version": get_nested(config, "data_version", "unknown"),
        "experiment": get_nested(config, "experiment", experiment),
    }

    with start_mlflow_run(config, experiment, run_name, tags=tags):
        log_config(config)
        log_params(_model_params(config, seed=seed))
        log_params(
            {
                "seed": seed,
                "n_folds": result.summary.get("n_folds"),
                "n_features": len(result.feature_names),
                "n_development_rows": len(result.oof),
            }
        )
        log_metrics(result.summary)
        for _, row in result.fold_metrics.iterrows():
            fold_metrics = {
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in {"seed", "fold"}
            }
            log_metrics(fold_metrics, step=int(row["fold"]))
        log_artifacts(paths.values())


def run_from_config(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    force: bool = False,
) -> int:
    """Execute one or more seeds from a loaded config."""
    setup_logging(get_nested(config, "logging.level", "INFO"))

    n_folds = int(get_nested(config, "split.n_folds", 5))
    split_seed = int(get_nested(config, "seeds.split", 42))
    seeds_value = get_nested(config, "seeds.model", [42])
    seeds = [int(seeds_value)] if isinstance(seeds_value, int) else [int(s) for s in seeds_value]

    metrics_config = dict(get_nested(config, "metrics", {}) or {})

    contract = load_contract(resolve_path(config, "dataset.contract"))
    id_col = str(contract["identifier_column"])
    target_col = str(contract["target_column"])
    fold_col = str(contract["fold_column"])
    if fold_col != "fold":
        raise ValueError(f"Runner expects contract fold_column='fold', got {fold_col!r}")

    columns = resolve_feature_columns(config, contract)
    categorical_features = get_nested(config, "features.categorical_features", "auto")

    train_frame = load_split_frame(
        config,
        "train",
        contract,
        columns=[id_col, target_col, fold_col, *columns],
        n_folds=n_folds,
    )
    metadata = load_split_metadata(resolve_path(config, "split.metadata"))
    n_holdout = parquet_row_count(dataset_path(config, "val"))
    validate_split_metadata(
        metadata,
        n_train=len(train_frame),
        n_val=n_holdout,
        n_folds=n_folds,
    )

    development = _prepare_development(
        train_frame,
        columns,
        id_col=id_col,
        target_col=target_col,
        sample=sample,
        sample_seed=split_seed,
    )
    LOGGER.info(
        "Experiment=%s variant=%s dataset=%s selection=%s: %d features, "
        "%d development rows (%.1f MB), %d hold-out rows kept locked",
        get_nested(config, "experiment", "unknown"),
        get_nested(config, "variant", "unknown"),
        contract["version"],
        get_nested(config, "features", {}),
        len(columns),
        len(development),
        frame_memory_mb(development),
        n_holdout,
    )

    for seed in seeds:
        set_seeds(seed)
        expected_outputs = [
            _render_output_path(config, "summary_csv", seed=seed, sample=sample),
            _render_output_path(config, "folds_csv", seed=seed, sample=sample),
            _render_output_path(config, "oof", seed=seed, sample=sample),
            _render_output_path(config, "run_json", seed=seed, sample=sample),
        ]
        if all(path.exists() for path in expected_outputs) and not force:
            LOGGER.info(
                "Skipping seed=%d: all outputs exist under %s (use --force to overwrite)",
                seed,
                expected_outputs[0].parent,
            )
            continue

        start = time.perf_counter()
        result = run_cv(
            development,
            id_col=id_col,
            target_col=target_col,
            model_factory=lambda model_seed: _model_factory(config, seed=model_seed),
            seed=seed,
            n_folds=n_folds,
            categorical_features=categorical_features,
            metrics_config=metrics_config,
        )
        elapsed = time.perf_counter() - start

        paths = _save_outputs(
            config,
            result=result,
            seed=seed,
            sample=sample,
            n_holdout=n_holdout,
            elapsed_seconds=elapsed,
        )
        try:
            _log_to_mlflow(config, result=result, seed=seed, paths=paths)
        except Exception as exc:  # noqa: BLE001 - MLflow must not lose results
            LOGGER.warning("MLflow logging failed; JSON run record was still written: %s", exc)

        LOGGER.info(
            "seed=%d complete: AUC=%.5f +/- %.5f, KS=%.5f, PR-AUC=%.5f, Brier=%.5f, ECE=%.5f",
            seed,
            result.summary.get("roc_auc_mean", float("nan")),
            result.summary.get("roc_auc_std", float("nan")),
            result.summary.get("ks_mean", float("nan")),
            result.summary.get("pr_auc_mean", float("nan")),
            result.summary.get("brier_mean", float("nan")),
            result.summary.get("ece_mean", float("nan")),
        )
        LOGGER.info("Artifacts: %s", ", ".join(str(path) for path in paths.values()))

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-safe cross-validation.")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
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
        help="Overwrite existing outputs for this seed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        return run_from_config(config, sample=args.sample, force=args.force)
    except Exception:
        setup_logging()
        LOGGER.exception("Cross-validation run failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
