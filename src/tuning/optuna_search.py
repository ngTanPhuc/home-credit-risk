"""Optuna TPE tuning for E4 and the default-vs-tuned comparison.

One study per model, shared by every model: same folds, features, seed, imbalance
variant (E3 decision: ``class_weight``) and early-stopping budget. The objective
is the mean validation AUC over the 5 development folds.

Leakage notes
-------------
* Every trial rebuilds the imbalance strategy and the model inside each fold, so
  resampling and target encoding see training rows only.
* Early stopping watches the fold's validation rows. That is tuning, not final
  evaluation: the hold-out stays locked, and the saved ``fixed_iterations``
  value lets E5 refit without early stopping or validation-fold peeking.
* The Optuna study is resumable through SQLite; ``--force`` deletes the scratch
  study database and starts over.

Usage
-----
``uv run python -m src.tuning.optuna_search --config configs/experiments/E4_tuning.yaml``

``--sample N`` runs on N development rows with sample-suffixed outputs,
``--trials N`` overrides the total trial budget (use it to time a subsample),
and ``--models lightgbm,xgboost`` restricts the models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score

from src.config import get_nested, load_config, resolve_path, set_seeds
from src.cv.dataset import (
    dataset_path,
    load_contract,
    load_split_frame,
    resolve_feature_columns,
)
from src.cv.runner import (
    _prepare_development,
    _resolve_categorical_features,
    render_output_path,
    run_from_config,
)
from src.cv.splits import load_split_metadata, parquet_row_count, validate_split_metadata
from src.imbalance import build_strategy, spec_from_config
from src.models import build_model
from src.utils.logging import setup_logging, write_json
from src.utils.memory import peak_rss_mb

LOGGER = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _single_seed(config: Mapping[str, Any]) -> int:
    """Return the one model seed E4 uses; studies cannot merge seeds."""
    seeds = get_nested(config, "seeds.model", [42])
    seeds = [int(seeds)] if isinstance(seeds, int) else [int(seed) for seed in seeds]
    if len(seeds) != 1:
        raise ValueError("E4 expects exactly one model seed")
    return seeds[0]


def _render_path(
    config: Mapping[str, Any],
    dotted_key: str,
    *,
    sample: int | None = None,
) -> Path:
    """Render a path template containing ``{exp}``, anchored at the repo root."""
    root = Path(str(get_nested(config, "_meta.project_root", "."))).expanduser().resolve()
    experiment_dir = str(get_nested(config, "outputs.exp", "E4_tuning"))
    path = Path(str(get_nested(config, dotted_key)).format(exp=experiment_dir)).expanduser()
    if not path.is_absolute():
        path = root / path
    if sample is not None:
        path = path.with_name(f"{path.stem}_sample{sample}{path.suffix}")
    return path


def model_definitions(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load each ``tuning.models`` entry: a YAML path or an inline model block."""
    declared = get_nested(config, "tuning.models", {}) or {}
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("tuning.models is empty; declare model name -> YAML path")

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
            raise ValueError(f"tuning.models[{name!r}] declares model name {model_block['name']!r}")
        if "tuning" not in model_block:
            raise ValueError(f"Model {name!r} has no tuning block (base params + search space)")
        definitions[str(name)] = model_block
    return definitions


def sample_params(trial: optuna.Trial, search_space: Mapping[str, Any]) -> dict[str, Any]:
    """Sample one parameter set from a YAML search space."""
    if not search_space:
        raise ValueError("tuning.search_space is empty")
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = str(spec.get("type", "")).lower()
        if kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported search-space type {kind!r} for {name!r}")
    return params


def _objective(
    trial: optuna.Trial,
    *,
    model_name: str,
    model_block: Mapping[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    fold_ids: np.ndarray,
    categorical: Sequence[str],
    imbalance: Any,
    seed: int,
    n_folds: int,
) -> float:
    """Fit one trial across all folds and return the mean validation AUC."""
    tuning = dict(model_block["tuning"])
    early_stopping_rounds = int(tuning.get("early_stopping_rounds", 100))
    sampled = sample_params(trial, tuning["search_space"])
    params = {
        **dict(model_block.get("params") or {}),
        **dict(tuning.get("base_params") or {}),
        **sampled,
    }
    trial_model = {**dict(model_block), "params": params}
    trial_model.pop("tuning", None)

    fold_aucs: list[float] = []
    best_iterations: list[int | None] = []
    for fold_id in range(n_folds):
        train_mask = fold_ids != fold_id
        valid_mask = fold_ids == fold_id
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Fold {fold_id} has no training or validation rows")

        strategy = build_strategy(imbalance, seed=seed)
        train_features, train_target, weight = strategy.fit_resample(
            features.loc[train_mask],
            target[train_mask],
            categorical_features=categorical,
        )
        model = build_model(trial_model, seed=seed)
        model.fit(
            train_features,
            train_target,
            categorical_features=categorical,
            sample_weight=weight,
            validation=(features.loc[valid_mask], target[valid_mask]),
            early_stopping_rounds=early_stopping_rounds,
        )
        probabilities = model.predict_proba(features.loc[valid_mask])
        fold_aucs.append(float(roc_auc_score(target[valid_mask], probabilities)))
        best_iterations.append(model.best_iteration)

        trial.report(float(np.mean(fold_aucs)), fold_id)
        if trial.should_prune():
            trial.set_user_attr("fold_aucs", fold_aucs)
            trial.set_user_attr("best_iterations", best_iterations)
            raise optuna.TrialPruned()

    trial.set_user_attr("fold_aucs", fold_aucs)
    trial.set_user_attr("best_iterations", best_iterations)
    trial.set_user_attr("model", model_name)
    return float(np.mean(fold_aucs))


def tune_model(
    config: Mapping[str, Any],
    model_name: str,
    model_block: Mapping[str, Any],
    *,
    features: pd.DataFrame,
    target: np.ndarray,
    fold_ids: np.ndarray,
    categorical: Sequence[str],
    imbalance: Any,
    seed: int,
    n_folds: int,
    n_trials: int,
    sample: int | None,
    force: bool,
) -> optuna.Study:
    """Create or resume the model's study and run missing trials."""
    storage = _render_path(config, "tuning.storage", sample=sample)
    storage.parent.mkdir(parents=True, exist_ok=True)
    if force and storage.is_file():
        LOGGER.warning("--force: deleting scratch Optuna study database %s", storage)
        storage.unlink()

    pruner_cfg = dict(get_nested(config, "tuning.pruner", {}) or {})
    study = optuna.create_study(
        study_name=f"E4_{model_name}",
        storage=f"sqlite:///{storage}",
        direction="maximize",
        sampler=TPESampler(seed=int(get_nested(config, "tuning.sampler_seed", seed))),
        pruner=MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 1)),
        ),
        load_if_exists=True,
    )

    remaining = max(0, int(n_trials) - len(study.trials))
    LOGGER.info(
        "E4 %s: %d existing trials, running %d more (storage=%s)",
        model_name,
        len(study.trials),
        remaining,
        storage,
    )
    if remaining:
        timeout = get_nested(config, "tuning.timeout_seconds", None)
        started = time.perf_counter()
        study.optimize(
            lambda trial: _objective(
                trial,
                model_name=model_name,
                model_block=model_block,
                features=features,
                target=target,
                fold_ids=fold_ids,
                categorical=categorical,
                imbalance=imbalance,
                seed=seed,
                n_folds=n_folds,
            ),
            n_trials=remaining,
            timeout=None if timeout is None else float(timeout),
            catch=(ValueError, RuntimeError),
        )
        LOGGER.info(
            "E4 %s: finished in %.1f s",
            model_name,
            time.perf_counter() - started,
        )

    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f"Study E4_{model_name} has no completed trial")
    return study


def best_payload(study: optuna.Study, model_name: str, model_block: Mapping[str, Any]) -> dict:
    """Summarize the best trial, including a fixed iteration count for E5."""
    best = study.best_trial
    best_iterations = list(best.user_attrs.get("best_iterations", []))
    usable = [int(value) for value in best_iterations if value is not None]
    base_params = dict(model_block["tuning"].get("base_params") or {})
    iterations_parameter = (
        "iterations" if str(model_block.get("name", "")).lower() == "catboost" else "n_estimators"
    )
    if usable:
        fixed_iterations = int(np.median(usable)) + 1
    else:
        # No early stop fired: the full tuning budget was used in every fold.
        fixed_iterations = base_params.get(iterations_parameter)

    final_params = {**base_params, **dict(best.params)}
    if fixed_iterations is not None:
        final_params[iterations_parameter] = int(fixed_iterations)

    return {
        "model": model_name,
        "study": study.study_name,
        "best_value": float(study.best_value),
        "best_params": dict(best.params),
        "base_params": base_params,
        "final_params": final_params,
        "fold_aucs": list(best.user_attrs.get("fold_aucs", [])),
        "best_iterations": best_iterations,
        # E5 refits with this fixed count: n_estimators for LightGBM/XGBoost,
        # iterations for CatBoost. Early stopping is a tuning-only device.
        "iterations_parameter": iterations_parameter,
        "fixed_iterations": fixed_iterations,
        "n_completed_trials": int(
            sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)
        ),
        "n_pruned_trials": int(
            sum(1 for trial in study.trials if trial.state == TrialState.PRUNED)
        ),
        "n_failed_trials": int(sum(1 for trial in study.trials if trial.state == TrialState.FAIL)),
    }


def history_frame(studies: Mapping[str, optuna.Study]) -> pd.DataFrame:
    """Return one row per trial with the running best, for the AUC-vs-trials curve."""
    rows: list[dict[str, Any]] = []
    for model_name, study in studies.items():
        best_so_far: float | None = None
        for trial in study.trials:
            value = trial.value if trial.state == TrialState.COMPLETE else None
            if value is not None:
                best_so_far = value if best_so_far is None else max(best_so_far, value)
            fold_aucs = list(trial.user_attrs.get("fold_aucs", []))
            rows.append(
                {
                    "model": model_name,
                    "trial": int(trial.number),
                    "state": trial.state.name,
                    "value": value,
                    "best_so_far": best_so_far,
                    "fold_auc_0": fold_aucs[0] if len(fold_aucs) > 0 else None,
                    "fold_auc_1": fold_aucs[1] if len(fold_aucs) > 1 else None,
                    "fold_auc_2": fold_aucs[2] if len(fold_aucs) > 2 else None,
                    "fold_auc_3": fold_aucs[3] if len(fold_aucs) > 3 else None,
                    "fold_auc_4": fold_aucs[4] if len(fold_aucs) > 4 else None,
                    "params": json.dumps(trial.params, sort_keys=True),
                    "duration_seconds": (
                        None if trial.duration is None else float(trial.duration.total_seconds())
                    ),
                }
            )
    return pd.DataFrame(rows)


def comparison_frame(
    studies: Mapping[str, optuna.Study],
    summaries: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Default vs tuned AUC per model, plus the saved iteration count."""
    rows: list[dict[str, Any]] = []
    for model_name, study in studies.items():
        payload = summaries[model_name]
        tuned = float(study.best_value)
        default = summaries.get(model_name, {}).get("default_auc")
        fold_aucs = payload["fold_aucs"]
        rows.append(
            {
                "model": model_name,
                "default_auc": default,
                "tuned_auc": tuned,
                "delta_auc": None if default is None else tuned - float(default),
                "tuned_auc_std": (
                    float(pd.Series(fold_aucs, dtype="float64").std(ddof=0)) if fold_aucs else None
                ),
                "n_trials": int(len(study.trials)),
                "n_completed_trials": payload["n_completed_trials"],
                "n_pruned_trials": payload["n_pruned_trials"],
                "n_failed_trials": payload["n_failed_trials"],
                "fixed_iterations": payload["fixed_iterations"],
                "best_params": json.dumps(payload["best_params"], sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def write_history_figure(studies: Mapping[str, optuna.Study], path: Path) -> Path:
    """Plot best-so-far AUC against completed trial count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for model_name, study in studies.items():
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            continue
        numbers = list(range(1, len(completed) + 1))
        values = pd.Series([float(t.value) for t in completed]).cummax().tolist()
        axis.plot(numbers, values, marker="o", markersize=3, label=model_name)

    axis.set_xlabel("Completed trial")
    axis.set_ylabel("Best CV mean ROC-AUC")
    axis.set_title("E4 Optuna TPE tuning")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _default_config(
    config: Mapping[str, Any],
    model_name: str,
    model_block: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the runner config for a model's default-parameter CV."""
    cfg = deepcopy(dict(config))
    cfg["model"] = {key: value for key, value in model_block.items() if key != "tuning"}
    cfg["variant"] = f"default_{model_name}"
    return cfg


def _default_auc(
    config: Mapping[str, Any],
    model_name: str,
    model_block: Mapping[str, Any],
    *,
    seed: int,
    sample: int | None,
    force: bool,
) -> float | None:
    """Run (or reuse) the default-parameter CV and return its mean AUC."""
    if not bool(get_nested(config, "tuning.compare_defaults", True)):
        return None
    cfg = _default_config(config, model_name, model_block)
    run_from_config(cfg, sample=sample, force=force)
    summary_path = render_output_path(cfg, "summary_csv", seed=seed, sample=sample)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Default run summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    return float(summary.loc[0, "roc_auc_mean"])


def run_tuning(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    force: bool = False,
    n_trials: int | None = None,
    models: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Run E4: default CV per model, Optuna studies, then the E4 tables."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    n_folds = int(get_nested(config, "split.n_folds", 5))
    seed = _single_seed(config)
    budget = int(n_trials if n_trials is not None else get_nested(config, "tuning.n_trials", 40))
    if budget < 1:
        raise ValueError("tuning budget must be at least 1 trial")

    definitions = model_definitions(config)
    selected = [str(name) for name in (models or definitions)]
    unknown = [name for name in selected if name not in definitions]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; known: {sorted(definitions)}")

    contract = load_contract(resolve_path(config, "dataset.contract"))
    id_col = str(contract["identifier_column"])
    target_col = str(contract["target_column"])
    fold_col = str(contract["fold_column"])
    columns = resolve_feature_columns(config, contract)
    categorical_spec = get_nested(config, "features.categorical_features", "auto")
    imbalance = spec_from_config(config)

    train_frame = load_split_frame(
        config,
        "train",
        contract,
        columns=[id_col, target_col, fold_col, *columns],
        n_folds=n_folds,
    )
    metadata = load_split_metadata(resolve_path(config, "split.metadata"))
    n_holdout = parquet_row_count(dataset_path(config, "val"))
    validate_split_metadata(metadata, n_train=len(train_frame), n_val=n_holdout, n_folds=n_folds)

    development = _prepare_development(
        train_frame,
        columns,
        id_col=id_col,
        target_col=target_col,
        sample=sample,
        sample_seed=int(get_nested(config, "seeds.split", 42)),
    )
    features = development.loc[:, columns]
    target = development[target_col].to_numpy(dtype="int8")
    fold_ids = development[fold_col].to_numpy(dtype="int8")
    categorical = _resolve_categorical_features(features, categorical_spec)
    LOGGER.info(
        "E4 start: %d features, %d development rows (%d folds), seed=%d, budget=%d/model",
        len(columns),
        len(development),
        n_folds,
        seed,
        budget,
    )

    set_seeds(seed)
    defaults = {
        name: _default_auc(config, name, definitions[name], seed=seed, sample=sample, force=force)
        for name in selected
    }

    studies: dict[str, optuna.Study] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name in selected:
        study = tune_model(
            config,
            name,
            definitions[name],
            features=features,
            target=target,
            fold_ids=fold_ids,
            categorical=categorical,
            imbalance=imbalance,
            seed=seed,
            n_folds=n_folds,
            n_trials=budget,
            sample=sample,
            force=force,
        )
        studies[name] = study
        summaries[name] = best_payload(study, name, definitions[name])
        summaries[name]["default_auc"] = defaults[name]

    table_path = _render_path(config, "tuning.table", sample=sample)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame(studies, summaries).to_csv(table_path, index=False)

    history_path = _render_path(config, "tuning.history_csv", sample=sample)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_frame(studies).to_csv(history_path, index=False)

    best_params_path = _render_path(config, "tuning.best_params", sample=sample)
    write_json(
        best_params_path,
        {
            "experiment": get_nested(config, "experiment", "E4_tuning"),
            "seed": seed,
            "n_features": len(columns),
            "n_development_rows": int(len(development)),
            "imbalance": imbalance,
            "models": summaries,
            "peak_memory_mb": float(peak_rss_mb()),
        },
    )

    figure_path = write_history_figure(
        studies, _render_path(config, "tuning.figure", sample=sample)
    )
    LOGGER.info("E4 comparison: %s", table_path)
    LOGGER.info("E4 best params: %s", best_params_path)
    return {
        "table": table_path,
        "history": history_path,
        "best_params": best_params_path,
        "figure": figure_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune models with Optuna TPE (E4).")
    parser.add_argument("--config", required=True, help="E4 tuning YAML config")
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
        help="Delete the scratch Optuna study database and rerun the default CV.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help="Total trial budget per model, overriding tuning.n_trials.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to tune (default: every tuning.models entry).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        models = None if args.models is None else [name.strip() for name in args.models.split(",")]
        paths = run_tuning(
            config,
            sample=args.sample,
            force=args.force,
            n_trials=args.trials,
            models=models,
        )
    except Exception:
        setup_logging()
        LOGGER.exception("E4 tuning failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("E4 %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
