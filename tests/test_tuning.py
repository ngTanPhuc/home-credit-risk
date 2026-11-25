"""E4 tuning smoke test: Optuna study, tables and best-params file on synthetic data."""

from __future__ import annotations

import json
from copy import deepcopy

import optuna
import pandas as pd
import pytest

from src.tuning.optuna_search import (
    best_payload,
    model_definitions,
    run_tuning,
    sample_params,
)


def _config(synthetic_dataset) -> dict:
    config = deepcopy(synthetic_dataset.config)
    config["experiment"] = "E4_test"
    config["outputs"] = {"exp": "e4_test"}
    config["imbalance"] = {"strategy": "none"}
    config["tuning"] = {
        "models": {
            "lightgbm": {
                "name": "lightgbm",
                "params": {"n_estimators": 10},
                "runtime": {"n_jobs": 1},
                "tuning": {
                    "early_stopping_rounds": 3,
                    "base_params": {"n_estimators": 15},
                    "search_space": {
                        "num_leaves": {"type": "int", "low": 2, "high": 4},
                        "learning_rate": {"type": "float", "low": 0.05, "high": 0.2},
                    },
                },
            }
        },
        "n_trials": 2,
        "compare_defaults": False,
        "sampler_seed": 42,
        "pruner": {"n_startup_trials": 1, "n_warmup_steps": 1},
        "storage": "runs/{exp}/optuna.db",
        "table": "tables/{exp}/comparison.csv",
        "history_csv": "tables/{exp}/history.csv",
        "best_params": "runs/{exp}/best_params.json",
        "figure": "figures/{exp}/history.png",
    }
    return config


def test_model_definitions_accepts_inline_blocks(synthetic_dataset):
    definitions = model_definitions(_config(synthetic_dataset))
    assert set(definitions) == {"lightgbm"}
    assert definitions["lightgbm"]["tuning"]["base_params"]["n_estimators"] == 15


def test_sample_params_builds_the_requested_types(synthetic_dataset):
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    params = sample_params(
        trial,
        {
            "depth": {"type": "int", "low": 3, "high": 5},
            "rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "loss": {"type": "categorical", "choices": ["a", "b"]},
        },
    )
    assert 3 <= params["depth"] <= 5
    assert 0.01 <= params["rate"] <= 0.1
    assert params["loss"] in {"a", "b"}


def test_run_tuning_writes_tables_and_best_params(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset)
    paths = run_tuning(config, n_trials=2, models=["lightgbm"])

    table = pd.read_csv(paths["table"])
    assert table["model"].tolist() == ["lightgbm"]
    assert table["n_completed_trials"].iloc[0] + table["n_pruned_trials"].iloc[0] >= 1
    assert table["default_auc"].isna().all()  # compare_defaults is off
    assert "best_params" in table.columns

    history = pd.read_csv(paths["history"])
    assert len(history) == 2
    assert history["best_so_far"].notna().any()
    assert paths["figure"].is_file()

    payload = json.loads(paths["best_params"].read_text(encoding="utf-8"))
    best = payload["models"]["lightgbm"]
    assert "fixed_iterations" in best
    assert "num_leaves" in best["best_params"]


def _study_with_best_iterations(values: list[int]) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    trial.set_user_attr("best_iterations", values)
    trial.set_user_attr("fold_aucs", [0.7] * len(values))
    study.tell(trial, 0.71)
    return study


def test_fixed_iterations_follow_each_library_index_base():
    # LightGBM reports the tree count, XGBoost and CatBoost report a zero-based index.
    study = _study_with_best_iterations([9, 11, 10, 12, 8])

    lightgbm = best_payload(
        study,
        "lightgbm",
        {"name": "lightgbm", "tuning": {"base_params": {"n_estimators": 2000}}},
    )
    xgboost = best_payload(
        study,
        "xgboost",
        {"name": "xgboost", "tuning": {"base_params": {"n_estimators": 2000}}},
    )
    catboost = best_payload(
        study,
        "catboost",
        {"name": "catboost", "tuning": {"base_params": {"iterations": 2000}}},
    )

    assert lightgbm["fixed_iterations"] == 10
    assert lightgbm["final_params"]["n_estimators"] == 10
    assert xgboost["fixed_iterations"] == 11
    assert xgboost["final_params"]["n_estimators"] == 11
    assert catboost["fixed_iterations"] == 11
    assert catboost["iterations_parameter"] == "iterations"
    assert catboost["final_params"]["iterations"] == 11


def test_fixed_iterations_fall_back_to_the_tuning_budget_without_early_stop():
    study = _study_with_best_iterations([None, None, None])

    payload = best_payload(
        study,
        "lightgbm",
        {"name": "lightgbm", "tuning": {"base_params": {"n_estimators": 2000}}},
    )

    assert payload["best_iterations"] == [None, None, None]
    assert payload["fixed_iterations"] == 2000


def test_unknown_model_iteration_semantics_fail_loudly():
    study = _study_with_best_iterations([5])

    with pytest.raises(ValueError, match="iteration offset"):
        best_payload(study, "magic", {"name": "magic", "tuning": {"base_params": {}}})
