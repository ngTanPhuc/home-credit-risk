"""Deployment path: the frozen bundle and the Kaggle submission (Phase 11 optional)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import calibration_metrics
from src.config import resolve_path
from src.cv.dataset import load_contract, resolve_feature_columns
from src.cv.runner import render_output_path
from src.evaluation.holdout import run_holdout
from src.evaluation.model_comparison import load_seed_oof, model_definitions
from src.evaluation.submit import run_submission
from src.models.final import FinalModel, fit_final_model

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def _config(dataset, tmp_path: Path) -> dict:
    config = deepcopy(dataset.config)
    config["seeds"] = {"split": 42, "model": [42]}
    config["comparison"] = {
        "models": {
            "lightgbm": {
                "name": "lightgbm",
                "params": {"n_estimators": 5, "num_leaves": 4, "verbosity": -1},
            },
            "xgboost": {
                "name": "xgboost",
                "params": {"n_estimators": 5, "max_depth": 3},
            },
        },
        "ensembles": ["stacking"],
        "weight_step": 0.5,
        "best_params": str(tmp_path / "E4_best_params.json"),
    }
    (tmp_path / "E4_best_params.json").write_text(
        json.dumps(
            {
                "models": {
                    "lightgbm": {"final_params": {"n_estimators": 5, "num_leaves": 4}},
                    "xgboost": {"final_params": {"n_estimators": 5, "max_depth": 3}},
                }
            }
        ),
        encoding="utf-8",
    )
    config["outputs"] = {
        "exp": "submit_test",
        "oof": "reports/predictions/{exp}/E5_{variant}_seed{seed}_oof.parquet",
    }
    config["holdout"] = {
        "table": "reports/tables/B3_calibration.csv",
        "delong_table": "reports/tables/B4_delong_pairs.csv",
        "figure": "reports/figures/holdout_reliability.png",
        "record": "reports/runs/{exp}/holdout_final.json",
    }
    config["submission"] = {
        "out": "reports/submissions/submission_{exp}.csv",
        "model": "reports/runs/{exp}/final_model.joblib",
        "record": "reports/runs/{exp}/submission.json",
    }
    config["final"] = {"arm": "stacking", "calibrator": "isotonic", "seed": 42}
    return config


def _write_fake_oof(config: dict, dataset) -> None:
    train = pd.read_parquet(Path(dataset.dataset_dir) / "train.parquet")
    rng = np.random.default_rng(9)
    for name in config["comparison"]["models"]:
        cfg = deepcopy(config)
        cfg["variant"] = name
        path = render_output_path(cfg, "oof", seed=42, sample=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = train[TARGET].to_numpy()
        noisy = np.clip(0.3 + 0.2 * target + rng.normal(0.0, 0.12, size=len(train)), 0.01, 0.99)
        pd.DataFrame(
            {
                ID: train[ID],
                TARGET: target,
                FOLD: train[FOLD],
                "seed": 42,
                "model": name,
                "y_prob": noisy,
            }
        ).to_parquet(path, index=False)


def test_submission_writes_csv_bundle_and_record(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    _write_fake_oof(config, synthetic_dataset)
    out = tmp_path / "submission.csv"
    bundle = tmp_path / "final_model.joblib"

    result = run_submission(config, out=out, save_model=bundle)

    test = pd.read_parquet(Path(synthetic_dataset.dataset_dir) / "test.parquet")
    submission = pd.read_csv(out)
    assert list(submission.columns) == [ID, TARGET]
    assert len(submission) == len(test)
    assert submission[ID].tolist() == test[ID].tolist()
    assert submission[TARGET].between(0.0, 1.0).all()
    assert result["n_rows"] == len(test)
    assert result["mean_probability"] == pytest.approx(submission[TARGET].mean())

    assert bundle.is_file()
    reloaded = FinalModel.load(bundle)
    assert reloaded.arm == "stacking"
    assert reloaded.calibrator is not None and reloaded.calibrator.name == "isotonic"
    features = test.loc[:, synthetic_dataset.contract["predictor_columns"]]
    assert np.allclose(
        reloaded.predict_proba(features), submission[TARGET].to_numpy(dtype="float64")
    )

    record = json.loads(Path(result["record"]).read_text(encoding="utf-8"))
    assert record["choice"] == {"arm": "stacking", "calibrator": "isotonic", "seed": 42}
    assert record["n_test_rows"] == len(test)
    assert record["n_development_rows"] == 60
    assert record["model_bundle"] == str(bundle)
    assert record["development_positive_rate"] == pytest.approx(0.5, abs=0.2)


def test_submission_reuses_the_saved_bundle(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    _write_fake_oof(config, synthetic_dataset)
    bundle = tmp_path / "final_model.joblib"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    run_submission(config, out=first, save_model=bundle)
    run_submission(config, out=second, reuse_model=bundle, force=True)

    assert pd.read_csv(first)[TARGET].tolist() == pytest.approx(
        pd.read_csv(second)[TARGET].tolist()
    )


def test_submission_smoke_mode_suffixed_and_small(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    _write_fake_oof(config, synthetic_dataset)

    result = run_submission(config, sample=30)

    assert "_sample30" in result["submission"]
    assert "_sample30" in result["record"]
    assert result["n_rows"] == 10  # the synthetic test split has 10 rows
    record = json.loads(Path(result["record"]).read_text(encoding="utf-8"))
    assert record["sample"] == 30


def test_final_model_reproduces_the_holdout_evaluation(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    _write_fake_oof(config, synthetic_dataset)
    holdout_paths = run_holdout(config, confirm_final=True)
    frozen = pd.read_csv(holdout_paths["table"]).query("is_final").iloc[0]

    contract = load_contract(resolve_path(config, "dataset.contract"))
    predictors = resolve_feature_columns(config, contract)
    categorical = [name for name in predictors if name in set(contract["categorical_columns"])]
    train = pd.read_parquet(Path(synthetic_dataset.dataset_dir) / "train.parquet")
    holdout = pd.read_parquet(Path(synthetic_dataset.dataset_dir) / "val.parquet")
    oof = load_seed_oof(
        config,
        list(model_definitions(config)),
        seed=42,
        sample=None,
        identifier_column=ID,
        target_column=TARGET,
        fold_column=FOLD,
    )
    model = fit_final_model(
        config,
        model_definitions(config),
        train.loc[:, predictors],
        train[TARGET].to_numpy(dtype="int8"),
        oof=oof,
        arm="stacking",
        calibrator="isotonic",
        predictor_columns=predictors,
        categorical_features=categorical,
        seed=42,
    )

    probabilities = model.predict_proba(holdout.loc[:, predictors])
    metrics = calibration_metrics(
        holdout[TARGET].to_numpy(dtype="int8"), probabilities, thresholds=(0.5,), ece_bins=5
    )

    assert metrics["brier"] == pytest.approx(frozen["brier"], abs=1e-9)
    assert metrics["roc_auc"] == pytest.approx(frozen["roc_auc"], abs=1e-9)
    assert metrics["ece"] == pytest.approx(frozen["ece"], abs=1e-9)
