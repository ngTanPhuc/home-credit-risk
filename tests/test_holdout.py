"""Phase 9: the hold-out runs once, behind --confirm-final."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cv.dataset import load_holdout_features, load_split_frame
from src.cv.runner import render_output_path
from src.evaluation.holdout import final_choice, run_holdout

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
    best_params = {
        "models": {
            "lightgbm": {"final_params": {"n_estimators": 5, "num_leaves": 4}},
            "xgboost": {"final_params": {"n_estimators": 5, "max_depth": 3}},
        }
    }
    (tmp_path / "E4_best_params.json").write_text(json.dumps(best_params), encoding="utf-8")

    config["outputs"] = {
        "exp": "holdout_test",
        "oof": "reports/predictions/{exp}/E5_{variant}_seed{seed}_oof.parquet",
    }
    config["holdout"] = {
        "table": "reports/tables/B3_calibration.csv",
        "delong_table": "reports/tables/B4_delong_pairs.csv",
        "figure": "reports/figures/holdout_reliability.png",
        "record": "reports/runs/{exp}/holdout_final.json",
    }
    config["final"] = {"arm": "stacking", "calibrator": "platt", "seed": 42}
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


def test_holdout_features_never_expose_labels(synthetic_dataset):
    features = load_holdout_features(synthetic_dataset.config, synthetic_dataset.contract)

    assert TARGET not in features.columns
    assert ID in features.columns
    assert set(features.columns) == {ID, *synthetic_dataset.contract["predictor_columns"]}

    with pytest.raises(ValueError, match="never returns the target"):
        load_holdout_features(
            synthetic_dataset.config, synthetic_dataset.contract, columns=[TARGET]
        )


def test_holdout_split_stays_locked_for_general_loaders(synthetic_dataset):
    with pytest.raises(PermissionError, match="hold-out"):
        load_split_frame(synthetic_dataset.config, "val", synthetic_dataset.contract)


def test_final_choice_rejects_an_unknown_arm(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    config["final"]["arm"] = "magic"

    with pytest.raises(ValueError, match="final.arm"):
        final_choice(config)


def test_final_choice_rejects_an_unknown_calibrator(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    config["final"]["calibrator"] = "temperature"

    with pytest.raises(ValueError, match="final.calibrator"):
        final_choice(config)


def test_holdout_refuses_to_run_without_the_confirmation_flag(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)

    with pytest.raises(PermissionError, match="confirm-final"):
        run_holdout(config, confirm_final=False)

    assert not (tmp_path / "reports" / "tables" / "B3_calibration.csv").exists()


def test_holdout_writes_b3_b4_and_refuses_a_second_run(synthetic_dataset, tmp_path):
    config = _config(synthetic_dataset, tmp_path)
    _write_fake_oof(config, synthetic_dataset)

    paths = run_holdout(config, confirm_final=True)

    table = pd.read_csv(paths["table"])
    assert set(table["arm"]) == {"lightgbm", "xgboost", "stacking"}
    assert set(table["calibrator"]) == {"none", "platt"}
    assert int(table["is_final"].sum()) == 1
    final_row = table.loc[table["is_final"]].iloc[0]
    assert final_row["arm"] == "stacking" and final_row["calibrator"] == "platt"
    assert {"roc_auc", "brier", "ece", "log_loss", "delta_brier_vs_none"} <= set(table.columns)
    assert (table["n_rows"] == 15).all()  # the synthetic hold-out

    delong = pd.read_csv(paths["delong_table"])
    assert len(delong) == 3  # three arms, three pairs
    assert delong["p_value"].between(0.0, 1.0).all()

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert record["choice"] == {"arm": "stacking", "calibrator": "platt", "seed": 42}
    assert record["n_holdout_rows"] == 15
    assert record["n_development_rows"] == 60
    assert record["metrics"]["brier"] == pytest.approx(final_row["brier"])

    with pytest.raises(FileExistsError, match="already evaluated"):
        run_holdout(config, confirm_final=True)
