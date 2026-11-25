"""E6 driver: Brier/ECE before vs after calibration from OOF files (Phase 8)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cv.runner import render_output_path
from src.evaluation.calibration_eval import (
    build_calibration_table,
    declared_calibrators,
    run_calibration,
)

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def _config(dataset) -> dict:
    config = deepcopy(dataset.config)
    config["seeds"] = {"split": 42, "model": [42]}
    config["comparison"] = {
        "models": {
            "lightgbm": "configs/models/lightgbm.yaml",
            "xgboost": "configs/models/xgboost.yaml",
        },
        "ensembles": ["stacking"],
        "weight_step": 0.5,
    }
    config["calibration"] = {
        "calibrators": ["none", "platt", "isotonic"],
        "min_rows": 10,
        "table": "reports/tables/E6_calibration.csv",
        "fold_table": "reports/tables/E6_calibration_folds.csv",
        "figure": "reports/figures/E6_reliability.png",
        "record": "reports/runs/{exp}/E6_calibration.json",
    }
    config["outputs"] = {
        "exp": "E6_test",
        "oof": "reports/predictions/{exp}/E5_{variant}_seed{seed}_oof.parquet",
    }
    config["final"] = {"arm": "stacking", "calibrator": "platt"}
    return config


def _write_fake_oof(config: dict, dataset) -> None:
    train = pd.read_parquet(Path(dataset.dataset_dir) / "train.parquet")
    rng = np.random.default_rng(4)
    for name in config["comparison"]["models"]:
        cfg = deepcopy(config)
        cfg["variant"] = name
        path = render_output_path(cfg, "oof", seed=42, sample=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = train[TARGET].to_numpy()
        noisy = np.clip(0.25 + 0.2 * target + rng.normal(0.0, 0.12, size=len(train)), 0.01, 0.99)
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


def test_declared_calibrators_requires_the_reference_row(synthetic_dataset):
    config = _config(synthetic_dataset)
    config["calibration"]["calibrators"] = ["platt", "isotonic"]

    with pytest.raises(ValueError, match="none"):
        declared_calibrators(config)


def test_run_calibration_writes_table_figure_and_record(synthetic_dataset):
    config = _config(synthetic_dataset)
    _write_fake_oof(config, synthetic_dataset)

    paths = run_calibration(config)

    table = pd.read_csv(paths["table"])
    assert set(table["arm"]) == {"lightgbm", "xgboost", "stacking"}
    assert set(table["calibrator"]) == {"none", "platt", "isotonic"}
    assert (table["n_seeds"] == 1).all()
    assert {"brier_mean", "ece_mean", "log_loss_mean", "delta_brier_vs_none"} <= set(table.columns)
    reference = table.loc[table["calibrator"] == "none"].set_index("arm")["brier_mean"]
    assert np.allclose(table.loc[table["calibrator"] == "none", "delta_brier_vs_none"], 0.0)
    assert (reference > 0).all()

    folds = pd.read_csv(paths["fold_table"])
    assert len(folds) == 3 * 3  # three arms, three calibrators, one seed
    assert paths["figure"].is_file() and paths["figure"].stat().st_size > 0

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert set(record["recommended"]) == {"lightgbm", "xgboost", "stacking"}
    assert record["final_choice"] == {"arm": "stacking", "calibrator": "platt"}
    assert record["seeds_detail"]["42"]["n_rows"] == 60


def test_build_calibration_table_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_calibration_table(pd.DataFrame({"arm": ["a"], "calibrator": ["none"]}))
