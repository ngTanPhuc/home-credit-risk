"""E5 driver: B2, pairwise DeLong and the ROC/PR figure from OOF files (Phase 7)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cv.runner import render_output_path
from src.evaluation.model_comparison import write_outputs

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def _e5_config(dataset) -> dict:
    config = deepcopy(dataset.config)
    config["seeds"] = {"split": 42, "model": [42, 43]}
    config["comparison"] = {
        "models": {},
        "ensembles": ["weighted_average", "stacking"],
        "weight_step": 0.25,
        "table": "reports/tables/B2_model_comparison.csv",
        "delong_table": "reports/tables/E5_delong_pairs.csv",
        "figure": "reports/figures/E5_roc_pr.png",
        "record": "reports/runs/{exp}/E5_ensembles.json",
    }
    config["outputs"] = {
        "exp": "E5_test",
        "folds_csv": "reports/tables/{exp}/E5_{variant}_folds_seed{seed}.csv",
        "summary_csv": "reports/tables/{exp}/E5_{variant}_summary_seed{seed}.csv",
        "importance_csv": "reports/tables/{exp}/E5_{variant}_importance_seed{seed}.csv",
        "oof": "reports/predictions/{exp}/E5_{variant}_seed{seed}_oof.parquet",
        "run_json": "reports/runs/{exp}/E5_{variant}_seed{seed}.json",
    }
    return config


def _write_fake_oof(
    config: dict,
    dataset,
    model_names: list[str],
    seeds: list[int],
) -> None:
    train = pd.read_parquet(Path(dataset.dataset_dir) / "train.parquet")
    rng = np.random.default_rng(3)
    for seed in seeds:
        for name in model_names:
            cfg = deepcopy(config)
            cfg["variant"] = name
            path = render_output_path(cfg, "oof", seed=seed, sample=None)
            path.parent.mkdir(parents=True, exist_ok=True)
            y = train[TARGET].to_numpy()
            noisy = np.clip(0.3 + 0.2 * y + rng.normal(0.0, 0.15, size=len(train)), 0.01, 0.99)
            pd.DataFrame(
                {
                    ID: train[ID],
                    TARGET: y,
                    FOLD: train[FOLD],
                    "seed": seed,
                    "model": name,
                    "y_prob": noisy,
                }
            ).to_parquet(path, index=False)


def test_write_outputs_builds_b2_delong_figure_and_record(synthetic_dataset):
    config = _e5_config(synthetic_dataset)
    model_names = ["lightgbm", "xgboost"]
    seeds = [42, 43]
    _write_fake_oof(config, synthetic_dataset, model_names, seeds)

    paths = write_outputs(config, model_names=model_names, seeds=seeds)

    table = pd.read_csv(paths["table"])
    assert list(table["arm"]) == [*model_names, "weighted_average", "stacking"]
    assert (table["n_seeds"] == 2).all()
    assert {"roc_auc_mean", "roc_auc_std", "brier_mean", "ece_mean"} <= set(table.columns)
    assert table["roc_auc_mean"].between(0.0, 1.0).all()

    delong = pd.read_csv(paths["delong_table"])
    assert len(delong) == 2 * 6  # four arms, six pairs per seed
    assert {"seed", "model_a", "model_b", "delta_auc", "z", "p_value"} <= set(delong.columns)
    assert delong["p_value"].between(0.0, 1.0).all()

    assert paths["figure"].is_file()
    assert paths["figure"].stat().st_size > 0

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert record["seeds"] == seeds
    assert set(record["seeds_detail"]) == {"42", "43"}
    weights = record["seeds_detail"]["42"]["average_weights_all_rows"]
    assert set(weights) == set(model_names)
    assert sum(weights.values()) == pytest.approx(1.0)
