"""B1 ablation table: deltas, group attribution and ladder guards."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.cv.runner import render_output_path
from src.evaluation.ablation import (
    build_b1_table,
    feature_set_config,
    run_ablation,
    write_b1_table,
)


def _config(synthetic_dataset) -> dict:
    config = deepcopy(synthetic_dataset.config)
    config.update(
        {
            "ablation": {
                "sets": ["main", "plus_bureau"],
                "table": "tables/B1_ablation.csv",
            },
            "outputs": {
                "exp": "E2_test",
                "folds_csv": "tables/{exp}/E2_{variant}_folds_seed{seed}.csv",
            },
            "seeds": {"model": [42]},
        }
    )
    return config


def _write_folds(
    config: dict, feature_set: str, aucs: list[float], *, sample: int | None = None
) -> None:
    path = render_output_path(
        feature_set_config(config, feature_set), "folds_csv", seed=42, sample=sample
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fold": range(len(aucs)), "roc_auc": aucs}).to_csv(path, index=False)


def test_b1_table_reports_deltas_and_group_gains(synthetic_dataset):
    config = _config(synthetic_dataset)
    _write_folds(config, "main", [0.70, 0.72, 0.71])
    _write_folds(config, "plus_bureau", [0.74, 0.76, 0.75])

    table = build_b1_table(config, seed=42)

    assert table["set"].tolist() == ["main", "plus_bureau"]
    assert table["n_features"].tolist() == [3, 4]
    assert table["n_features_added"].tolist() == [3, 1]
    assert table["group_added"].tolist() == ["application_original", "bureau"]

    first, second = table.iloc[0], table.iloc[1]
    assert first["auc_mean"] == pytest.approx(0.71)
    assert first["auc_std"] == pytest.approx(0.008164965809277268)
    assert pd.isna(first["delta_auc_vs_previous"])
    assert pd.isna(first["gain_per_group_auc"])
    assert second["auc_mean"] == pytest.approx(0.75)
    assert second["delta_auc_vs_previous"] == pytest.approx(0.04)
    assert second["gain_per_group_auc"] == pytest.approx(0.04)

    assert table["auc_fold_2"].tolist() == [0.71, 0.75]


def test_b1_rejects_a_ladder_that_does_not_add_groups(synthetic_dataset):
    config = _config(synthetic_dataset)
    config["ablation"]["sets"] = ["plus_bureau", "main"]
    _write_folds(config, "plus_bureau", [0.70, 0.70, 0.70])
    _write_folds(config, "main", [0.71, 0.71, 0.71])

    with pytest.raises(ValueError, match="adds no feature group"):
        build_b1_table(config, seed=42)


def test_b1_missing_feature_set_output_names_the_file(synthetic_dataset):
    config = _config(synthetic_dataset)
    _write_folds(config, "main", [0.70, 0.70, 0.70])

    with pytest.raises(FileNotFoundError, match="plus_bureau_folds_seed42.csv"):
        build_b1_table(config, seed=42)


def test_b1_rejects_a_wrong_fold_count(synthetic_dataset):
    config = _config(synthetic_dataset)
    _write_folds(config, "main", [0.70, 0.71])
    _write_folds(config, "plus_bureau", [0.72, 0.73, 0.74])

    with pytest.raises(ValueError, match="exactly folds 0..2"):
        build_b1_table(config, seed=42)


def test_write_b1_table_uses_the_sample_suffix(synthetic_dataset):
    config = _config(synthetic_dataset)
    _write_folds(config, "main", [0.70, 0.71, 0.72], sample=500)
    _write_folds(config, "plus_bureau", [0.73, 0.74, 0.75], sample=500)

    path = write_b1_table(config, seed=42, sample=500)

    assert path.name == "B1_ablation_sample500.csv"
    assert path.is_file()
    path.unlink()


def test_e2_rejects_multiple_model_seeds(synthetic_dataset):
    config = _config(synthetic_dataset)
    config["seeds"]["model"] = [42, 43]

    with pytest.raises(ValueError, match="exactly one model seed"):
        run_ablation(config, table_only=True)
