"""Unit tests for the per-fold imbalance strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.imbalance import build_strategy, spec_from_config
from src.models.lgbm import LGBMModel

CATEGORICAL = ["CAT"]


def _toy(n_rows: int = 400, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "NUM": rng.normal(size=n_rows),
            "NUM_MISSING": rng.normal(size=n_rows),
            "CAT": pd.Series(rng.choice(["a", "b", "c"], n_rows), dtype="category"),
        }
    )
    features.loc[::7, "NUM_MISSING"] = np.nan
    features.loc[::11, "CAT"] = np.nan
    target = (rng.random(n_rows) < 0.15).astype("int8")
    return features, target


def _minority_ratio(target: np.ndarray) -> float:
    positives = int(np.asarray(target).sum())
    return positives / (len(target) - positives)


def test_none_keeps_rows_and_adds_no_weight():
    features, target = _toy()
    out_features, out_target, weights = build_strategy("none").fit_resample(
        features, target, categorical_features=CATEGORICAL
    )

    assert out_features is features
    assert np.array_equal(out_target, target)
    assert weights is None


def test_class_weight_matches_scale_pos_weight():
    features, target = _toy()
    _, out_target, weights = build_strategy("class_weight").fit_resample(
        features, target, categorical_features=CATEGORICAL
    )

    positives = int(target.sum())
    negatives = len(target) - positives
    assert weights is not None and len(weights) == len(target)
    assert np.allclose(weights[out_target == 1], negatives / positives)
    assert np.allclose(weights[out_target == 0], 1.0)


def test_undersample_reaches_ratio_and_keeps_dtypes():
    features, target = _toy()
    out_features, out_target, weights = build_strategy(
        {"name": "undersample", "sampling_strategy": 0.5}, seed=42
    ).fit_resample(features, target, categorical_features=CATEGORICAL)

    assert weights is None
    assert len(out_features) < len(features)
    assert _minority_ratio(out_target) == pytest.approx(0.5, abs=0.02)
    assert list(out_features.columns) == list(features.columns)
    assert isinstance(out_features["CAT"].dtype, pd.CategoricalDtype)


def test_smotenc_fills_missing_values_and_restores_categories():
    features, target = _toy()
    out_features, out_target, weights = build_strategy(
        {"name": "smotenc", "sampling_strategy": 0.5, "n_neighbors": 5}, seed=42
    ).fit_resample(features, target, categorical_features=CATEGORICAL)

    assert weights is None
    assert len(out_features) >= len(features)
    assert not out_features.isna().any().any()
    assert _minority_ratio(out_target) == pytest.approx(0.5, abs=0.02)
    assert isinstance(out_features["CAT"].dtype, pd.CategoricalDtype)
    assert set(out_features["CAT"].cat.categories) <= {"a", "b", "c"}


def test_smotenc_rejects_unseen_categorical_columns():
    features, target = _toy()
    with pytest.raises(KeyError, match="NOT_A_COLUMN"):
        build_strategy("smotenc", seed=42).fit_resample(
            features, target, categorical_features=["NOT_A_COLUMN"]
        )


def test_unknown_strategy_and_option_fail_loudly():
    with pytest.raises(ValueError, match="Unknown imbalance strategy"):
        build_strategy("magic")
    with pytest.raises(ValueError, match="Unsupported imbalance options"):
        build_strategy({"name": "undersample", "bogus": 1})


def test_spec_from_config_ignores_driver_settings():
    config = {
        "imbalance": {
            "strategy": "smotenc",
            "variants": ["none", "smotenc"],
            "sampling_strategy": 0.5,
            "n_neighbors": 5,
            "table": "reports/tables/E3_imbalance_comparison.csv",
        }
    }

    assert spec_from_config(config) == {
        "name": "smotenc",
        "sampling_strategy": 0.5,
        "n_neighbors": 5,
    }


def test_lgbm_rejects_mismatched_sample_weight():
    features, target = _toy(n_rows=120)
    model = LGBMModel(params={"n_estimators": 5, "min_child_samples": 5}, seed=42)
    with pytest.raises(ValueError, match="sample_weight length"):
        model.fit(features, target, categorical_features=CATEGORICAL, sample_weight=np.ones(3))
