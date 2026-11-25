"""Model wrappers added in Phase 6: XGBoost target encoding and CatBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import build_model
from src.models.catboost import CatBoostModel
from src.models.xgb import XGBModel

CATEGORICAL = ["CAT"]


def _toy(n_rows: int = 240, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "NUM": rng.normal(size=n_rows),
            "NUM_MISSING": rng.normal(size=n_rows),
            "CAT": pd.Series(rng.choice(["a", "b", "c"], n_rows), dtype="category"),
        }
    )
    features.loc[::9, "NUM_MISSING"] = np.nan
    signal = features["CAT"].astype("object").map({"a": -1.0, "b": 0.0, "c": 1.0})
    target = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-signal.to_numpy()))).astype("int8")
    return features, target


def test_xgb_target_encoding_learns_only_training_categories():
    features, target = _toy()
    model = XGBModel(params={"n_estimators": 20, "max_depth": 3}, seed=42)
    model.fit(features, target, categorical_features=CATEGORICAL)

    assert set(model._encoding_maps["CAT"].index) == {"a", "b", "c"}
    probabilities = model.predict_proba(features.head(7))
    assert probabilities.shape == (7,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert not model.feature_importance().empty


def test_xgb_unseen_and_missing_categories_fall_back_to_prior():
    features, target = _toy()
    model = XGBModel(params={"n_estimators": 10, "max_depth": 3}, seed=42)
    model.fit(features, target, categorical_features=CATEGORICAL)

    unseen = features.head(5).copy()
    unseen["CAT"] = pd.Categorical(["z", "z", None, "z", "z"], categories=["a", "b", "c", "z"])
    probabilities = model.predict_proba(unseen)

    assert np.all(np.isfinite(probabilities))
    assert "z" not in model._encoding_maps["CAT"].index


def test_xgb_rejects_single_class_fold():
    features, target = _toy(n_rows=120)
    model = XGBModel(params={"n_estimators": 5})
    with pytest.raises(ValueError, match="both classes"):
        model.fit(features, np.zeros(len(target), dtype="int8"))


def test_catboost_handles_native_categories_and_missing():
    features, target = _toy()
    features.loc[::11, "CAT"] = np.nan
    model = CatBoostModel(
        params={"iterations": 20, "depth": 3, "learning_rate": 0.1},
        seed=42,
    )
    model.fit(features, target, categorical_features=CATEGORICAL)

    probabilities = model.predict_proba(features.head(7))
    assert probabilities.shape == (7,)
    assert np.all(np.isfinite(probabilities))
    assert not model.feature_importance().empty


def test_early_stopping_sets_best_iteration_for_all_wrappers():
    features, target = _toy(n_rows=300)
    split = 200
    train_features, train_target = features.iloc[:split], target[:split]
    valid_features, valid_target = features.iloc[split:], target[split:]

    for model in (
        build_model(
            {
                "name": "lightgbm",
                "params": {"n_estimators": 40, "min_child_samples": 5},
                "runtime": {"n_jobs": 1},
            },
            seed=42,
        ),
        build_model(
            {
                "name": "xgboost",
                "params": {"n_estimators": 40, "max_depth": 3},
                "runtime": {"n_jobs": 1},
            },
            seed=42,
        ),
        build_model(
            {
                "name": "catboost",
                "params": {"iterations": 40, "depth": 3},
                "runtime": {"thread_count": 1},
            },
            seed=42,
        ),
    ):
        model.fit(
            train_features,
            train_target,
            categorical_features=CATEGORICAL,
            validation=(valid_features, valid_target),
            early_stopping_rounds=5,
        )
        assert model.best_iteration is not None
        assert model.best_iteration >= 0


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "magic"}, seed=42)
