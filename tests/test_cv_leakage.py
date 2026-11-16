"""Leakage guards for the fold loop and the LightGBM wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cv.runner import run_cv
from src.models.base import BaseModel
from src.models.lgbm import LGBMModel
from tests.conftest import ID_COL, PREDICTORS, TARGET_COL


class RecordingModel(BaseModel):
    """Fake estimator that records which rows each fold actually fits on."""

    name = "recording"
    calls: list[dict] = []

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, features, target, *, categorical_features=None):
        RecordingModel.calls.append(
            {
                "probe": set(features["PROBE_ROW"].tolist()),
                "n_rows": len(features),
                "n_positive": int(np.asarray(target).sum()),
            }
        )
        return self

    def predict_proba(self, features):
        return np.full(len(features), 0.5, dtype="float64")


@pytest.fixture()
def development(synthetic_dataset):
    frame = pd.read_parquet(synthetic_dataset.dataset_dir / "train.parquet")
    frame = frame.loc[:, [ID_COL, TARGET_COL, "fold", *PREDICTORS]].copy()
    frame["PROBE_ROW"] = np.arange(len(frame))
    RecordingModel.calls = []
    return frame


def test_fit_sees_only_training_fold_rows(development):
    result = run_cv(
        development,
        id_col=ID_COL,
        target_col=TARGET_COL,
        model_factory=lambda seed: RecordingModel(seed),
        seed=42,
        n_folds=3,
        categorical_features=[],
        metrics_config={"ece_bins": 5, "thresholds": [0.2]},
    )

    fold_of_row = development.set_index("PROBE_ROW")["fold"].to_dict()
    assert len(RecordingModel.calls) == 3
    for fold_id, call in enumerate(RecordingModel.calls):
        expected_train = {row for row, fold in fold_of_row.items() if fold != fold_id}
        expected_valid = {row for row, fold in fold_of_row.items() if fold == fold_id}
        assert call["probe"] == expected_train
        assert not call["probe"] & expected_valid
        assert call["n_rows"] == len(expected_train)

    assert len(result.fold_metrics) == 3
    assert len(result.oof) == len(development)
    assert result.oof[ID_COL].is_unique
    assert result.oof["fold"].value_counts().sort_index().tolist() == [20, 20, 20]
    assert result.summary["n_folds"] == 3.0


def test_oof_has_one_prediction_per_row_and_no_holdout(development, synthetic_dataset):
    result = run_cv(
        development,
        id_col=ID_COL,
        target_col=TARGET_COL,
        model_factory=lambda seed: RecordingModel(seed),
        seed=42,
        n_folds=3,
        categorical_features=[],
        metrics_config={"ece_bins": 5, "thresholds": []},
    )
    holdout_ids = set(pd.read_parquet(synthetic_dataset.dataset_dir / "val.parquet")[ID_COL])
    assert set(result.oof[ID_COL]).isdisjoint(holdout_ids)


def test_lgbm_wrapper_fits_categories_and_importances(development):
    features = development.loc[:, PREDICTORS]
    target = development[TARGET_COL]
    model = LGBMModel(params={"n_estimators": 20, "min_child_samples": 5}, seed=42)
    model.fit(features, target, categorical_features=["NAME_CONTRACT_TYPE"])

    probabilities = model.predict_proba(features.head(7))
    assert probabilities.shape == (7,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert not model.feature_importance().empty


def test_lgbm_rejects_single_class_fold(development):
    features = development.loc[:, PREDICTORS]
    target = pd.Series(np.zeros(len(development), dtype="int8"))
    with pytest.raises(ValueError, match="both classes"):
        LGBMModel(params={"n_estimators": 5}).fit(features, target)
