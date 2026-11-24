"""Cross-fitted OOF ensembling (Phase 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models.ensemble import (
    align_oof,
    cross_fitted_average,
    cross_fitted_stacking,
    fit_average_weights,
)

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def _frames(
    y: np.ndarray,
    fold: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    ids = np.arange(1_000, 1_000 + len(y))
    return {
        name: pd.DataFrame({ID: ids, TARGET: y, FOLD: fold, "y_prob": values})
        for name, values in probabilities.items()
    }


def _toy_split(n_folds: int = 5, per_fold: int = 60, seed: int = 5):
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(n_folds), per_fold)
    y = rng.integers(0, 2, size=fold.size).astype("int8")
    signal = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0.0, 0.15, size=fold.size), 0.01, 0.99)
    return y, fold, signal, rng


def test_align_oof_sorts_rows_and_keeps_model_order():
    y, fold, signal, rng = _toy_split()
    frames = _frames(y, fold, {"a": signal, "b": rng.random(fold.size)})
    frames["b"] = frames["b"].iloc[::-1].reset_index(drop=True)

    matrix = align_oof(frames)

    assert list(matrix.probabilities.columns) == ["a", "b"]
    assert matrix.meta[ID].is_monotonic_increasing
    first_id = matrix.meta[ID].iloc[0]
    original = frames["b"].set_index(ID).loc[first_id, "y_prob"]
    assert matrix.probabilities["b"].iloc[0] == pytest.approx(original)


def test_align_oof_rejects_a_different_row_set():
    y, fold, signal, rng = _toy_split()
    frames = _frames(y, fold, {"a": signal, "b": rng.random(fold.size)})
    frames["b"] = frames["b"].iloc[:-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="different row set"):
        align_oof(frames)


def test_align_oof_rejects_duplicated_identifiers():
    y, fold, signal, _ = _toy_split()
    frame = _frames(y, fold, {"a": signal})["a"]
    frame.loc[1, ID] = frame.loc[0, ID]

    with pytest.raises(ValueError, match="duplicated identifiers"):
        align_oof({"a": frame})


def test_fit_average_weights_prefers_the_better_model():
    y = np.array([0, 1] * 100)
    probabilities = pd.DataFrame({"good": y.astype(float), "bad": np.full(y.size, 0.5)})

    weights = fit_average_weights(y, probabilities)

    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in weights.values())
    assert weights["good"] == pytest.approx(1.0)


def test_fit_average_weights_needs_both_classes():
    y = np.zeros(20)
    probabilities = pd.DataFrame({"a": np.full(20, 0.3)})

    with pytest.raises(ValueError, match="both classes"):
        fit_average_weights(y, probabilities)


def test_cross_fitted_average_does_not_fit_the_fold_it_predicts():
    y, fold, signal, rng = _toy_split()
    # `leaky` is perfect on fold 0 and pure noise everywhere else. Leak-free
    # weights for fold 0 are fitted on folds 1..4, which must ignore it.
    leaky = np.where(fold == 0, y.astype(float), rng.random(fold.size))
    matrix = align_oof(_frames(y, fold, {"signal": signal, "leaky": leaky}))

    prediction = cross_fitted_average(matrix, step=0.1)

    assert prediction.fold_weights is not None
    assert prediction.fold_weights[0]["leaky"] < 0.5
    assert np.all((prediction.y_prob >= 0.0) & (prediction.y_prob <= 1.0))
    fold_zero = fold == 0
    assert roc_auc_score(y[fold_zero], prediction.y_prob[fold_zero]) < 0.9


def test_cross_fitted_stacking_does_not_fit_the_fold_it_predicts():
    y, fold, signal, rng = _toy_split()
    leaky = np.where(fold == 0, y.astype(float), rng.random(fold.size))
    matrix = align_oof(_frames(y, fold, {"signal": signal, "leaky": leaky}))

    prediction = cross_fitted_stacking(matrix, seed=42)

    assert prediction.y_prob.shape == (len(y),)
    assert np.all((prediction.y_prob > 0.0) & (prediction.y_prob < 1.0))
    fold_zero = fold == 0
    assert roc_auc_score(y[fold_zero], prediction.y_prob[fold_zero]) < 0.8
