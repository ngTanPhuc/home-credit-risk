"""Calibration (Phase 8): cross-fitted Platt and isotonic, never in-sample."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.calibration import (
    CALIBRATORS,
    build_calibrator,
    calibration_metrics,
    cross_fitted_predictions,
    recommend_calibrator,
    summarize_calibration,
)


def _toy(
    n_folds: int = 5,
    per_fold: int = 400,
    seed: int = 11,
    inflation: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(target, overconfident-ish probabilities, folds)``."""
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(n_folds), per_fold)
    latent = rng.normal(size=fold.size)
    target = (rng.random(latent.size) < 1.0 / (1.0 + np.exp(-latent))).astype("int8")
    probabilities = np.clip(1.0 / (1.0 + np.exp(-latent)) * inflation, 1e-6, 1 - 1e-6)
    return target, probabilities, fold


def test_platt_repairs_the_scale_and_keeps_the_ranking():
    target, probabilities, folds = _toy()

    calibrated, details = cross_fitted_predictions(
        target, probabilities, folds, build_calibrator("platt"), min_rows=100
    )

    assert brier_score_loss(target, calibrated) < brier_score_loss(target, probabilities)
    # A monotone map cannot change a rank metric within a fold. The pooled AUC can
    # still move a little because every fold gets its own map.
    for fold in np.unique(folds):
        mask = folds == fold
        assert roc_auc_score(target[mask], calibrated[mask]) == pytest.approx(
            roc_auc_score(target[mask], probabilities[mask]), abs=1e-12
        )
    # Each fold's calibrator was fitted on the other four folds.
    assert {item["n_fit"] for item in details.values()} == {1600}


def test_isotonic_is_monotone_and_bounded():
    target, probabilities, _ = _toy()
    calibrator = build_calibrator("isotonic").fit(target, probabilities)

    grid = np.linspace(0.0, 1.0, 250)
    calibrated = calibrator.predict(grid)

    assert np.all(np.diff(calibrated) >= -1e-12)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def test_cross_fitted_predictions_never_fit_the_scored_fold():
    target, probabilities, folds = _toy()
    calibrated, _ = cross_fitted_predictions(
        target, probabilities, folds, build_calibrator("platt"), min_rows=100
    )

    flipped = target.copy()
    flipped[folds == 0] = 1 - flipped[folds == 0]
    with_flipped, _ = cross_fitted_predictions(
        flipped, probabilities, folds, build_calibrator("platt"), min_rows=100
    )

    assert np.allclose(calibrated[folds == 0], with_flipped[folds == 0])
    assert not np.allclose(calibrated[folds != 0], with_flipped[folds != 0])


def test_cross_fitted_predictions_refuse_a_small_fit_set():
    target, probabilities, folds = _toy(n_folds=5, per_fold=20)
    with pytest.raises(ValueError, match="below min_rows"):
        cross_fitted_predictions(
            target, probabilities, folds, build_calibrator("platt"), min_rows=100
        )


def test_cross_fitted_predictions_require_both_classes_in_the_fit_rows():
    target, probabilities, folds = _toy()
    target = target.copy()
    target[folds != 0] = 0
    with pytest.raises(ValueError, match="both classes"):
        cross_fitted_predictions(
            target, probabilities, folds, build_calibrator("isotonic"), min_rows=100
        )


def test_summarize_calibration_keeps_none_as_the_identity():
    target, probabilities, folds = _toy()

    rows = summarize_calibration(
        target, probabilities, folds, calibrators=CALIBRATORS, min_rows=100, ece_bins=5
    )

    by_name = {row["calibrator"]: row for row in rows}
    assert set(by_name) == set(CALIBRATORS)
    assert by_name["none"]["brier_pooled"] == pytest.approx(brier_score_loss(target, probabilities))
    assert by_name["none"]["roc_auc_pooled"] == pytest.approx(roc_auc_score(target, probabilities))
    assert by_name["platt"]["brier_pooled"] < by_name["none"]["brier_pooled"]
    for row in rows:
        assert row["fit_rows_min"] == 1600
        assert row["n_folds"] == 5


def test_calibration_metrics_adds_log_loss_and_thresholds():
    target, probabilities, _ = _toy(per_fold=100)
    metrics = calibration_metrics(target, probabilities, thresholds=(0.2, 0.5), ece_bins=5)

    assert {"roc_auc", "brier", "ece", "log_loss", "precision", "recall"} <= set(metrics)
    assert "recall_at_0.2" in metrics


def test_recommend_calibrator_breaks_brier_ties_on_ece():
    rows = [
        {"arm": "a", "calibrator": "none", "brier_mean": 0.10, "ece_mean": 0.010},
        {"arm": "a", "calibrator": "platt", "brier_mean": 0.09, "ece_mean": 0.020},
        {"arm": "a", "calibrator": "isotonic", "brier_mean": 0.09, "ece_mean": 0.005},
    ]

    assert recommend_calibrator(rows)["a"]["calibrator"] == "isotonic"


def test_build_calibrator_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown calibrator"):
        build_calibrator("temperature")


def test_platt_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        build_calibrator("platt").fit(np.zeros(50, dtype="int8"), np.linspace(0.1, 0.9, 50))


def test_predictions_must_be_probabilities():
    calibrator = build_calibrator("none")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator.fit(np.array([0, 1]), np.array([0.2, 1.4]))
