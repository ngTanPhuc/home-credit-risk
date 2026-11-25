"""Probability calibration for E6 and the final hold-out step.

Every calibrator here is a fit/transform object fitted on *out-of-fold*
development predictions only: a fold's calibrated probability is produced by a
calibrator fitted on the other folds' OOF rows, so no row is scored by a
calibrator that has seen its label. The final hold-out (Phase 9) fits the
chosen calibrator once on all development OOF rows and applies it to hold-out
probabilities; it never fits anything on hold-out labels.

Variants
--------
``none``
    Identity: the model's own probability, the before-calibration reference.
``platt``
    Logistic regression on the logit of the probability (a sigmoid on the
    decision value). Unregularised, so it is the maximum-likelihood sigmoid
    rather than Platt's smoothed-target variant.
``isotonic``
    Piecewise-constant monotone regression (:class:`IsotonicRegression`),
    clipped to ``[0, 1]`` at prediction time.

All three are monotone non-decreasing, so calibration changes the probability
scale (Brier, log loss, ECE, thresholds) and leaves ROC-AUC unchanged. That
invariant is what makes a rank metric immune to a bad calibrator, and it is
asserted in the tests.

Usage
-----
``from src.calibration import build_calibrator, cross_fitted_predictions``

``python -m src.evaluation.calibration_eval --config configs/experiments/E6_calibration.yaml``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.evaluation.metrics import compute_metrics

CALIBRATORS = ("none", "platt", "isotonic")

#: A calibrator fitted on fewer rows than this is refused: Platt on a few dozen
#: rows is noise, and isotonic needs enough distinct probabilities to step.
DEFAULT_MIN_ROWS = 200

_EPSILON = 1e-6


def logit(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``log(p / (1 - p))`` with the probabilities clipped away from 0 and 1."""
    values = np.clip(np.asarray(probabilities, dtype="float64").reshape(-1), _EPSILON, 1 - _EPSILON)
    return np.log(values / (1.0 - values))


def _check_predictions(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype="float64").reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot calibrate an empty probability vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities must be finite")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("Probabilities must lie in [0, 1]")
    return values


class Calibrator(ABC):
    """Fit/transform object mapping raw probabilities to calibrated ones."""

    name: str = "base"

    @abstractmethod
    def fit(self, target: Sequence[int] | np.ndarray, probabilities: Sequence[float]) -> Calibrator:
        """Fit on ``(target, probabilities)`` rows and return ``self``."""

    @abstractmethod
    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        """Apply the fitted mapping; never learns from the supplied values."""

    def fit_predict(
        self,
        target: Sequence[int] | np.ndarray,
        probabilities: Sequence[float],
        applied_to: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Fit on one probability vector and optionally apply to another."""
        self.fit(target, probabilities)
        return self.predict(probabilities if applied_to is None else applied_to)


class NoCalibration(Calibrator):
    """Reference: return the model probability unchanged."""

    name = "none"

    def fit(
        self, target: Sequence[int] | np.ndarray, probabilities: Sequence[float]
    ) -> NoCalibration:
        _check_predictions(probabilities)
        return self

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        return _check_predictions(probabilities)


class PlattCalibrator(Calibrator):
    """Unregularised logistic regression on the logit of the probability."""

    name = "platt"

    def __init__(self, *, max_iter: int = 1000) -> None:
        self.max_iter = int(max_iter)
        self._model: LogisticRegression | None = None
        self.coefficients: tuple[float, float] | None = None

    def fit(
        self, target: Sequence[int] | np.ndarray, probabilities: Sequence[float]
    ) -> PlattCalibrator:
        labels = np.asarray(target).reshape(-1).astype("int8")
        values = _check_predictions(probabilities)
        if labels.shape[0] != values.shape[0]:
            raise ValueError("target length does not match probabilities")
        if np.unique(labels).size < 2:
            raise ValueError("Platt scaling needs both classes in the fitting rows")

        self._model = LogisticRegression(C=1e6, max_iter=self.max_iter, solver="lbfgs")
        self._model.fit(logit(values).reshape(-1, 1), labels)
        self.coefficients = (
            float(self._model.coef_.ravel()[0]),
            float(self._model.intercept_.ravel()[0]),
        )
        return self

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator must be fitted before predict")
        values = _check_predictions(probabilities)
        return np.asarray(
            self._model.predict_proba(logit(values).reshape(-1, 1))[:, 1], dtype="float64"
        )


class IsotonicCalibrator(Calibrator):
    """Monotone piecewise-constant mapping clipped to ``[0, 1]``."""

    name = "isotonic"

    def __init__(self, *, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = str(out_of_bounds)
        self._model: IsotonicRegression | None = None

    def fit(
        self, target: Sequence[int] | np.ndarray, probabilities: Sequence[float]
    ) -> IsotonicCalibrator:
        labels = np.asarray(target).reshape(-1).astype("float64")
        values = _check_predictions(probabilities)
        if labels.shape[0] != values.shape[0]:
            raise ValueError("target length does not match probabilities")
        if np.unique(labels).size < 2:
            raise ValueError("isotonic calibration needs both classes in the fitting rows")

        self._model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=self.out_of_bounds
        )
        self._model.fit(values, labels)
        return self

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator must be fitted before predict")
        values = _check_predictions(probabilities)
        return np.asarray(self._model.predict(values), dtype="float64").clip(0.0, 1.0)


_CLASSES: dict[str, type[Calibrator]] = {
    "none": NoCalibration,
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def build_calibrator(name: str | None) -> Calibrator:
    """Return a fresh calibrator for a config value (``none`` when missing)."""
    key = str(name or "none").strip().lower()
    try:
        return _CLASSES[key]()
    except KeyError:
        raise ValueError(f"Unknown calibrator {name!r}; known: {list(CALIBRATORS)}") from None


def cross_fitted_predictions(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    folds: Sequence[int] | np.ndarray,
    calibrator: Calibrator,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Calibrate every fold with a calibrator fitted on the other folds only.

    Returns
    -------
    tuple
        ``(calibrated, fold_details)``. ``calibrated`` is aligned with the input
        rows. ``fold_details`` records the rows used to fit each fold's
        calibrator so a report can show the protocol was respected.

    Raises
    ------
    ValueError
        If a fitting set is smaller than ``min_rows`` or has a single class.
    """
    labels = np.asarray(target).reshape(-1).astype("int8")
    values = _check_predictions(probabilities)
    fold_ids = np.asarray(folds).reshape(-1)
    if not (labels.shape[0] == values.shape[0] == fold_ids.shape[0]):
        raise ValueError("target, probabilities and folds must have the same length")

    calibrated = np.empty(values.shape[0], dtype="float64")
    details: dict[int, dict[str, Any]] = {}

    for fold in np.unique(fold_ids):
        fitting = fold_ids != fold
        applied = ~fitting
        n_fitting = int(fitting.sum())
        if n_fitting < min_rows:
            raise ValueError(
                f"Fold {fold}: calibrator fit on {n_fitting} rows, below min_rows={min_rows}"
            )
        if np.unique(labels[fitting]).size < 2:
            raise ValueError(f"Fold {fold}: calibration rows need both classes")

        calibrated[applied] = calibrator.fit(labels[fitting], values[fitting]).predict(
            values[applied]
        )
        details[int(fold)] = {
            "n_fit": n_fitting,
            "n_applied": int(applied.sum()),
            "calibrator": calibrator.name,
        }

    return calibrated, details


def calibration_metrics(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    thresholds: Sequence[float] = (0.5,),
    ece_bins: int = 15,
) -> dict[str, float]:
    """Return the shared metrics plus log loss for one probability vector."""
    labels = np.asarray(target).reshape(-1).astype("int8")
    values = _check_predictions(probabilities)
    metrics = dict(compute_metrics(labels, values, thresholds=thresholds, ece_bins=ece_bins))
    metrics["log_loss"] = float(log_loss(labels, np.clip(values, _EPSILON, 1 - _EPSILON)))
    return metrics


def reliability_curve(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    bins: int = 15,
) -> dict[str, np.ndarray]:
    """Bin rows by predicted probability for a reliability diagram.

    Bins are equal-count (quantile) bins, the same definition the ECE metric
    uses, so the plotted points and the reported ECE describe the same partition.

    Returns
    -------
    dict
        ``predicted`` and ``observed`` per bin, plus ``count``.
    """
    labels = np.asarray(target).reshape(-1).astype("float64")
    values = _check_predictions(probabilities)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("target length does not match probabilities")
    if bins < 1:
        raise ValueError("bins must be positive")

    order = np.argsort(values, kind="stable")
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for chunk in np.array_split(order, int(bins)):
        if chunk.size == 0:
            continue
        predicted.append(float(values[chunk].mean()))
        observed.append(float(labels[chunk].mean()))
        counts.append(int(chunk.size))
    return {
        "predicted": np.asarray(predicted, dtype="float64"),
        "observed": np.asarray(observed, dtype="float64"),
        "count": np.asarray(counts, dtype="int64"),
    }


def summarize_calibration(
    target: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    folds: Sequence[int] | np.ndarray,
    *,
    calibrators: Sequence[str] = CALIBRATORS,
    min_rows: int = DEFAULT_MIN_ROWS,
    thresholds: Sequence[float] = (0.5,),
    ece_bins: int = 15,
) -> list[dict[str, Any]]:
    """Return one row per calibrator with pooled and per-fold cross-fitted metrics.

    Metrics are computed on the cross-fitted calibrated probabilities, so a
    calibrator that memorises its fitting rows does not look good here.
    """
    labels = np.asarray(target).reshape(-1).astype("int8")
    values = _check_predictions(probabilities)
    fold_ids = np.asarray(folds).reshape(-1)

    rows: list[dict[str, Any]] = []
    for name in calibrators:
        calibrator = build_calibrator(name)
        calibrated, details = cross_fitted_predictions(
            labels, values, fold_ids, calibrator, min_rows=min_rows
        )
        pooled = calibration_metrics(labels, calibrated, thresholds=thresholds, ece_bins=ece_bins)
        folded = [
            calibration_metrics(
                labels[fold_ids == fold],
                calibrated[fold_ids == fold],
                thresholds=thresholds,
                ece_bins=ece_bins,
            )
            for fold in np.unique(fold_ids)
        ]
        row: dict[str, Any] = {
            "calibrator": calibrator.name,
            "n_rows": int(labels.shape[0]),
            "n_folds": len(folded),
            "fit_rows_min": min(item["n_fit"] for item in details.values()),
        }
        for metric in ("roc_auc", "ks", "pr_auc", "brier", "ece", "log_loss"):
            row[f"{metric}_pooled"] = float(pooled[metric])
            per_fold = np.array([item[metric] for item in folded], dtype="float64")
            row[f"{metric}_mean"] = float(per_fold.mean())
            row[f"{metric}_std"] = float(per_fold.std(ddof=0))
        rows.append(row)
    return rows


def calibration_by_arm(
    arms: Mapping[str, Mapping[str, np.ndarray]],
    folds: Sequence[int] | np.ndarray,
    *,
    calibrators: Sequence[str] = CALIBRATORS,
    min_rows: int = DEFAULT_MIN_ROWS,
    thresholds: Sequence[float] = (0.5,),
    ece_bins: int = 15,
) -> list[dict[str, Any]]:
    """Score every arm under every calibrator, one row per arm/calibrator pair.

    ``arms`` maps an arm name to the arrays ``target`` and ``y_prob``, which is
    the same shape the E5 driver already builds from OOF files.
    """
    rows: list[dict[str, Any]] = []
    for arm, payload in arms.items():
        for row in summarize_calibration(
            payload["target"],
            payload["y_prob"],
            folds,
            calibrators=calibrators,
            min_rows=min_rows,
            thresholds=thresholds,
            ece_bins=ece_bins,
        ):
            rows.append({"arm": str(arm), **row})
    return rows


def recommend_calibrator(
    rows: Sequence[Mapping[str, Any]],
    *,
    brier_key: str = "brier_mean",
    ece_key: str = "ece_mean",
) -> dict[str, dict[str, Any]]:
    """Pick the lowest-Brier calibrator per arm, breaking ties on ECE.

    The uncalibrated row is kept in the comparison; an arm whose model is
    already calibrated therefore recommends ``none`` rather than adding a
    mapping that does not help.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        arm = str(row["arm"])
        candidate = (float(row[brier_key]), float(row[ece_key]))
        current = best.get(arm)
        if current is None or candidate < (float(current[brier_key]), float(current[ece_key])):
            best[arm] = dict(row)
    return best
