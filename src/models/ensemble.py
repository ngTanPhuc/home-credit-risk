"""Cross-fitted OOF ensembling for E5: weighted average and logistic stacking.

Both arms are fitted fold by fold on the *other* folds' out-of-fold
predictions and applied to the held-out fold, so no row contributes to its own
ensemble prediction. The E5 driver (``src/evaluation/model_comparison.py``) is
the only caller.

Usage
-----
``from src.models.ensemble import align_oof, cross_fitted_average, cross_fitted_stacking``
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class OOFMatrix:
    """OOF predictions of several models aligned on one frozen split."""

    meta: pd.DataFrame  # identifier, target and fold, sorted by identifier
    probabilities: pd.DataFrame  # one column per model, index aligned with meta
    identifier_column: str
    target_column: str
    fold_column: str


@dataclass(frozen=True)
class EnsemblePrediction:
    """One ensemble arm's OOF probabilities plus the fitted weights."""

    name: str
    y_prob: np.ndarray
    weights: Mapping[str, float] | None = None
    fold_weights: Mapping[int, Mapping[str, float]] | None = None


def align_oof(
    frames: Mapping[str, pd.DataFrame],
    *,
    identifier_column: str = "SK_ID_CURR",
    target_column: str = "TARGET",
    fold_column: str = "fold",
    probability_column: str = "y_prob",
) -> OOFMatrix:
    """Align per-model OOF frames on the identifier and validate the shared split.

    Every frame must contain one row per development applicant and the same
    target and fold values; a model trained on a different split is rejected
    instead of silently blended.
    """
    if not frames:
        raise ValueError("No OOF frames supplied")

    meta: pd.DataFrame | None = None
    reference_ids: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}

    for name, frame in frames.items():
        required = [identifier_column, target_column, fold_column, probability_column]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"OOF frame {name!r} is missing columns: {missing}")

        table = frame.loc[:, required].copy()
        table[probability_column] = pd.to_numeric(table[probability_column], errors="raise").astype(
            "float64"
        )
        if table[identifier_column].duplicated().any():
            raise ValueError(f"OOF frame {name!r} has duplicated identifiers")
        if not np.isfinite(table[probability_column]).all():
            raise ValueError(f"OOF frame {name!r} has non-finite probabilities")

        table = table.sort_values(identifier_column).reset_index(drop=True)
        ids = table[identifier_column].to_numpy()
        if meta is None:
            meta = table.loc[:, [identifier_column, target_column, fold_column]]
            reference_ids = ids
        else:
            if not np.array_equal(ids, reference_ids):
                raise ValueError(
                    f"OOF frame {name!r} covers a different row set; "
                    "every model must predict the same frozen split"
                )
            for column in (target_column, fold_column):
                if not np.array_equal(table[column].to_numpy(), meta[column].to_numpy()):
                    raise ValueError(f"OOF frame {name!r} has different {column!r} values")
        columns[name] = table[probability_column].to_numpy()

    assert meta is not None  # frames is non-empty
    probabilities = pd.DataFrame(columns, index=meta.index)
    return OOFMatrix(
        meta=meta,
        probabilities=probabilities,
        identifier_column=identifier_column,
        target_column=target_column,
        fold_column=fold_column,
    )


def _compositions(total: int, parts: int):
    """Yield every way to split ``total`` units into ``parts`` non-negative counts."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _simplex_grid(n_models: int, step: float) -> np.ndarray:
    """Return every weight vector on the simplex on a fixed grid."""
    if n_models < 1:
        raise ValueError("At least one model is required")
    units = int(round(1.0 / step))
    if units < 1 or not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError(f"weight step must divide 1.0 exactly, got {step}")
    return np.asarray(list(_compositions(units, n_models)), dtype="float64") / units


def fit_average_weights(
    y_true: np.ndarray,
    probabilities: pd.DataFrame,
    *,
    step: float = 0.05,
) -> dict[str, float]:
    """Return simplex weights that minimize log loss on the supplied rows.

    The grid is deterministic, so the blend is reproducible without a solver
    dependency. E5 fits it inside each fold on the other folds' OOF rows.
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    if probabilities.empty:
        raise ValueError("Cannot fit weights on an empty prediction frame")
    if len(y) != len(probabilities):
        raise ValueError(f"target length {len(y)} does not match predictions {len(probabilities)}")
    if np.unique(y).size < 2:
        raise ValueError("weight fitting needs both classes in the supplied rows")

    names = [str(name) for name in probabilities.columns]
    if len(names) == 1:
        return {names[0]: 1.0}

    values = probabilities.to_numpy(dtype="float64")
    best_weights: np.ndarray | None = None
    best_loss = math.inf
    for weights in _simplex_grid(len(names), step):
        blended = np.clip(values @ weights, 1e-15, 1.0 - 1e-15)
        loss = -float(np.mean(y * np.log(blended) + (1.0 - y) * np.log1p(-blended)))
        if loss < best_loss - 1e-12:
            best_loss, best_weights = loss, weights

    assert best_weights is not None
    return {name: float(weight) for name, weight in zip(names, best_weights, strict=True)}


def cross_fitted_average(
    oof: OOFMatrix,
    *,
    step: float = 0.05,
) -> EnsemblePrediction:
    """Blend the models with weights fitted on the other folds only."""
    probabilities = oof.probabilities
    y = oof.meta[oof.target_column].to_numpy()
    folds = oof.meta[oof.fold_column].to_numpy()
    names = [str(name) for name in probabilities.columns]

    prediction = np.empty(len(oof.meta), dtype="float64")
    fold_weights: dict[int, dict[str, float]] = {}
    for fold in np.unique(folds):
        train = folds != fold
        valid = ~train
        weights = fit_average_weights(y[train], probabilities.loc[train], step=step)
        prediction[valid] = probabilities.loc[valid].to_numpy() @ np.array(
            [weights[name] for name in names], dtype="float64"
        )
        fold_weights[int(fold)] = weights

    return EnsemblePrediction(
        name="weighted_average",
        y_prob=prediction,
        weights=fit_average_weights(y, probabilities, step=step),
        fold_weights=fold_weights,
    )


def cross_fitted_stacking(
    oof: OOFMatrix,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> EnsemblePrediction:
    """Fit a logistic meta-learner on other folds' OOF rows, predict the held-out fold."""
    probabilities = oof.probabilities
    y = oof.meta[oof.target_column].to_numpy()
    folds = oof.meta[oof.fold_column].to_numpy()

    prediction = np.empty(len(oof.meta), dtype="float64")
    for fold in np.unique(folds):
        train = folds != fold
        valid = ~train
        if np.unique(y[train]).size < 2:
            raise ValueError(f"Fold {fold}: stacking needs both classes in the meta-training rows")
        meta_model = LogisticRegression(max_iter=int(max_iter), random_state=int(seed))
        meta_model.fit(probabilities.loc[train], y[train])
        prediction[valid] = meta_model.predict_proba(probabilities.loc[valid])[:, 1]

    return EnsemblePrediction(name="stacking", y_prob=prediction)
