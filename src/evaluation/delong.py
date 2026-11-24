"""DeLong's test for two correlated ROC-AUCs on the same rows.

Implements the O(n log n) midrank formulation (Sun & Xu, 2014) of DeLong et al.
(1988). E5 uses it to test whether two models' OOF AUCs differ on the same
development rows; the hold-out version belongs to Phase 9.

Usage
-----
``from src.evaluation.delong import delong_test``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeLongResult:
    """Paired AUC comparison for two score vectors."""

    auc_a: float
    auc_b: float
    difference: float
    z: float
    p_value: float


def _midranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based midranks, tied values sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype="float64")
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop

    out = np.empty(len(values), dtype="float64")
    out[order] = ranks
    return out


def _aucs_and_covariance(scores: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-model AUCs and their DeLong covariance.

    Input rows are already sorted with the positive class first and columns are
    one score vector per model.
    """
    n_models, n_rows = scores.shape
    m = int(label_1_count)
    n = n_rows - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs at least one row from each class")

    positive_ranks = np.empty((n_models, m), dtype="float64")
    negative_ranks = np.empty((n_models, n), dtype="float64")
    all_ranks = np.empty((n_models, n_rows), dtype="float64")
    for row in range(n_models):
        positive_ranks[row] = _midranks(scores[row, :m])
        negative_ranks[row] = _midranks(scores[row, m:])
        all_ranks[row] = _midranks(scores[row])

    aucs = all_ranks[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    # Per-positive and per-negative placement values (Sun & Xu section 2).
    v10 = (all_ranks[:, :m] - positive_ranks) / n
    v01 = 1.0 - (all_ranks[:, m:] - negative_ranks) / m
    covariance = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return aucs, covariance


def delong_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> DeLongResult:
    """Compare two ROC-AUCs computed on the same rows.

    Parameters
    ----------
    y_true:
        Binary labels, 0/1.
    score_a, score_b:
        Two score vectors aligned with ``y_true``; higher means more positive.

    Returns
    -------
    DeLongResult
        Both AUCs, their difference ``auc_a - auc_b``, the normal ``z`` and the
        two-sided p-value.
    """
    y = np.asarray(y_true).ravel()
    a = np.asarray(score_a, dtype="float64").ravel()
    b = np.asarray(score_b, dtype="float64").ravel()
    if not (len(y) == len(a) == len(b)):
        raise ValueError(f"lengths differ: y={len(y)}, a={len(a)}, b={len(b)}")

    classes = np.unique(y)
    if not np.all(np.isin(classes, (0, 1))):
        raise ValueError(f"y_true must be binary, got values {classes.tolist()}")
    if classes.size < 2:
        raise ValueError("DeLong needs both classes in y_true")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores must be finite")

    order = np.argsort(-y, kind="mergesort")  # positives first, ties keep row order
    scores = np.vstack([a[order], b[order]])
    aucs, covariance = _aucs_and_covariance(scores, int(y.sum()))

    difference = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if variance <= 0.0:
        # Identical or fully tied scores leave no variance to divide by.
        if difference == 0.0:
            return DeLongResult(float(aucs[0]), float(aucs[1]), 0.0, 0.0, 1.0)
        return DeLongResult(
            float(aucs[0]), float(aucs[1]), difference, math.copysign(math.inf, difference), 0.0
        )

    z = difference / math.sqrt(variance)
    return DeLongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        difference=difference,
        z=float(z),
        p_value=float(math.erfc(abs(z) / math.sqrt(2.0))),
    )
