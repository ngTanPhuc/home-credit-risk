"""DeLong paired AUC test (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.delong import delong_test


def _toy_problem(n_rows: int = 400, seed: int = 11):
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < 0.3).astype("int8")
    latent = y + rng.normal(0.0, 1.2, n_rows)
    return y, latent, latent + rng.normal(0.0, 0.05, n_rows)


def _bruteforce_variance(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """DeLong variance of the AUC difference, straight from the definition."""
    positives = y == 1
    negatives = ~positives
    m, n = int(positives.sum()), int(negatives.sum())

    def placements(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, neg = scores[positives], scores[negatives]
        greater = (pos[:, None] > neg[None, :]).astype("float64")
        tied = (pos[:, None] == neg[None, :]).astype("float64")
        psi = greater + 0.5 * tied
        return psi.mean(axis=1), psi.mean(axis=0)  # per positive, per negative

    v10_a, v01_a = placements(a)
    v10_b, v01_b = placements(b)
    covariance = np.cov(np.vstack([v10_a, v10_b])) / m + np.cov(np.vstack([v01_a, v01_b])) / n
    return float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])


def test_aucs_match_sklearn_and_variance_matches_bruteforce():
    y, a, b = _toy_problem()
    # Round to force ties through the midrank path.
    a = np.round(a, 1)
    b = np.round(b, 1)

    result = delong_test(y, a, b)

    assert result.auc_a == pytest.approx(roc_auc_score(y, a))
    assert result.auc_b == pytest.approx(roc_auc_score(y, b))
    assert result.difference == pytest.approx(result.auc_a - result.auc_b)
    implied_variance = (result.difference / result.z) ** 2
    assert implied_variance == pytest.approx(_bruteforce_variance(y, a, b), rel=1e-9)


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    a = np.array([0.1, 0.2, 0.8, 0.9])
    b = np.array([0.4, 0.3, 0.2, 0.1])

    result = delong_test(y, a, b)

    assert result.auc_a == pytest.approx(1.0)
    assert result.auc_b == pytest.approx(0.0)
    assert result.z > 0
    assert result.p_value < 0.05


def test_identical_scores_are_not_significant():
    y, a, _ = _toy_problem()
    result = delong_test(y, a, a.copy())

    assert result.difference == 0.0
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_large_difference_is_detected():
    rng = np.random.default_rng(5)
    y = (rng.random(500) < 0.3).astype("int8")
    strong = y + rng.normal(0.0, 0.8, size=y.size)
    weak = rng.normal(0.0, 1.0, size=y.size)

    result = delong_test(y, strong, weak)

    assert result.z > 0
    assert result.p_value < 0.01


def test_swapping_scores_flips_the_sign_and_keeps_the_p_value():
    y, a, b = _toy_problem()
    forward = delong_test(y, a, b)
    backward = delong_test(y, b, a)

    assert backward.z == pytest.approx(-forward.z)
    assert backward.p_value == pytest.approx(forward.p_value)
    assert backward.auc_a == pytest.approx(forward.auc_b)


def test_rejects_single_class_and_mismatched_lengths():
    with pytest.raises(ValueError, match="both classes"):
        delong_test(np.zeros(10), np.arange(10.0), np.arange(10.0))

    with pytest.raises(ValueError, match="lengths differ"):
        delong_test(np.array([0, 1, 0]), np.arange(3.0), np.arange(2.0))
