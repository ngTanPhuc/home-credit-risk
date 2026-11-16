"""Metrics adapter: stable keys, threshold extras, ECE and degenerate inputs."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import all_metrics, compute_metrics


def test_perfect_predictions():
    labels = np.array([0, 0, 1, 1], dtype="int8")
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(labels, probabilities, thresholds=(0.5,), ece_bins=4)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["gini"] == pytest.approx(1.0)
    assert metrics["ks"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_keys_are_snake_case_and_thresholds_added():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200).astype("int8")
    probabilities = rng.random(200)
    metrics = compute_metrics(labels, probabilities, thresholds=(0.2, 0.5), ece_bins=5)

    for key in ("roc_auc", "gini", "ks", "pr_auc", "brier", "ece", "precision", "recall", "f1"):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert "precision_at_0.2" in metrics
    assert "recall_at_0.2" in metrics
    assert "precision_at_0.5" not in metrics  # 0.5 is already the base point
    assert all(isinstance(value, float) for value in metrics.values())


def test_all_metrics_unchanged_by_adapter():
    labels = np.array([0, 1, 0, 1, 1], dtype="int8")
    probabilities = np.array([0.1, 0.9, 0.4, 0.6, 0.55])
    base = all_metrics(labels, probabilities, threshold=0.5, n_bins=5)
    adapted = compute_metrics(labels, probabilities, thresholds=(0.5,), ece_bins=5)

    assert adapted["roc_auc"] == pytest.approx(base["ROC-AUC"])
    assert adapted["ece"] == pytest.approx(base["ECE"])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="lengths differ"):
        compute_metrics(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_no_positive_prediction_does_not_warn():
    labels = np.array([0, 1, 0, 1], dtype="int8")
    probabilities = np.array([0.01, 0.02, 0.03, 0.04])
    metrics = compute_metrics(labels, probabilities, thresholds=(0.9,), ece_bins=2)
    assert metrics["precision_at_0.9"] == pytest.approx(0.0)
    assert metrics["recall_at_0.9"] == pytest.approx(0.0)
