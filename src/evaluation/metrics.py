import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def all_metrics(y, p, threshold=0.5, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)

    auc = roc_auc_score(y, p)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y, p)
    ks = np.max(tpr - fpr)
    pr_auc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    # ECE with equal-count (quantile) bins
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    ece = sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins)

    pred = (p >= threshold).astype(int)
    return {
        "ROC-AUC": auc,
        "Gini": gini,
        "KS": ks,
        "PR-AUC": pr_auc,
        "PR-AUC baseline": y.mean(),
        "Brier": brier,
        "Brier baseline": y.mean() * (1 - y.mean()),
        "ECE": ece,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


# Canonical snake_case names used by the CV runner, MLflow and report tables.
_KEY_MAP = {
    "ROC-AUC": "roc_auc",
    "Gini": "gini",
    "KS": "ks",
    "PR-AUC": "pr_auc",
    "PR-AUC baseline": "pr_auc_baseline",
    "Brier": "brier",
    "Brier baseline": "brier_baseline",
    "ECE": "ece",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
}


def compute_metrics(y, p, thresholds=(0.5,), ece_bins=15):
    """Return all_metrics under stable snake_case keys plus per-threshold scores.

    Parameters
    ----------
    y:
        Binary labels.
    p:
        Predicted positive-class probabilities.
    thresholds:
        Decision thresholds to report precision/recall/F1 for. The 0.5 point is
        already covered by ``all_metrics`` and is not duplicated.
    ece_bins:
        Number of equal-count bins used for ECE.

    Returns
    -------
    dict[str, float]
        AUC/Gini/KS/PR-AUC/Brier/ECE at 0.5 plus ``*_at_<t>`` extras.
    """
    y = np.asarray(y).astype("int8").reshape(-1)
    p = np.asarray(p, dtype="float64").reshape(-1)
    if y.shape != p.shape:
        raise ValueError(f"label and probability lengths differ: {y.shape} vs {p.shape}")

    metrics = {
        _KEY_MAP[key]: float(value)
        for key, value in all_metrics(y, p, threshold=0.5, n_bins=ece_bins).items()
    }

    for threshold in thresholds:
        value = float(threshold)
        if value == 0.5:
            continue
        predicted = (p >= value).astype("int8")
        suffix = f"at_{value:g}"
        metrics[f"precision_{suffix}"] = float(precision_score(y, predicted, zero_division=0))
        metrics[f"recall_{suffix}"] = float(recall_score(y, predicted, zero_division=0))
        metrics[f"f1_{suffix}"] = float(f1_score(y, predicted, zero_division=0))
    return metrics
