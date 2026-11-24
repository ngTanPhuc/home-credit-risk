"""Full-development fit shared by the final hold-out and the explanation steps.

E1-E5 always fit inside one training fold. Once the protocol reaches the final
stage there is no further model selection, so one fit on *all* development rows
is the right estimator. This module keeps that fit in one place so the hold-out
script and the SHAP script cannot drift apart: both go through the E3 imbalance
strategy object and the model factory, and neither ever sees a hold-out label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.imbalance import build_strategy, spec_from_config
from src.models import build_model
from src.models.base import BaseModel


def fit_development_model(
    config: Mapping[str, Any],
    model_block: Mapping[str, Any],
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    categorical_features: Sequence[str] | None = None,
    seed: int = 42,
    imbalance: Any = None,
) -> BaseModel:
    """Fit one model on every development row.

    Parameters
    ----------
    config:
        Resolved experiment config; its ``imbalance`` block selects the strategy
        unless ``imbalance`` is given explicitly.
    model_block:
        One ``comparison.models`` entry (``name``, ``params``, ``runtime``, ...).
        Callers that come from E4 pass the ``final_params`` version.
    features, target:
        Development rows only. Passing hold-out rows here would leak the final
        evaluation into the fitted model; callers load them separately.
    categorical_features:
        Contract categorical columns, forwarded to the model wrapper.
    seed:
        Model seed; the imbalance strategy is rebuilt with the same value.
    imbalance:
        Optional override for the ``imbalance`` config block.

    Returns
    -------
    BaseModel
        A fitted model exposing ``predict_proba``, ``feature_importance`` and
        ``shap_contributions``.
    """
    if features.empty:
        raise ValueError("Cannot fit the final model on an empty development frame")
    labels = np.asarray(target).reshape(-1).astype("int8")
    if labels.shape[0] != len(features):
        raise ValueError("target length does not match development features")

    specification = spec_from_config(config) if imbalance is None else imbalance
    strategy = build_strategy(specification, seed=seed)
    train_features, train_target, sample_weight = strategy.fit_resample(
        features, labels, categorical_features=categorical_features
    )

    model = build_model(model_block, seed=seed)
    model.fit(
        train_features,
        train_target,
        categorical_features=categorical_features,
        sample_weight=sample_weight,
    )
    return model
