"""Common model interface used by the CV runner.

Models must be pure estimators: resampling and feature selection belong to
dedicated fit/transform objects. A model may encode its own categorical columns
inside :meth:`BaseModel.fit` (XGBoost target encoding), because ``fit`` only
receives the rows of one training fold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Minimal binary-classification estimator interface."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
        sample_weight: np.ndarray | Sequence[float] | None = None,
        validation: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> BaseModel:
        """Fit the model on training-fold rows and return ``self``.

        ``sample_weight`` is aligned with ``target`` and is ``None`` when the
        imbalance variant does not weight rows.

        ``validation`` is an optional ``(features, target)`` pair from the same
        fold, used by E4 early stopping. It is never used to fit the model. The
        CV runner leaves it ``None``, so final E5/E6 runs use a fixed iteration
        count instead of selecting one on the validation fold.
        """

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for ``features``."""

    def feature_importance(self) -> pd.Series:
        """Return feature importances indexed by feature name, or an empty Series."""
        return pd.Series(dtype="float64", name="importance")

    def shap_contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return positive-class TreeSHAP contributions, one column per input feature.

        Rows keep ``features.index``. The values are additive log-odds
        contributions from each library's own exact TreeSHAP implementation;
        the base value (expected log-odds) is not included. Backends that do not
        implement this raise :class:`NotImplementedError`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement SHAP contributions")

    @property
    def best_iteration(self) -> int | None:
        """Early-stopping iteration selected in :meth:`fit`, or ``None``."""
        return None
