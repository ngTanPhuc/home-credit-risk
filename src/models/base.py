"""Common model interface used by the CV runner.

Models must be pure estimators: no target encoding, resampling, or feature
selection belongs here. Those label-dependent steps are fitted inside each
training fold by dedicated fit/transform objects.
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
    ) -> BaseModel:
        """Fit the model on training-fold rows and return ``self``.

        ``sample_weight`` is aligned with ``target`` and is ``None`` when the
        imbalance variant does not weight rows.
        """

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for ``features``."""

    def feature_importance(self) -> pd.Series:
        """Return feature importances indexed by feature name, or an empty Series."""
        return pd.Series(dtype="float64", name="importance")
