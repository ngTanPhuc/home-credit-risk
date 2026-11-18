"""LightGBM wrapper.

Phase 2 uses strict LightGBM defaults: callers must not tune hyperparameters
through this module's runtime block. Categorical handling is fitted on the
training fold only (categories are derived from training-fold values), so
validation data cannot influence encoding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.models.base import BaseModel


class LGBMModel(BaseModel):
    """Thin sklearn-style LightGBM classifier with train-fold category handling."""

    name = "lightgbm"

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int = 42,
        n_jobs: int = -1,
        deterministic: bool = True,
        force_col_wise: bool = True,
    ) -> None:
        base_params: dict[str, Any] = {
            "objective": "binary",
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
            "verbosity": -1,
        }
        if deterministic:
            base_params["deterministic"] = True
        if force_col_wise:
            base_params["force_col_wise"] = True

        # Runtime settings are defaults; explicit model params may override them.
        self.params = {**base_params, **dict(params or {})}
        self.seed = int(seed)
        self._model: LGBMClassifier | None = None
        self._categorical_features: list[str] = []
        self._categories: dict[str, pd.Index] = {}
        self._feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
        sample_weight: np.ndarray | Sequence[float] | None = None,
        validation: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> LGBMModel:
        """Fit on training-fold rows.

        Parameters
        ----------
        features:
            Training-fold feature frame (no IDs or target).
        target:
            Binary target aligned with ``features``.
        categorical_features:
            Columns to treat as categorical. When ``None``, all object/string
            columns are detected automatically.
        sample_weight:
            Optional per-row weights aligned with ``target``.
        validation:
            Optional same-fold validation pair used for early stopping (E4).
        early_stopping_rounds:
            Required with ``validation``; rounds without improvement.

        Notes
        -----
        Category vocabularies are learned from this training fold only and
        reused unchanged at prediction time.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty:
            raise ValueError("Cannot fit on an empty feature frame")

        labels = np.asarray(target).reshape(-1)
        if labels.shape[0] != len(features):
            raise ValueError("target length does not match features")
        if not set(np.unique(labels).tolist()).issubset({0, 1}):
            raise ValueError("target must be binary")
        if len(np.unique(labels)) < 2:
            raise ValueError("Training fold must contain both classes")

        frame = features.copy()
        self._categorical_features = self._resolve_categorical_features(frame, categorical_features)
        frame = self._encode_categories(frame, fit=True)
        self._feature_names = list(frame.columns)

        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype="float64").reshape(-1)
            if weights.shape[0] != labels.shape[0]:
                raise ValueError("sample_weight length does not match target")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("sample_weight must be finite and non-negative")

        self._model = LGBMClassifier(**self.params)
        eval_X = None
        eval_y = None
        callbacks = None
        if validation is not None:
            if early_stopping_rounds is None:
                raise ValueError("validation requires early_stopping_rounds")
            valid_features, valid_target = validation
            valid_frame = valid_features.loc[:, self._feature_names].copy()
            valid_frame = self._encode_categories(valid_frame, fit=False)
            eval_X = valid_frame
            eval_y = np.asarray(valid_target).reshape(-1)
            callbacks = [early_stopping(int(early_stopping_rounds), verbose=False)]

        self._model.fit(
            frame,
            labels,
            sample_weight=weights,
            eval_X=eval_X,
            eval_y=eval_y,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities."""
        if self._model is None:
            raise RuntimeError("Model must be fitted before predict_proba")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")

        frame = features.loc[:, self._feature_names].copy()
        frame = self._encode_categories(frame, fit=False)
        probabilities = self._model.predict_proba(frame)
        return np.asarray(probabilities[:, 1], dtype="float64")

    def feature_importance(self) -> pd.Series:
        """Return gain-based feature importance, sorted descending."""
        if self._model is None:
            return pd.Series(dtype="float64", name="importance")
        booster = self._model.booster_
        values = booster.feature_importance(importance_type="gain")
        return (
            pd.Series(values, index=booster.feature_name(), name="importance")
            .astype("float64")
            .sort_values(ascending=False)
        )

    @property
    def best_iteration(self) -> int | None:
        """Iteration chosen by the early-stopping callback, if any."""
        if self._model is None:
            return None
        best = getattr(self._model, "best_iteration_", None)
        return int(best) if best else None

    @staticmethod
    def _resolve_categorical_features(
        features: pd.DataFrame,
        categorical_features: Sequence[str] | None,
    ) -> list[str]:
        if categorical_features is None:
            return list(features.select_dtypes(include=["object", "string", "category"]).columns)

        missing = [column for column in categorical_features if column not in features.columns]
        if missing:
            raise KeyError(f"Categorical columns not found in features: {missing}")
        return list(categorical_features)

    def _encode_categories(self, features: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        for column in self._categorical_features:
            values = features[column].astype("object")
            if fit:
                unique = pd.Index(values.dropna().unique())
                try:
                    unique = unique.sort_values()
                except TypeError:
                    pass
                self._categories[column] = unique
            categories = self._categories[column]
            features[column] = pd.Categorical(values, categories=categories)
        return features
