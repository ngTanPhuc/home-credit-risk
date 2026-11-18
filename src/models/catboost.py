"""CatBoost wrapper with native categorical handling.

CatBoost receives the contract's categorical columns as strings and splits its
own ordered target statistics inside each fitted model. No encoding is learned
outside the current training fold, so validation rows cannot influence it.

CatBoost rejects NaN in categorical columns, so missing values become the
explicit ``__MISSING__`` level; numeric NaN is left to CatBoost's own missing
value handling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost classifier using the native categorical split strategy."""

    name = "catboost"

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int = 42,
        thread_count: int = -1,
    ) -> None:
        base_params: dict[str, Any] = {
            "loss_function": "Logloss",
            "random_seed": int(seed),
            "thread_count": int(thread_count),
            "verbose": False,
            "allow_writing_files": False,
        }
        self.params = {**base_params, **dict(params or {})}
        self.seed = int(seed)
        self._model: CatBoostClassifier | None = None
        self._categorical_features: list[str] = []
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
    ) -> CatBoostModel:
        """Fit on training-fold rows with native categorical features."""
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

        self._categorical_features = self._resolve_categorical_features(
            features, categorical_features
        )
        self._feature_names = list(features.columns)
        frame = self._transform(features)

        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype="float64").reshape(-1)
            if weights.shape[0] != labels.shape[0]:
                raise ValueError("sample_weight length does not match target")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("sample_weight must be finite and non-negative")

        params = dict(self.params)
        eval_set = None
        if validation is not None or early_stopping_rounds is not None:
            if validation is None or early_stopping_rounds is None:
                raise ValueError("validation and early_stopping_rounds must be given together")
            valid_features, valid_target = validation
            eval_set = (self._transform(valid_features), np.asarray(valid_target).reshape(-1))
            params.update(
                {
                    "od_type": "Iter",
                    "od_wait": int(early_stopping_rounds),
                    "use_best_model": True,
                }
            )

        self._model = CatBoostClassifier(**params)
        self._model.fit(
            frame,
            labels,
            cat_features=self._categorical_features,
            sample_weight=weights,
            eval_set=eval_set,
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities."""
        if self._model is None:
            raise RuntimeError("Model must be fitted before predict_proba")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        probabilities = self._model.predict_proba(self._transform(features))
        return np.asarray(probabilities[:, 1], dtype="float64")

    def feature_importance(self) -> pd.Series:
        """Return CatBoost feature importance, sorted descending."""
        if self._model is None:
            return pd.Series(dtype="float64", name="importance")
        values = self._model.get_feature_importance()
        return (
            pd.Series(values, index=self._feature_names, name="importance")
            .astype("float64")
            .sort_values(ascending=False)
        )

    @property
    def best_iteration(self) -> int | None:
        """Zero-based best iteration chosen by early stopping, if any."""
        if self._model is None:
            return None
        best = self._model.get_best_iteration()
        return int(best) if best else None

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = features.loc[:, self._feature_names].copy().reset_index(drop=True)
        for column in self._categorical_features:
            values = frame[column].astype("object").fillna("__MISSING__").astype("str")
            frame[column] = values.to_numpy()
        return frame

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
