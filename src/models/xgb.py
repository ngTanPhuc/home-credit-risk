"""XGBoost wrapper with leakage-safe out-of-fold target encoding.

XGBoost's native categorical support is avoided on purpose: the contract has 16
categorical columns with high cardinality, and the project decides the encoding
per fold. Categorical columns are target-encoded inside :meth:`XGBModel.fit`
from this training fold only.

A training row's encoding never uses its own label: the training fold is split
into a nested stratified K-fold, and each row is encoded from the other nested
parts. Validation and prediction rows use the full training-fold statistics,
smoothed toward the training-fold prior. Unseen categories at prediction time
fall back to that prior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.models.base import BaseModel


class XGBModel(BaseModel):
    """XGBoost classifier with per-fold categorical target encoding."""

    name = "xgboost"

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int = 42,
        n_jobs: int = -1,
        target_encoding: Mapping[str, Any] | None = None,
    ) -> None:
        base_params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
        }
        self.params = {**base_params, **dict(params or {})}
        encoding = dict(target_encoding or {})
        self.smoothing = float(encoding.get("smoothing", 10.0))
        self.n_splits = int(encoding.get("n_splits", 5))
        if self.smoothing < 0:
            raise ValueError("target_encoding.smoothing must be non-negative")
        if self.n_splits < 2:
            raise ValueError("target_encoding.n_splits must be at least 2")

        self.seed = int(seed)
        self._model: XGBClassifier | None = None
        self._categorical_features: list[str] = []
        self._feature_names: list[str] = []
        self._encoding_maps: dict[str, pd.Series] = {}
        self._prior = 0.0

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
        sample_weight: np.ndarray | Sequence[float] | None = None,
        validation: tuple[pd.DataFrame, pd.Series | np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> XGBModel:
        """Fit on training-fold rows and learn the fold's category encodings."""
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

        frame = features.copy().reset_index(drop=True)
        self._categorical_features = self._resolve_categorical_features(frame, categorical_features)
        self._feature_names = list(frame.columns)
        encoded = self._learn_encodings(frame, labels)

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
            labels_valid = np.asarray(valid_target).reshape(-1)
            if len(labels_valid) != len(valid_features):
                raise ValueError("validation target length does not match validation features")
            eval_set = [(self._transform(valid_features), labels_valid)]
            params["early_stopping_rounds"] = int(early_stopping_rounds)

        self._model = XGBClassifier(**params)
        self._model.fit(
            encoded,
            labels,
            sample_weight=weights,
            eval_set=eval_set,
            verbose=False,
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
        """Return gain-based feature importance, sorted descending."""
        if self._model is None:
            return pd.Series(dtype="float64", name="importance")
        values = self._model.feature_importances_
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
        best = getattr(self._model, "best_iteration", None)
        return int(best) if best else None

    def _learn_encodings(self, frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Return the training frame encoded OOF, and store maps for prediction."""
        self._prior = float(labels.mean())
        self._encoding_maps = {}
        encoded = frame.copy()
        if not self._categorical_features:
            return encoded

        counts = np.bincount(labels.astype("int64"), minlength=2)
        n_splits = min(self.n_splits, int(counts.min()))
        if n_splits < 2:
            raise ValueError(
                "target encoding needs at least two rows per class in the training fold"
            )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        for column in self._categorical_features:
            values = self._category_values(frame[column])
            oof = pd.Series(np.nan, index=frame.index, dtype="float64")
            for fit_index, hold_index in splitter.split(values, labels):
                means = self._smoothed_means(values.iloc[fit_index], labels[fit_index])
                oof.iloc[hold_index] = values.iloc[hold_index].map(means).to_numpy()
            encoded[column] = oof.fillna(self._prior).astype("float64")
            self._encoding_maps[column] = self._smoothed_means(values, labels)
        return encoded

    def _transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-fold encoding maps; never learns from ``features``."""
        frame = features.loc[:, self._feature_names].copy().reset_index(drop=True)
        for column in self._categorical_features:
            values = self._category_values(frame[column])
            frame[column] = values.map(self._encoding_maps[column]).fillna(self._prior)
            frame[column] = frame[column].astype("float64")
        return frame

    def _smoothed_means(self, values: pd.Series, labels: np.ndarray) -> pd.Series:
        """Category means smoothed toward the prior: (sum + m*p) / (count + m)."""
        prior = float(np.mean(labels))
        frame = pd.DataFrame({"value": values, "target": labels.astype("float64")})
        stats = frame.groupby("value", dropna=False)["target"].agg(["sum", "count"])
        means = (stats["sum"] + self.smoothing * prior) / (stats["count"] + self.smoothing)
        means.name = "encoding"
        return means

    @staticmethod
    def _category_values(values: pd.Series) -> pd.Series:
        """Convert any categorical dtype to strings with an explicit missing level."""
        return values.astype("object").fillna("__MISSING__").astype("str")

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
