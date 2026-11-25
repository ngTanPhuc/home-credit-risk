"""Per-fold imbalance handling (E3).

Every strategy here is a fit/resample object built fresh inside one fold. It
receives only that fold's training rows and returns the rows the model may
train on, so validation rows can never be resampled or oversampled into the
training set.

Variants
--------
``none``
    No change.
``class_weight``
    Per-row weights equivalent to ``scale_pos_weight = n_negative / n_positive``.
    Expressed as sample weights so the same object works for every model in
    Phase 6.
``undersample``
    Random undersampling of the majority class (:class:`RandomUnderSampler`).
``smotenc``
    SMOTENC on the contract's categorical columns. SMOTENC rejects NaN, so the
    training fold is filled first with its own median (numeric) and mode
    (categorical); validation keeps the original missing values.

All strategies keep the row order-insensitive contract: features, target and
an optional sample-weight vector of the same length.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

LOGGER = logging.getLogger(__name__)


class ImbalanceStrategy(ABC):
    """Fit/resample object applied to one training fold."""

    name: str = "base"

    @abstractmethod
    def fit_resample(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        """Return ``(features, target, sample_weight)`` for this fold's training rows.

        ``sample_weight`` is ``None`` when the variant does not weight rows.
        """


def _already_meets_ratio(labels: np.ndarray, sampling_strategy: float) -> bool:
    """True when minority/majority already reaches the sampler's target ratio.

    imblearn raises instead of no-oping in that case, which would break small
    folds and smoke runs where the classes happen to be nearly balanced.
    """
    counts = np.bincount(labels, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("both classes are required in the training fold")
    minority, majority = sorted(int(count) for count in counts)
    # Mirrors imblearn's own guard: it truncates the target count and raises
    # instead of no-oping when the ratio is already met.
    return int(majority * sampling_strategy - minority) <= 0


class NoImbalance(ImbalanceStrategy):
    """Reference variant: train on the fold-training rows unchanged."""

    name = "none"

    def fit_resample(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        return features, np.asarray(target).reshape(-1).astype("int8"), None


class ClassWeight(ImbalanceStrategy):
    """Weight the positive class like ``scale_pos_weight``, without resampling."""

    name = "class_weight"

    def fit_resample(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        labels = np.asarray(target).reshape(-1).astype("int8")
        n_positive = int((labels == 1).sum())
        n_negative = int(len(labels) - n_positive)
        if n_positive == 0 or n_negative == 0:
            raise ValueError("class_weight needs both classes in the training fold")
        weights = np.where(labels == 1, n_negative / n_positive, 1.0)
        return features, labels, weights.astype("float64")


class RandomUndersample(ImbalanceStrategy):
    """Drop majority-class rows until ``sampling_strategy`` is met."""

    name = "undersample"

    def __init__(self, *, seed: int = 42, sampling_strategy: float = 0.5) -> None:
        self.seed = int(seed)
        self.sampling_strategy = float(sampling_strategy)

    def fit_resample(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        labels = np.asarray(target).reshape(-1).astype("int8")
        if len(np.unique(labels)) < 2:
            raise ValueError("undersample needs both classes in the training fold")
        if _already_meets_ratio(labels, self.sampling_strategy):
            LOGGER.debug("undersample: ratio already met, rows unchanged")
            return features, labels, None

        sampler = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.seed,
        )
        resampled, resampled_target = sampler.fit_resample(features, labels)
        return (
            resampled.reset_index(drop=True),
            np.asarray(resampled_target).reshape(-1).astype("int8"),
            None,
        )


class SmoteNC(ImbalanceStrategy):
    """SMOTENC oversampling on the categorical columns of the contract."""

    name = "smotenc"

    def __init__(
        self,
        *,
        seed: int = 42,
        sampling_strategy: float = 0.5,
        n_neighbors: int = 5,
    ) -> None:
        self.seed = int(seed)
        self.sampling_strategy = float(sampling_strategy)
        self.n_neighbors = int(n_neighbors)

    def fit_resample(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        *,
        categorical_features: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        labels = np.asarray(target).reshape(-1).astype("int8")
        n_positive = int((labels == 1).sum())
        if n_positive == 0 or n_positive == len(labels):
            raise ValueError("smotenc needs both classes in the training fold")
        if n_positive <= self.n_neighbors:
            raise ValueError(
                f"smotenc needs more than n_neighbors={self.n_neighbors} minority rows, "
                f"got {n_positive}"
            )

        requested = [str(column) for column in (categorical_features or [])]
        missing = [column for column in requested if column not in features.columns]
        if missing:
            raise KeyError(f"Categorical columns not found in features: {missing}")

        frame = _fill_missing(features, requested)
        vocabularies = {column: pd.Index(frame[column].unique()) for column in requested}
        if _already_meets_ratio(labels, self.sampling_strategy):
            LOGGER.debug("smotenc: ratio already met, rows unchanged")
            for column in requested:
                frame[column] = pd.Categorical(frame[column], categories=vocabularies[column])
            return frame, labels, None

        for column in requested:
            frame[column] = frame[column].astype("object")

        sampler = SMOTENC(
            categorical_features=requested,
            sampling_strategy=self.sampling_strategy,
            random_state=self.seed,
            k_neighbors=self.n_neighbors,
        )
        resampled, resampled_target = sampler.fit_resample(frame, labels)

        resampled = pd.DataFrame(resampled, columns=frame.columns)
        for column in requested:
            resampled[column] = pd.Categorical(
                resampled[column].astype("object"), categories=vocabularies[column]
            )
        LOGGER.debug(
            "smotenc: %d -> %d training rows (categorical=%d)",
            len(features),
            len(resampled),
            len(requested),
        )
        return (
            resampled.reset_index(drop=True),
            np.asarray(resampled_target).reshape(-1).astype("int8"),
            None,
        )


def _fill_missing(features: pd.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    """Fill NaN with the training fold's median (numeric) or mode (categorical).

    SMOTENC rejects missing values, and the frozen dataset keeps structural
    missingness by design, so this fill is fitted on the training fold only.
    """
    filled = features.copy()
    categorical_names = set(categorical)
    for column in filled.columns:
        values = filled[column]
        if not values.isna().any():
            continue
        if column in categorical_names or not pd.api.types.is_numeric_dtype(values):
            modes = values.mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else ""
        else:
            fill_value = values.median()
            if pd.isna(fill_value):
                fill_value = 0.0
        filled[column] = values.fillna(fill_value)
    return filled


_STRATEGIES: dict[str, type[ImbalanceStrategy]] = {
    "none": NoImbalance,
    "class_weight": ClassWeight,
    "undersample": RandomUndersample,
    "smotenc": SmoteNC,
}

_OPTION_KEYS = ("sampling_strategy", "n_neighbors")

# Options each variant actually accepts; a shared config block may carry all of them.
_ACCEPTED_OPTIONS: dict[str, tuple[str, ...]] = {
    "undersample": ("sampling_strategy",),
    "smotenc": ("sampling_strategy", "n_neighbors"),
}


def build_strategy(specification: Any = "none", *, seed: int = 42) -> ImbalanceStrategy:
    """Build one strategy from a name or a mapping with ``name`` and options."""
    if specification is None:
        specification = "none"

    if isinstance(specification, str):
        name, options = specification, {}
    elif isinstance(specification, Mapping):
        options = dict(specification)
        name = str(options.pop("name", options.pop("strategy", "none")))
    else:
        raise TypeError(f"imbalance must be a string or mapping, got {type(specification)!r}")

    name = name.strip().lower()
    try:
        strategy_class = _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown imbalance strategy {name!r}; known: {sorted(_STRATEGIES)}"
        ) from None

    unknown = sorted(set(options) - set(_OPTION_KEYS))
    if unknown:
        raise ValueError(f"Unsupported imbalance options for {name!r}: {unknown}")

    if name in {"none", "class_weight"}:
        return strategy_class()
    accepted = _ACCEPTED_OPTIONS.get(name, ())
    return strategy_class(seed=seed, **{key: options[key] for key in accepted if key in options})


def spec_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the per-run strategy spec from an experiment config.

    The ``imbalance`` block also holds the E3 variant list and table path, which are
    driver settings and are deliberately ignored here.
    """
    block = config.get("imbalance") or {}
    if not isinstance(block, Mapping):
        raise TypeError("imbalance must be a mapping")
    spec: dict[str, Any] = {"name": str(block.get("strategy", "none"))}
    spec.update({key: block[key] for key in _OPTION_KEYS if key in block})
    return spec
