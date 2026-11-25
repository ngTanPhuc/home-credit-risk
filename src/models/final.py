"""Full-development fit shared by the final hold-out, the explanation and the submission.

E1-E5 always fit inside one training fold. Once the protocol reaches the final
stage there is no further model selection, so one fit on *all* development rows
is the right estimator. This module keeps that fit in one place so the hold-out
script, the SHAP script and the submission cannot drift apart: all of them go
through the E3 imbalance strategy object and the model factory, and none ever
sees a hold-out label.

:class:`FinalModel` is the deployable artifact: the base models, the ensemble
head (logistic stack or fitted average) and the frozen calibrator, with one
``predict_proba``. It is what ``src/evaluation/submit.py`` scores ``test.parquet``
with and what ``--save-model`` persists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.calibration import Calibrator, build_calibrator
from src.imbalance import build_strategy, spec_from_config
from src.models import build_model
from src.models.base import BaseModel
from src.models.ensemble import (
    OOFMatrix,
    cross_fitted_average,
    cross_fitted_stacking,
    fit_average_weights,
)

SUPPORTED_ARMS = ("stacking", "weighted_average")


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


@dataclass
class FinalModel:
    """The frozen predictor: base models plus one ensemble head and one calibrator.

    ``predict_proba`` is the only entry point a deployment needs. The base models
    are fitted on all development rows; ``meta`` (logistic stack) and ``weights``
    (log-loss-fitted average) are fitted on *development OOF* predictions, and the
    calibrator on the arm's cross-fitted development OOF vector, never on rows the
    models were trained on.
    """

    arm: str
    models: dict[str, BaseModel]
    predictor_columns: list[str] = field(default_factory=list)
    meta: LogisticRegression | None = None
    weights: dict[str, float] | None = None
    calibrator: Calibrator | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.arm in SUPPORTED_ARMS:
            pass
        elif self.arm not in self.models:
            raise ValueError(
                f"arm {self.arm!r} is neither a fitted model {list(self.models)} "
                f"nor one of {list(SUPPORTED_ARMS)}"
            )
        if self.arm == "stacking" and self.meta is None:
            raise ValueError("stacking needs a fitted meta-learner")
        if self.arm == "weighted_average" and not self.weights:
            raise ValueError("weighted_average needs fitted weights")

    def base_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return each base model's probability for ``features``, columns in fit order."""
        missing = [name for name in self.predictor_columns if name not in features.columns]
        if missing:
            raise KeyError(
                f"features are missing {len(missing)} predictor columns, e.g. {missing[:5]}"
            )
        frame = features.loc[:, self.predictor_columns]
        return pd.DataFrame(
            {name: model.predict_proba(frame) for name, model in self.models.items()},
            index=features.index,
        )

    def arm_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """Return the arm's probability before calibration."""
        base = self.base_probabilities(features)
        if self.arm == "stacking":
            assert self.meta is not None  # checked in __post_init__
            order = list(getattr(self.meta, "feature_names_in_", base.columns))
            return np.asarray(self.meta.predict_proba(base.loc[:, order])[:, 1], dtype="float64")
        if self.arm == "weighted_average":
            assert self.weights is not None
            weights = np.array(
                [float(self.weights[str(name)]) for name in base.columns], dtype="float64"
            )
            return base.to_numpy(dtype="float64") @ weights
        return base[self.arm].to_numpy(dtype="float64")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the calibrated positive-class probability."""
        raw = self.arm_probabilities(features)
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.predict(raw), dtype="float64")

    def save(self, path: str | Path) -> Path:
        """Persist the fitted bundle with joblib (gitignored by ``*.joblib``)."""
        import joblib

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> FinalModel:
        """Load a bundle written by :meth:`save` and check its type."""
        import joblib

        bundle = joblib.load(Path(path))
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a FinalModel, got {type(bundle)!r}")
        return bundle


def fit_final_model(
    config: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    oof: OOFMatrix,
    arm: str,
    calibrator: str = "none",
    predictor_columns: Sequence[str],
    categorical_features: Sequence[str] | None = None,
    seed: int = 42,
    weight_step: float = 0.05,
) -> FinalModel:
    """Fit the deployable frozen predictor.

    Parameters
    ----------
    config, definitions:
        Resolved config and the E4 ``final_params`` model blocks, as
        ``src.evaluation.model_comparison.model_definitions`` returns them.
    features, target:
        Development rows only.
    oof:
        Development OOF predictions of the same models, used for the meta-learner,
        the blend weights and the calibrator. Never touched by ``fit``.
    arm:
        ``stacking``, ``weighted_average`` or one of the base model names.
    calibrator:
        ``none``, ``platt`` or ``isotonic``; fitted on the arm's cross-fitted
        development OOF vector.
    predictor_columns:
        Contract predictor order used at prediction time.
    seed:
        One seed drives every model, the meta-learner and the calibration.
    """
    labels = np.asarray(target).reshape(-1).astype("int8")
    if labels.shape[0] != len(features):
        raise ValueError("target length does not match development features")
    names = list(definitions)
    if arm not in names and arm not in SUPPORTED_ARMS:
        raise ValueError(f"arm {arm!r} is not one of {[*names, *SUPPORTED_ARMS]}")
    for name in names:
        if name not in oof.probabilities.columns:
            raise ValueError(f"OOF predictions for {name!r} are required for the ensemble head")

    models = {
        str(name): fit_development_model(
            config,
            block,
            features,
            labels,
            categorical_features=categorical_features,
            seed=seed,
        )
        for name, block in definitions.items()
    }

    meta: LogisticRegression | None = None
    weights: dict[str, float] | None = None
    if arm == "stacking":
        meta = LogisticRegression(max_iter=1000, random_state=int(seed))
        meta.fit(oof.probabilities, oof.meta[oof.target_column].to_numpy())
        development_probabilities = cross_fitted_stacking(oof, seed=int(seed)).y_prob
    elif arm == "weighted_average":
        weights = fit_average_weights(
            oof.meta[oof.target_column].to_numpy(), oof.probabilities, step=float(weight_step)
        )
        development_probabilities = cross_fitted_average(oof, step=float(weight_step)).y_prob
    else:
        development_probabilities = oof.probabilities[str(arm)].to_numpy(dtype="float64")

    fitted_calibrator = build_calibrator(calibrator).fit(
        oof.meta[oof.target_column].to_numpy(), development_probabilities
    )

    return FinalModel(
        arm=str(arm),
        models=models,
        predictor_columns=[str(name) for name in predictor_columns],
        meta=meta,
        weights=weights,
        calibrator=fitted_calibrator,
        seed=int(seed),
    )
