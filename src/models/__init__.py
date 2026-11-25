"""Model factory shared by the CV runner and the E4 tuner.

The model block of a YAML config (``name``, ``params``, ``runtime`` and any
model-specific keys) is the only input; callers never import wrappers directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.models.base import BaseModel

SUPPORTED_MODELS = ("lightgbm", "xgboost", "catboost")


def build_model(model_config: Mapping[str, Any], *, seed: int) -> BaseModel:
    """Build a fresh model from a resolved ``model`` config block."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model must be a mapping")
    name = str(model_config.get("name", "lightgbm")).strip().lower()
    params = dict(model_config.get("params") or {})
    runtime = dict(model_config.get("runtime") or {})

    if name == "lightgbm":
        from src.models.lgbm import LGBMModel

        return LGBMModel(
            params=params,
            seed=seed,
            n_jobs=int(runtime.get("n_jobs", -1)),
            deterministic=bool(runtime.get("deterministic", True)),
            force_col_wise=bool(runtime.get("force_col_wise", True)),
        )
    if name == "xgboost":
        from src.models.xgb import XGBModel

        return XGBModel(
            params=params,
            seed=seed,
            n_jobs=int(runtime.get("n_jobs", -1)),
            target_encoding=model_config.get("target_encoding"),
        )
    if name == "catboost":
        from src.models.catboost import CatBoostModel

        return CatBoostModel(
            params=params,
            seed=seed,
            thread_count=int(runtime.get("thread_count", -1)),
        )
    raise ValueError(f"Unknown model {name!r}; supported: {list(SUPPORTED_MODELS)}")
