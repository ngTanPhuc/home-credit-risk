"""Dataset and feature-contract loading for modelling.

The input is the frozen dataset written by
``notebooks/04_dataset_assembly_validation.ipynb``:

* ``train.parquet`` — ``SK_ID_CURR``, ``TARGET``, ``fold``, predictors (development rows)
* ``val.parquet`` — ``SK_ID_CURR``, ``TARGET``, predictors (final hold-out)
* ``test.parquet`` — ``SK_ID_CURR``, predictors (Kaggle test)

``feature_contract.json`` is the schema source of truth: predictor order,
categorical columns, per-split column lists and excluded keys. Feature groups
are declared in ``configs/features.yaml`` and must stay inside that contract.

Hold-out labels are locked: :func:`load_split_frame` refuses ``split="val"``
unless the caller passes ``allow_holdout=True``, which only
``src/evaluation/holdout.py`` may do.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.config import get_nested, resolve_path

PathLike = str | os.PathLike[str]

# Files in data/processed/datasets/ that are part of the frozen dataset
SPLIT_FILES: dict[str, str] = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}

# The keys that must be present in the feature_contract.json file
_CONTRACT_KEYS = (
    "version",
    "identifier_column",
    "target_column",
    "fold_column",
    "predictor_columns",
    "categorical_columns",
    "numeric_columns",
    "excluded_from_X",
    "dataset_columns",
)


def load_contract(path: PathLike) -> dict[str, Any]:
    """Load and validate ``feature_contract.json``.

    Parameters
    ----------
    path:
        Contract file produced by the dataset notebook.

    Returns
    -------
    dict
        Parsed contract with predictor order and categorical columns.

    Raises
    ------
    FileNotFoundError
        If the contract is missing (the dataset is not built on this machine).
    ValueError
        If required fields are absent or internally inconsistent.
    """
    import json

    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Feature contract not found: {contract_path}. "
            "Build the dataset with notebooks/01..04 or copy data/processed/datasets/."
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    missing = [key for key in _CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(f"Feature contract is missing fields {missing}: {contract_path}")

    predictors = list(contract["predictor_columns"])
    if len(predictors) != len(set(predictors)):
        raise ValueError("Feature contract lists duplicate predictor columns")
    if int(contract["n_predictors"]) != len(predictors):
        raise ValueError(
            f"Contract declares n_predictors={contract['n_predictors']} but lists {len(predictors)}"
        )

    categorical = set(contract["categorical_columns"])
    numeric = set(contract["numeric_columns"])
    if categorical & numeric:
        raise ValueError("Columns listed as both categorical and numeric")
    if categorical | numeric != set(predictors):
        raise ValueError("Categorical + numeric columns do not match the predictor list")

    for split, columns in contract["dataset_columns"].items():
        if split not in SPLIT_FILES:
            raise ValueError(f"Contract describes an unknown split: {split}")
        if not set(predictors) <= set(columns):
            raise ValueError(f"Contract split {split!r} is missing predictor columns")

    return contract


def dataset_path(config: Mapping[str, Any], split: str) -> Path:
    """Return the Parquet path for ``split`` resolved against the repo root."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    return resolve_path(config, "dataset.dir") / SPLIT_FILES[split]


# Most important function
def load_split_frame(
    config: Mapping[str, Any],
    split: str,
    contract: Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
    allow_holdout: bool = False,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Load one dataset split and validate it against the contract.

    Parameters
    ----------
    config:
        Resolved config; ``dataset.dir`` points at the dataset directory.
    split:
        ``train``, ``val`` or ``test``.
    contract:
        Output of :func:`load_contract`.
    columns:
        Optional subset to read. Key columns are always kept.
    allow_holdout:
        Required to load ``val``. Only the locked hold-out script sets it.
    n_folds:
        Expected fold count when loading ``train``.

    Returns
    -------
    pandas.DataFrame
        Frame with contract column order.

    Raises
    ------
    PermissionError
        If ``split="val"`` and ``allow_holdout`` is false.
    ValueError
        If the file does not match the contract.
    """

    # Holdout lock, prevent accidental leakage of val labels into train or test
    if split == "val" and not allow_holdout:
        raise PermissionError(
            "val.parquet carries hold-out labels; only src/evaluation/holdout.py "
            "may load it (allow_holdout=True)."
        )

    path = dataset_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}. Build it with notebooks/01..04.")

    expected = list(contract["dataset_columns"][split])
    if columns is None:
        selected = expected
    else:
        unknown = [name for name in columns if name not in expected]
        if unknown:
            raise KeyError(f"Columns not in the {split} contract: {unknown}")
        wanted = set(columns)
        selected = [name for name in expected if name in wanted]

    # Check the Parquet schema without reading data
    # It checks if the columns in the file exactly match the columns defined in the contract
    # If someone added a new column, deleted a column, or renamed one, it crashes
    file_columns = list(pq.read_schema(path).names)
    if file_columns != expected:
        if split == "test" and contract["target_column"] in file_columns:
            raise ValueError("test.parquet must not contain TARGET")
        extra = [name for name in file_columns if name not in expected]
        missing = [name for name in expected if name not in file_columns]
        raise ValueError(
            f"{split}.parquet schema does not match the contract: "
            f"extra={extra[:5]} missing={missing[:5]}"
        )

    frame = pd.read_parquet(path, columns=selected).loc[:, selected]
    _validate_frame(frame, split=split, contract=contract, n_folds=n_folds)  # validate the loaded frame against the contract
    return frame


def _validate_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    contract: Mapping[str, Any],
    n_folds: int | None,
) -> None:
    """Enforce key, label and fold rules for one split."""
    id_col = str(contract["identifier_column"])
    target_col = str(contract["target_column"])
    fold_col = str(contract["fold_column"])

    for key in (id_col, target_col, fold_col):
        if key in frame.columns and not pd.api.types.is_integer_dtype(frame[key]):
            raise ValueError(
                f"{split}.parquet column {key} must be integer, got {frame[key].dtype}"
            )
    if id_col not in frame.columns:
        raise ValueError(f"{split}.parquet is missing {id_col}")
    if frame[id_col].isna().any():
        raise ValueError(f"{split}.parquet has missing {id_col}")
    if not frame[id_col].is_unique:
        raise ValueError(f"{split}.parquet has duplicate {id_col}")

    if split == "test":
        if target_col in frame.columns:
            raise ValueError("test.parquet must not contain TARGET")
        return

    if target_col not in frame.columns:
        raise ValueError(f"{split}.parquet is missing {target_col}")
    if frame[target_col].isna().any():
        raise ValueError(f"{split}.parquet has missing {target_col}")
    if not set(frame[target_col].unique().tolist()) <= {0, 1}:
        raise ValueError(f"{split}.parquet {target_col} must be binary")

    if split == "train":
        if fold_col not in frame.columns:
            raise ValueError("train.parquet is missing the fold column")
        if frame[fold_col].isna().any():
            raise ValueError("train.parquet has missing fold values")
        folds = sorted(frame[fold_col].unique().tolist())
        if folds != list(range(len(folds))):
            raise ValueError(f"Fold ids must be 0..k-1, found {folds}")
        if n_folds is not None and len(folds) != int(n_folds):
            raise ValueError(f"Config expects {n_folds} folds but train.parquet has {len(folds)}")


def resolve_feature_columns(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Feature Selector - Resolve the configured feature set to contract-ordered column names.
    This uses the file ``configs/features.yaml`` to select groups of features, and the contract to ensure that the selected features are valid.

    Config contract
    ---------------
    ``feature_groups``
        Mapping of group name to its columns (see ``configs/features.yaml``).
    ``feature_sets``
        Mapping of set name to a list of group names.
    ``features.use_set`` or ``features.groups``
        Exactly one selection; ``features.exclude`` drops individual columns.

    Returns
    -------
    list[str]
        Predictor columns in contract order.

    Raises
    ------
    ValueError
        If no selection is configured, a group is unknown, or the result is empty.
    KeyError
        If a group or exclusion references columns outside the contract.
    """
    groups_cfg = get_nested(config, "feature_groups", {}) or {}
    sets_cfg = get_nested(config, "feature_sets", {}) or {}
    selection = get_nested(config, "features", {}) or {}
    if not isinstance(selection, Mapping):
        raise TypeError("features must be a mapping")

    use_set = selection.get("use_set")
    listed_groups = selection.get("groups")
    if use_set and listed_groups:
        raise ValueError("Set only one of features.use_set or features.groups")
    if use_set:
        if use_set not in sets_cfg:
            raise ValueError(f"Unknown feature set {use_set!r}; known sets: {sorted(sets_cfg)}")
        groups = list(sets_cfg[use_set])
    elif listed_groups:
        groups = list(listed_groups)
    else:
        raise ValueError("Config must set features.use_set or features.groups")

    unknown = [group for group in groups if group not in groups_cfg]
    if unknown:
        raise ValueError(f"Unknown feature group(s) {unknown}; known: {sorted(groups_cfg)}")

    chosen: set[str] = set()
    for group in groups:
        chosen.update(str(name) for name in groups_cfg[group])

    predictors = [str(name) for name in contract["predictor_columns"]]
    outside = sorted(chosen - set(predictors))
    if outside:
        raise ValueError(
            f"Feature groups reference {len(outside)} columns absent from the contract, "
            f"e.g. {outside[:5]}"
        )

    excluded = {str(name) for name in (selection.get("exclude") or [])}
    unknown_exclusions = sorted(excluded - set(predictors))
    if unknown_exclusions:
        raise KeyError(f"Excluded columns are not predictors: {unknown_exclusions}")

    columns = [name for name in predictors if name in chosen and name not in excluded]
    if not columns:
        raise ValueError("Feature selection resolved to zero columns")

    keys = {
        str(contract["identifier_column"]),
        str(contract["target_column"]),
        str(contract["fold_column"]),
    }
    if keys & set(columns):
        raise AssertionError("Key columns must never be predictors")
    return columns
