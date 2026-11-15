"""Split provenance for the frozen dataset.

The split is created once by ``notebooks/01_data_overview_eda.ipynb`` and stored
as ``data/splits/{train,val}.parquet`` plus ``train_val_metadata.json``. Fold
membership is copied into ``data/processed/datasets/train.parquet`` as the
``fold`` column, so modelling code never re-splits or reshuffles.

This module only reads and validates that metadata; it never creates a split.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PathLike = str | os.PathLike[str]

_REQUIRED_KEYS = (
    "source_sha256",
    "holdout_ratio",
    "holdout_seed",
    "cv_seed",
    "n_folds",
    "n_full",
    "n_train",
    "n_val",
)


def load_split_metadata(path: PathLike) -> dict[str, Any]:
    """Load ``train_val_metadata.json`` and require the protocol fields.

    Parameters
    ----------
    path:
        Metadata file written by the split notebook.

    Returns
    -------
    dict
        Parsed metadata, including seeds and row counts.

    Raises
    ------
    FileNotFoundError
        If the metadata file is missing.
    ValueError
        If a protocol field is absent.
    """
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Split metadata not found: {metadata_path}. "
            "Run notebooks/01_data_overview_eda.ipynb or restore data/splits/."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Split metadata is missing fields {missing}: {metadata_path}")
    return metadata


def parquet_row_count(path: PathLike) -> int:
    """Return a Parquet file's row count from its footer, without reading data."""
    return int(pq.ParquetFile(Path(path)).metadata.num_rows)


def validate_split_metadata(
    metadata: Mapping[str, Any],
    *,
    n_train: int,
    n_val: int,
    n_folds: int,
) -> None:
    """Check that the loaded dataset still matches the frozen split.

    Parameters
    ----------
    metadata:
        Output of :func:`load_split_metadata`.
    n_train:
        Development rows present in ``train.parquet``.
    n_val:
        Hold-out rows counted from the ``val.parquet`` footer (no labels read).
    n_folds:
        Fold count expected by the experiment config.

    Raises
    ------
    ValueError
        On any count or fold mismatch.
    """
    if int(metadata["n_folds"]) != int(n_folds):
        raise ValueError(
            f"Config expects {n_folds} folds but the split declares {metadata['n_folds']}"
        )
    if int(metadata["n_train"]) != int(n_train):
        raise ValueError(
            f"train.parquet has {n_train} rows but the split declares {metadata['n_train']}"
        )
    if int(metadata["n_val"]) != int(n_val):
        raise ValueError(
            f"Split declares {metadata['n_val']} hold-out rows, config expects {n_val}"
        )
    if int(metadata["n_train"]) + int(metadata["n_val"]) != int(metadata["n_full"]):
        raise ValueError("Split metadata row counts are inconsistent (train + val != full)")
