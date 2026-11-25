"""Split provenance guards: metadata, counts and the hold-out lock."""

from __future__ import annotations

import copy

import pytest

from src.cv.splits import (
    load_split_metadata,
    parquet_row_count,
    validate_split_metadata,
)


def test_metadata_loads(synthetic_dataset):
    metadata = load_split_metadata(synthetic_dataset.root / "splits/train_val_metadata.json")
    assert metadata["n_folds"] == 3
    assert metadata["holdout_seed"] == 2026
    assert metadata["n_train"] + metadata["n_val"] == metadata["n_full"]


def test_counts_match_dataset(synthetic_dataset):
    metadata = load_split_metadata(synthetic_dataset.root / "splits/train_val_metadata.json")
    n_train = parquet_row_count(synthetic_dataset.dataset_dir / "train.parquet")
    n_val = parquet_row_count(synthetic_dataset.dataset_dir / "val.parquet")
    validate_split_metadata(metadata, n_train=n_train, n_val=n_val, n_folds=3)


def test_wrong_row_count_is_rejected(synthetic_dataset):
    metadata = load_split_metadata(synthetic_dataset.root / "splits/train_val_metadata.json")
    with pytest.raises(ValueError, match="train.parquet has"):
        validate_split_metadata(metadata, n_train=1, n_val=15, n_folds=3)


def test_wrong_fold_count_is_rejected(synthetic_dataset):
    metadata = load_split_metadata(synthetic_dataset.root / "splits/train_val_metadata.json")
    with pytest.raises(ValueError, match="folds"):
        validate_split_metadata(metadata, n_train=60, n_val=15, n_folds=5)


def test_inconsistent_metadata_is_rejected(synthetic_dataset):
    metadata = copy.deepcopy(
        load_split_metadata(synthetic_dataset.root / "splits/train_val_metadata.json")
    )
    metadata["n_full"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_split_metadata(metadata, n_train=60, n_val=15, n_folds=3)


def test_missing_metadata_names_the_fix(synthetic_dataset):
    with pytest.raises(FileNotFoundError, match="notebooks/01_data_overview_eda.ipynb"):
        load_split_metadata(synthetic_dataset.root / "splits/missing.json")
