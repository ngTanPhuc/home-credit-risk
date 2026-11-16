"""Shared fixtures: a tiny synthetic dataset that mirrors the real contract.

The real artifacts under ``data/processed/datasets`` are large and gitignored, so
tests build a small dataset with the same layout: train/val/test Parquet,
``feature_contract.json`` and split metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
FOLD_COL = "fold"
PREDICTORS = ["AMT_CREDIT", "DAYS_BIRTH", "NAME_CONTRACT_TYPE", "BUREAU_N_LOANS"]
CATEGORICAL = ["NAME_CONTRACT_TYPE"]
N_TRAIN, N_VAL, N_TEST, N_FOLDS = 60, 15, 10, 3


@dataclass
class SyntheticDataset:
    root: Path
    config: dict
    contract: dict

    @property
    def dataset_dir(self) -> Path:
        return self.root / "datasets"


def _features(rng: np.random.Generator, n: int) -> dict:
    return {
        "AMT_CREDIT": rng.normal(500_000, 50_000, n),
        "DAYS_BIRTH": rng.integers(-20_000, -7_000, n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "BUREAU_N_LOANS": rng.integers(0, 5, n).astype("float64"),
    }


def _write_split(dataset_dir: Path, split: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(dataset_dir / f"{split}.parquet", index=False)


def _build_dataset(root: Path) -> SyntheticDataset:
    rng = np.random.default_rng(7)
    dataset_dir = root / "datasets"
    dataset_dir.mkdir(parents=True)
    split_dir = root / "splits"
    split_dir.mkdir()

    train_ids = np.arange(100_000, 100_000 + N_TRAIN, dtype="int32")
    val_ids = np.arange(200_000, 200_000 + N_VAL, dtype="int32")
    test_ids = np.arange(300_000, 300_000 + N_TEST, dtype="int32")

    train = pd.DataFrame({ID_COL: train_ids})
    train[TARGET_COL] = rng.integers(0, 2, N_TRAIN).astype("int8")
    train[FOLD_COL] = np.arange(N_TRAIN) % N_FOLDS
    for name, values in _features(rng, N_TRAIN).items():
        train[name] = values

    val = pd.DataFrame({ID_COL: val_ids})
    val[TARGET_COL] = rng.integers(0, 2, N_VAL).astype("int8")
    for name, values in _features(rng, N_VAL).items():
        val[name] = values

    test = pd.DataFrame({ID_COL: test_ids})
    for name, values in _features(rng, N_TEST).items():
        test[name] = values

    _write_split(dataset_dir, "train", train[[ID_COL, TARGET_COL, FOLD_COL, *PREDICTORS]])
    _write_split(dataset_dir, "val", val[[ID_COL, TARGET_COL, *PREDICTORS]])
    _write_split(dataset_dir, "test", test[[ID_COL, *PREDICTORS]])

    contract = {
        "version": "test_v1",
        "identifier_column": ID_COL,
        "target_column": TARGET_COL,
        "fold_column": FOLD_COL,
        "predictor_columns": PREDICTORS,
        "categorical_columns": CATEGORICAL,
        "numeric_columns": [name for name in PREDICTORS if name not in CATEGORICAL],
        "excluded_from_X": [ID_COL, TARGET_COL, FOLD_COL],
        "n_predictors": len(PREDICTORS),
        "dataset_columns": {
            "train": [ID_COL, TARGET_COL, FOLD_COL, *PREDICTORS],
            "val": [ID_COL, TARGET_COL, *PREDICTORS],
            "test": [ID_COL, *PREDICTORS],
        },
    }
    contract_path = dataset_dir / "feature_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    metadata = {
        "source_sha256": "0" * 64,
        "holdout_ratio": 0.2,
        "holdout_seed": 2026,
        "cv_seed": 42,
        "n_folds": N_FOLDS,
        "n_full": N_TRAIN + N_VAL,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
    }
    (split_dir / "train_val_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    config = {
        "_meta": {"project_root": str(root), "config_path": str(root / "config.yaml")},
        "dataset": {"dir": "datasets", "contract": "datasets/feature_contract.json"},
        "split": {"n_folds": N_FOLDS, "metadata": "splits/train_val_metadata.json"},
        "feature_groups": {
            "application_original": ["AMT_CREDIT", "DAYS_BIRTH", "NAME_CONTRACT_TYPE"],
            "bureau": ["BUREAU_N_LOANS"],
        },
        "feature_sets": {
            "main": ["application_original"],
            "plus_bureau": ["application_original", "bureau"],
        },
        "features": {"use_set": "main", "categorical_features": "auto"},
        "metrics": {"ece_bins": 5, "thresholds": [0.2, 0.5]},
    }
    return SyntheticDataset(root=root, config=config, contract=contract)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> SyntheticDataset:
    """Synthetic dataset plus a config pointing at it."""
    return _build_dataset(tmp_path)
