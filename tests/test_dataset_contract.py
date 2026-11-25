"""Contract, split-loading and feature-selection guards."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.cv.dataset import (
    load_contract,
    load_split_frame,
    resolve_feature_columns,
)
from tests.conftest import ID_COL, PREDICTORS, TARGET_COL


def _contract_path(dataset) -> str:
    return dataset.config["dataset"]["contract"]


def _rewrite_contract(dataset, contract: dict) -> None:
    (dataset.root / _contract_path(dataset)).write_text(json.dumps(contract), encoding="utf-8")


def test_contract_roundtrip(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    assert contract["predictor_columns"] == PREDICTORS
    assert contract["n_predictors"] == len(PREDICTORS)
    assert set(contract["categorical_columns"]) == {"NAME_CONTRACT_TYPE"}


def test_train_split_loads_in_contract_order(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    frame = load_split_frame(
        synthetic_dataset.config,
        "train",
        contract,
        n_folds=synthetic_dataset.config["split"]["n_folds"],
    )
    assert list(frame.columns) == contract["dataset_columns"]["train"]
    assert frame[ID_COL].is_unique
    assert sorted(frame["fold"].unique().tolist()) == [0, 1, 2]
    assert set(frame[TARGET_COL].unique().tolist()) <= {0, 1}


def test_holdout_requires_explicit_flag(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    with pytest.raises(PermissionError, match="hold-out labels"):
        load_split_frame(synthetic_dataset.config, "val", contract)


def test_test_split_has_no_target(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    frame = load_split_frame(synthetic_dataset.config, "test", contract)
    assert TARGET_COL not in frame.columns

    tampered = pd.read_parquet(synthetic_dataset.dataset_dir / "test.parquet")
    tampered[TARGET_COL] = 0
    tampered.to_parquet(synthetic_dataset.dataset_dir / "test.parquet", index=False)
    with pytest.raises(ValueError, match="must not contain TARGET"):
        load_split_frame(synthetic_dataset.config, "test", contract)


def test_contract_mismatch_is_rejected(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    broken = copy.deepcopy(contract)
    broken["dataset_columns"]["train"].remove("BUREAU_N_LOANS")
    with pytest.raises(KeyError, match="not in the train contract"):
        load_split_frame(
            synthetic_dataset.config,
            "train",
            broken,
            columns=[ID_COL, TARGET_COL, "fold", "BUREAU_N_LOANS"],
        )


def test_incomplete_contract_is_rejected(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    broken = copy.deepcopy(contract)
    broken["dataset_columns"]["train"].remove("BUREAU_N_LOANS")
    _rewrite_contract(synthetic_dataset, broken)
    with pytest.raises(ValueError, match="missing predictor columns"):
        load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))


def test_duplicate_predictors_are_rejected(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    broken = copy.deepcopy(contract)
    broken["predictor_columns"].append("AMT_CREDIT")
    _rewrite_contract(synthetic_dataset, broken)
    with pytest.raises(ValueError, match="duplicate predictor"):
        load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))


def test_missing_dataset_gives_actionable_error(synthetic_dataset):
    config = copy.deepcopy(synthetic_dataset.config)
    config["dataset"]["dir"] = "does-not-exist"
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    with pytest.raises(FileNotFoundError, match="Build it with notebooks"):
        load_split_frame(config, "train", contract)


def test_feature_sets_follow_contract_order(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    main = resolve_feature_columns(synthetic_dataset.config, contract)
    assert main == ["AMT_CREDIT", "DAYS_BIRTH", "NAME_CONTRACT_TYPE"]

    config = copy.deepcopy(synthetic_dataset.config)
    config["features"]["use_set"] = "plus_bureau"
    wider = resolve_feature_columns(config, contract)
    assert wider == PREDICTORS
    assert set(main) < set(wider)


def test_groups_selection_matches_use_set(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    config = copy.deepcopy(synthetic_dataset.config)
    config["features"] = {"groups": ["application_original", "bureau"]}
    assert resolve_feature_columns(config, contract) == PREDICTORS


def test_conflicting_selection_is_rejected(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    config = copy.deepcopy(synthetic_dataset.config)
    config["features"] = {"use_set": "main", "groups": ["bureau"]}
    with pytest.raises(ValueError, match="Set only one of"):
        resolve_feature_columns(config, contract)


def test_unknown_group_is_rejected(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    config = copy.deepcopy(synthetic_dataset.config)
    config["features"] = {"groups": ["time_windows"]}
    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_feature_columns(config, contract)


def test_exclusions_and_empty_selection(synthetic_dataset):
    contract = load_contract(synthetic_dataset.root / _contract_path(synthetic_dataset))
    config = copy.deepcopy(synthetic_dataset.config)
    config["features"] = {"use_set": "main", "exclude": ["DAYS_BIRTH"]}
    assert resolve_feature_columns(config, contract) == ["AMT_CREDIT", "NAME_CONTRACT_TYPE"]

    config["features"]["exclude"] = list(PREDICTORS)
    with pytest.raises(ValueError, match="zero columns"):
        resolve_feature_columns(config, contract)
