"""Explanation (Phase 10): SHAP additivity, reason codes and model agreement."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import logit
from src.cv.dataset import resolve_feature_columns
from src.explain.agreement import (
    load_importance,
    pairwise_agreement,
    run_agreement,
    spearman_rank,
    top_k_jaccard,
)
from src.explain.shap_analysis import (
    base_value,
    numeric_frame,
    reason_codes,
    run_explanation,
    sample_positions,
    select_applicants,
)
from src.models.lgbm import LGBMModel
from src.models.xgb import XGBModel

CATEGORICAL = ["CAT"]


def _toy(n_rows: int = 300, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "NUM": rng.normal(size=n_rows),
            "NUM_MISSING": rng.normal(size=n_rows),
            "CAT": pd.Series(rng.choice(["a", "b", None], n_rows), dtype="category"),
        }
    )
    features.loc[::9, "NUM_MISSING"] = np.nan
    signal = features["CAT"].astype("object").map({"a": -1.0, "b": 0.0}).fillna(1.0)
    target = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-signal.to_numpy()))).astype("int8")
    return features, target


@pytest.mark.parametrize(
    ("factory", "params"),
    [
        (LGBMModel, {"n_estimators": 20}),
        (XGBModel, {"n_estimators": 20}),
    ],
)
def test_shap_contributions_are_additive_with_the_model_log_odds(factory, params):
    features, target = _toy()
    model = factory(params, seed=42).fit(features, target, categorical_features=CATEGORICAL)

    contributions = model.shap_contributions(features)

    assert list(contributions.columns) == list(features.columns)
    assert len(contributions) == len(features)
    residuals = logit(model.predict_proba(features)) - contributions.sum(axis=1).to_numpy()
    assert residuals.std() < 1e-4
    assert residuals.max() - residuals.min() < 1e-4


def test_base_value_recovers_the_expected_log_odds():
    features, target = _toy()
    model = LGBMModel({"n_estimators": 20}, seed=42).fit(
        features, target, categorical_features=CATEGORICAL
    )
    outcome = model.predict_proba(features)
    contributions = model.shap_contributions(features)

    expected = base_value(outcome, contributions)

    assert isinstance(expected, float)
    assert np.isfinite(expected)


def test_base_value_rejects_non_additive_contributions():
    contributions = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, -0.1]})
    with pytest.raises(ValueError, match="not additive"):
        base_value(np.array([0.5, 0.5]), contributions)


def test_reason_codes_rank_the_risk_increasing_features_first():
    contributions = pd.Series({"a": 0.4, "b": -0.9, "c": 0.7, "d": 0.01})
    values = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    codes = reason_codes(contributions, values, k=2)

    assert [code["feature"] for code in codes] == ["c", "a"]
    assert [code["rank"] for code in codes] == [1, 2]
    assert all(code["direction"] == "increases risk" for code in codes)
    assert "moved the predicted default log-odds" in codes[0]["reason"]


def test_reason_codes_still_return_k_rows_with_few_positive_values():
    contributions = pd.Series({"a": 0.2, "b": -0.9, "c": -0.4, "d": -0.05})
    values = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    codes = reason_codes(contributions, values, k=3)

    assert len(codes) == 3
    assert codes[0]["feature"] == "a"
    assert codes[1]["feature"] == "b"  # largest magnitude among the negatives
    assert codes[1]["direction"] == "decreases risk"


def test_select_applicants_picks_the_extremes_and_checks_their_side():
    probabilities = np.array([0.1, 0.85, 0.4])
    identifiers = np.array([11, 12, 13])

    chosen = select_applicants(probabilities, identifiers, threshold=0.5)

    assert chosen["rejected"]["row"] == 1
    assert chosen["approved"]["row"] == 0
    assert chosen["rejected"]["identifier"] == 12


def test_select_applicants_refuses_a_sample_without_a_rejection():
    with pytest.raises(ValueError, match="No rejected applicant"):
        select_applicants(np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3]), threshold=0.5)


def test_numeric_frame_encodes_categories_as_codes():
    features = pd.DataFrame(
        {"c": pd.Series(["b", "a", None], dtype="category"), "n": [1.0, 2.0, 3.0]}
    )

    encoded = numeric_frame(features, ["c"])

    assert encoded["c"].dtype == "float64"
    assert encoded["c"].iloc[2] == -1.0  # the missing level keeps the code -1
    assert encoded["c"].iloc[0] != encoded["c"].iloc[1]


def test_sample_positions_are_deterministic_and_sorted():
    first = sample_positions(1_000, 100, seed=7)
    second = sample_positions(1_000, 100, seed=7)

    assert np.array_equal(first, second)
    assert first.size == 100
    assert np.all(np.diff(first) > 0)
    assert sample_positions(10, 50, seed=7).size == 10


def test_top_k_jaccard_matches_a_hand_computed_overlap():
    left = pd.Series([5.0, 4.0, 3.0], index=["a", "b", "c"])
    right = pd.Series([9.0, 4.0, 1.0], index=["b", "a", "d"])

    assert top_k_jaccard(left, right, 2) == pytest.approx(1.0)
    assert top_k_jaccard(left, right, 3) == pytest.approx(2 / 4)


def test_spearman_rank_is_one_for_identical_and_minus_one_for_reversed():
    base = pd.Series([4.0, 3.0, 2.0, 1.0], index=list("abcd"))
    reversed_values = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    assert spearman_rank(base, base) == pytest.approx(1.0)
    assert spearman_rank(base, reversed_values) == pytest.approx(-1.0)


def test_pairwise_agreement_returns_one_row_per_pair():
    importances = {
        "a": pd.Series([3.0, 2.0, 1.0], index=list("xyz")),
        "b": pd.Series([3.0, 1.0, 2.0], index=list("xyz")),
        "c": pd.Series([1.0, 2.0, 3.0], index=list("xyz")),
    }

    table = pairwise_agreement(importances, top_k=[2, 3])

    assert len(table) == 3
    assert {"model_a", "model_b", "spearman_rank", "jaccard_top2", "jaccard_top3"} <= set(
        table.columns
    )
    assert table["spearman_rank"].between(-1.0, 1.0).all()


def test_load_importance_sorts_and_rejects_duplicates(tmp_path):
    path = tmp_path / "importance.csv"
    pd.DataFrame({"feature": ["a", "b"], "importance_mean": [1.0, 5.0]}).to_csv(path, index=False)

    series = load_importance(path, name="x")
    assert list(series.index) == ["b", "a"]
    assert series.name == "x"

    duplicated = tmp_path / "dup.csv"
    pd.DataFrame({"feature": ["a", "a"], "importance_mean": [1.0, 2.0]}).to_csv(
        duplicated, index=False
    )
    with pytest.raises(ValueError, match="duplicated"):
        load_importance(duplicated)


def _explain_config(dataset, tmp_path: Path) -> dict:
    config = deepcopy(dataset.config)
    config["seeds"] = {"split": 42, "model": [42]}
    config["comparison"] = {
        "models": {"lightgbm": {"name": "lightgbm", "params": {"n_estimators": 12}}},
        "best_params": str(tmp_path / "E4_best_params.json"),
    }
    (tmp_path / "E4_best_params.json").write_text(
        json.dumps({"models": {"lightgbm": {"final_params": {"n_estimators": 12}}}}),
        encoding="utf-8",
    )
    config["explain"] = {
        "model": "lightgbm",
        "seed": 42,
        "sample": 20,
        "top_k": 3,
        "reason_codes": 3,
        "threshold": 0.5,
        "importance_table": "reports/tables/E7_shap_importance.csv",
        "reason_table": "reports/tables/E7_reason_codes.csv",
        "summary_figure": "reports/figures/E7_shap_summary.png",
        "dependence_figure": "reports/figures/E7_shap_dependence.png",
        "waterfall_dir": "reports/figures",
        "record": "reports/runs/{exp}/E7_shap.json",
    }
    config["agreement"] = {
        "importance": {
            "lightgbm": "reports/tables/lightgbm_importance.csv",
            "xgboost": "reports/tables/xgboost_importance.csv",
        },
        "top_k": [2, 3],
        "table": "reports/tables/E7_feature_agreement.csv",
        "record": "reports/runs/{exp}/E7_agreement.json",
    }
    config["outputs"] = {"exp": "E7_test"}
    return config


def test_run_explanation_writes_global_local_and_reason_artifacts(synthetic_dataset, tmp_path):
    config = _explain_config(synthetic_dataset, tmp_path)

    paths = run_explanation(config)

    importance = pd.read_csv(paths["importance"])
    selected = resolve_feature_columns(config, synthetic_dataset.contract)
    assert set(importance["feature"]) == set(selected)
    assert importance["rank"].is_monotonic_increasing
    assert importance["share"].sum() == pytest.approx(1.0)

    codes = pd.read_csv(paths["reason_table"])
    assert len(codes) == 3
    assert codes["rank"].tolist() == [1, 2, 3]
    assert {"feature", "value", "shap", "direction", "reason", "applicant_id"} <= set(codes.columns)

    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    assert record["model"] == "lightgbm"
    assert record["n_sample"] == 15  # the synthetic hold-out has 15 rows
    assert set(record["applicants"]) == {"approved", "rejected"}
    assert record["applicants"]["rejected"]["y_prob"] >= record["applicants"]["approved"]["y_prob"]
    for key in ("summary", "dependence", "waterfall_approved", "waterfall_rejected"):
        assert Path(record["artifacts"][key]).is_file()

    with pytest.raises(FileExistsError, match="already exist"):
        run_explanation(config)


def test_run_agreement_builds_the_pairwise_table(synthetic_dataset, tmp_path):
    config = _explain_config(synthetic_dataset, tmp_path)
    root = Path(config["_meta"]["project_root"])
    for name, values in {
        "lightgbm": [4.0, 3.0, 2.0, 1.0],
        "xgboost": [4.0, 2.0, 3.0, 1.0],
    }.items():
        path = root / config["agreement"]["importance"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": ["a", "b", "c", "d"], "importance_mean": values}).to_csv(
            path, index=False
        )

    paths = run_agreement(config)

    table = pd.read_csv(paths["table"])
    assert len(table) == 1
    assert table.loc[0, "model_a"] == "lightgbm"
    # lightgbm top-2 is {a, b}, xgboost top-2 is {a, c}: one shared of three.
    assert table.loc[0, "jaccard_top2"] == pytest.approx(1.0 / 3.0)
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    assert record["models"] == ["lightgbm", "xgboost"]
