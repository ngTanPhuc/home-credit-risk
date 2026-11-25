"""E7: global and local TreeSHAP explanation on hold-out *features* only.

The model is the frozen E4/E5 configuration refitted on all development rows
(``src.models.final``). It is then applied to a fixed-size sample of the
hold-out split, which is read through
:func:`src.cv.dataset.load_holdout_features`: identifiers and predictors, never
labels. No metric in this module uses the target.

Outputs
-------
``E7_shap_importance.csv``
    Mean absolute SHAP per feature, ranked, with each feature's share.
``E7_shap_summary.png``, ``E7_shap_dependence.png``
    Beeswarm summary and the dependence of the top features.
``E7_reason_codes.csv``
    Ranked insurance-style reason codes for the rejected applicant.
``E7_shap_waterfall_{approved,rejected}.png``
    Local explanation for one approved and one rejected applicant.
``E7_shap.json``
    Protocol record: model, seed, sample size, chosen applicants, top features.

Applicant selection is deterministic: the sampled applicant with the highest
predicted probability is the rejected case and the lowest is the approved case.
Both must fall on the expected side of the decision threshold, otherwise the run
stops instead of explaining the wrong story.

Usage
-----
``uv run python -m src.explain.shap_analysis --config configs/experiments/E7_explain.yaml``

``--sample N`` overrides the configured row count (10k by default), ``--model``
switches backend, and ``--force`` overwrites existing artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_nested, load_config, resolve_path, set_seeds
from src.cv.dataset import (
    load_contract,
    load_holdout_features,
    load_split_frame,
    resolve_feature_columns,
)
from src.evaluation.model_comparison import model_definitions, model_seeds
from src.models.final import fit_development_model
from src.utils.logging import setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _number(value: Any) -> Any:
    """Convert numpy scalars to JSON-friendly Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def sample_positions(n_rows: int, sample: int, *, seed: int) -> np.ndarray:
    """Return sorted, deterministic positions for the explanation sample."""
    if sample < 1:
        raise ValueError("--sample must be positive")
    if sample >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=int(sample), replace=False))


def numeric_frame(features: pd.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    """Return a numeric view where categories become integer codes (for plotting)."""
    frame = features.copy()
    for column in categorical:
        if column in frame.columns:
            frame[column] = pd.Categorical(frame[column].astype("object")).codes.astype("float64")
    return frame


def base_value(probabilities: np.ndarray, contributions: pd.DataFrame) -> float:
    """Recover the TreeSHAP expected value from the additive identity.

    Contributions sum with the expected value to the model's log-odds, so
    ``logit(p) - sum(contributions)`` is that expected value. It is constant
    across rows for a tree ensemble; the median is returned and the spread is
    checked to catch a mismatched contribution matrix.
    """
    from src.calibration import logit

    residuals = logit(probabilities) - contributions.sum(axis=1).to_numpy(dtype="float64")
    spread = float(residuals.max() - residuals.min())
    if spread > 1e-3:
        raise ValueError(
            f"SHAP contributions are not additive with the model output (spread {spread:.4g})"
        )
    return float(np.median(residuals))


def select_applicants(
    probabilities: np.ndarray,
    identifiers: np.ndarray,
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Pick the highest- and lowest-risk sampled applicants and check their side."""
    values = np.asarray(probabilities, dtype="float64").reshape(-1)
    if values.size < 2:
        raise ValueError("At least two sampled applicants are needed")
    rejected = int(np.argmax(values))
    approved = int(np.argmin(values))
    if values[rejected] < threshold:
        raise ValueError(
            f"No rejected applicant in the sample: max probability {values[rejected]:.4f} "
            f"is below the threshold {threshold}"
        )
    if values[approved] >= threshold:
        raise ValueError(
            f"No approved applicant in the sample: min probability {values[approved]:.4f} "
            f"is at or above the threshold {threshold}"
        )
    return {
        "rejected": {
            "row": rejected,
            "identifier": _number(identifiers[rejected]),
            "y_prob": float(values[rejected]),
        },
        "approved": {
            "row": approved,
            "identifier": _number(identifiers[approved]),
            "y_prob": float(values[approved]),
        },
    }


def reason_codes(
    contributions: pd.Series,
    values: pd.Series,
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the ``k`` features that pushed this applicant's risk up the most.

    Positive TreeSHAP values increase the default log-odds, which is what a
    rejected applicant's adverse-action reasons must cite. When fewer than ``k``
    features are positive, the remaining slots are filled by absolute size so the
    list always has ``k`` entries.
    """
    ordered = contributions.sort_values(ascending=False)
    positive = ordered[ordered > 0].head(k)
    if len(positive) < k:
        remainder = ordered.drop(index=positive.index)
        extra = remainder.reindex(remainder.abs().sort_values(ascending=False).index).head(
            k - len(positive)
        )
        positive = pd.concat([positive, extra])

    codes: list[dict[str, Any]] = []
    for rank, (feature, shap_value) in enumerate(positive.items(), start=1):
        value = values.get(feature)
        codes.append(
            {
                "rank": rank,
                "feature": str(feature),
                "value": _number(value) if not isinstance(value, (str, bool)) else value,
                "shap": float(shap_value),
                "direction": "increases risk" if shap_value > 0 else "decreases risk",
                "reason": (
                    f"{feature} = {value} moved the predicted default log-odds "
                    f"{'up' if shap_value > 0 else 'down'} by {abs(float(shap_value)):.4f}"
                ),
            }
        )
    return codes


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def write_summary_figure(explanation: Any, top_k: int, *, path: Path, title: str) -> Path:
    """Save a beeswarm summary of the top features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.beeswarm(explanation, max_display=int(top_k), show=False)
    figure = plt.gcf()
    figure.suptitle(title, fontsize=9)
    return _save(figure, path)


def write_dependence_figure(
    explanation: Any,
    features: pd.DataFrame,
    importance: pd.Series,
    *,
    top: int,
    path: Path,
) -> Path:
    """Save one SHAP-vs-feature-value panel per top feature."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importance.index[:top])
    figure, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.6), squeeze=False)
    contributions = pd.DataFrame(
        explanation.values, columns=explanation.feature_names, index=features.index
    )
    for axis, feature in zip(axes.ravel(), names, strict=False):
        axis.scatter(
            features[feature].to_numpy(dtype="float64"),
            contributions[feature].to_numpy(dtype="float64"),
            s=6,
            alpha=0.35,
            linewidths=0,
        )
        axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axis.set_title(feature, fontsize=9)
        axis.set_xlabel("Feature value")
        axis.set_ylabel("SHAP contribution (log-odds)")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return _save(figure, path)


def write_waterfall(
    explanation: Any,
    row: int,
    *,
    path: Path,
    title: str,
    max_display: int,
) -> Path:
    """Save a local waterfall for one row of the explanation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.waterfall(explanation[row], max_display=int(max_display), show=False)
    figure = plt.gcf()
    figure.suptitle(title, fontsize=9)
    return _save(figure, path)


def run_explanation(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    model_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit the frozen model, explain a hold-out sample and write the E7 artifacts."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    seed = int(get_nested(config, "explain.seed", model_seeds(config)[0]))
    set_seeds(seed)

    name = str(model_name or get_nested(config, "explain.model", "lightgbm")).strip().lower()
    definitions = model_definitions(config)
    if name not in definitions:
        raise ValueError(f"Unknown model {name!r}; declared: {sorted(definitions)}")

    contract = load_contract(resolve_path(config, "dataset.contract"))
    identifier_column = str(contract["identifier_column"])
    fold_column = str(contract["fold_column"])
    target_column = str(contract["target_column"])
    predictors = resolve_feature_columns(config, contract)
    categorical = [name_ for name_ in predictors if name_ in set(contract["categorical_columns"])]

    paths = {
        "importance": resolve_path(config, "explain.importance_table"),
        "reason_table": resolve_path(config, "explain.reason_table"),
        "summary": resolve_path(config, "explain.summary_figure"),
        "dependence": resolve_path(config, "explain.dependence_figure"),
        "record": resolve_path(config, "explain.record"),
    }
    paths["record"] = Path(
        str(paths["record"]).format(exp=get_nested(config, "outputs.exp", "E7_explain"))
    )
    waterfall_dir = resolve_path(config, "explain.waterfall_dir")
    waterfall_paths = {
        "approved": waterfall_dir / "E7_shap_waterfall_approved.png",
        "rejected": waterfall_dir / "E7_shap_waterfall_rejected.png",
    }

    existing = [str(path) for path in [*paths.values(), *waterfall_paths.values()] if path.exists()]
    if existing and not force:
        raise FileExistsError(f"E7 artifacts already exist (use --force to overwrite): {existing}")

    development = load_split_frame(
        config,
        "train",
        contract,
        columns=[identifier_column, target_column, fold_column, *predictors],
        n_folds=int(get_nested(config, "split.n_folds", 5)),
    )
    holdout = load_holdout_features(config, contract, columns=[identifier_column, *predictors])

    n_rows = int(sample if sample is not None else get_nested(config, "explain.sample", 10_000))
    positions = sample_positions(len(holdout), n_rows, seed=seed)
    features = holdout.iloc[positions].loc[:, predictors]
    identifiers = holdout.iloc[positions][identifier_column].to_numpy()
    LOGGER.info(
        "Explaining %d hold-out rows (features only) with %s, seed=%d, %d features",
        len(features),
        name,
        seed,
        len(predictors),
    )

    model = fit_development_model(
        config,
        definitions[name],
        development.loc[:, predictors],
        development[target_column].to_numpy(dtype="int8"),
        categorical_features=categorical,
        seed=seed,
    )
    import shap

    probabilities = model.predict_proba(features)
    contributions = model.shap_contributions(features)
    expected_value = base_value(probabilities, contributions)

    plot_features = numeric_frame(features, categorical)
    explanation = shap.Explanation(
        values=contributions.to_numpy(dtype="float64"),
        base_values=np.full(len(features), expected_value),
        data=plot_features.to_numpy(dtype="float64"),
        feature_names=list(plot_features.columns),
    )

    importance = (
        contributions.abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .to_frame()
    )
    importance["mean_shap"] = contributions.mean(axis=0)
    importance["share"] = importance["mean_abs_shap"] / float(importance["mean_abs_shap"].sum())
    importance.index.name = "feature"
    importance = importance.reset_index()
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    threshold = float(get_nested(config, "explain.threshold", 0.5))
    applicants = select_applicants(probabilities, identifiers, threshold=threshold)
    k_codes = int(get_nested(config, "explain.reason_codes", 5))
    codes = reason_codes(
        contributions.iloc[applicants["rejected"]["row"]],
        features.iloc[applicants["rejected"]["row"]],
        k=k_codes,
    )
    for code in codes:
        code["applicant_id"] = applicants["rejected"]["identifier"]
        code["y_prob"] = applicants["rejected"]["y_prob"]

    paths["importance"].parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(paths["importance"], index=False)
    pd.DataFrame(codes).to_csv(paths["reason_table"], index=False)

    top_k = int(get_nested(config, "explain.top_k", 20))
    top_features = list(importance["feature"].head(top_k))
    write_summary_figure(
        explanation,
        top_k,
        path=paths["summary"],
        title=f"E7 SHAP summary — {name}, {len(features)} hold-out applicants",
    )
    write_dependence_figure(
        explanation,
        plot_features,
        importance.set_index("feature")["mean_abs_shap"],
        top=3,
        path=paths["dependence"],
    )
    for label, figure_path in waterfall_paths.items():
        applicant = applicants[label]
        write_waterfall(
            explanation,
            applicant["row"],
            path=figure_path,
            title=(
                f"{label} applicant {applicant['identifier']} "
                f"(p={applicant['y_prob']:.4f}, threshold {threshold})"
            ),
            max_display=min(10, len(predictors)),
        )

    record = {
        "experiment": get_nested(config, "experiment", "E7_explain"),
        "model": name,
        "seed": seed,
        "n_sample": int(len(features)),
        "n_holdout": int(len(holdout)),
        "n_features": len(predictors),
        "expected_value_log_odds": expected_value,
        "threshold": threshold,
        "applicants": applicants,
        "top_features": top_features,
        "reason_codes": codes,
        "artifacts": {
            **{key: str(path) for key, path in paths.items()},
            **{f"waterfall_{key}": str(path) for key, path in waterfall_paths.items()},
        },
        "note": (
            "Hold-out features only; no hold-out label was read. Global importance is the "
            "mean absolute TreeSHAP contribution over the explained sample."
        ),
    }
    write_json(paths["record"], record)
    LOGGER.info(
        "Top features: %s",
        ", ".join(
            f"{row['feature']}={row['mean_abs_shap']:.4f}"
            for _, row in importance.head(5).iterrows()
        ),
    )
    return {"record": paths["record"], **{key: str(path) for key, path in paths.items()}}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain the frozen model with TreeSHAP.")
    parser.add_argument("--config", required=True, help="E7 explanation YAML config")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Hold-out rows to explain (default: explain.sample from the config).",
    )
    parser.add_argument("--model", default=None, help="Model to explain (default: explain.model)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = run_explanation(config, sample=args.sample, model_name=args.model, force=args.force)
    except Exception:
        setup_logging()
        LOGGER.exception("E7 SHAP explanation failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("E7 %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
