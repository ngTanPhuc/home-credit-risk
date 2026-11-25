"""Phase 9: the single final hold-out evaluation.

This is the only module allowed to pass ``allow_holdout=True``. It runs **once
per project**: the hold-out is the audit set, and a second look invalidates it.

What it does, in order
----------------------
1. Refuses to run without ``--confirm-final``.
2. Refuses to run when its outputs already exist, so a second evaluation needs
   a deliberate file deletion and a decision review.
3. Fits every E5 model on *all* development rows with the E4 ``final_params``
   and the E3 imbalance variant (no early stopping, no validation pair).
4. Rebuilds the ensemble arms on the hold-out: a logistic stack whose
   meta-learner is fitted on development OOF rows, and the log-loss-weighted
   average with development-fitted weights.
5. Fits the frozen calibrator on development OOF predictions for each arm and
   applies it to the hold-out, then reports Brier/ECE before vs after.

Outputs
-------
``B3_calibration.csv``
    Hold-out metrics per arm for the uncalibrated and the frozen calibration.
``B4_delong_pairs.csv``
    Pairwise DeLong tests across arms on the hold-out labels.
``holdout_final.json``
    Protocol record: choice, row counts, git commit, paths, final metrics.
``holdout_reliability.png``
    Reliability curves, before vs after calibration.

Usage
-----
``uv run python -m src.evaluation.holdout --config configs/experiments/holdout_final.yaml --confirm-final``
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
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    CALIBRATORS,
    build_calibrator,
    calibration_metrics,
    reliability_curve,
)
from src.config import (
    config_digest,
    get_nested,
    load_config,
    public_config,
    resolve_path,
    set_seeds,
)
from src.cv.dataset import load_contract, load_split_frame, resolve_feature_columns
from src.evaluation.delong import delong_test
from src.evaluation.model_comparison import (
    ensembles,
    load_seed_oof,
    model_definitions,
    model_seeds,
)
from src.models.ensemble import (
    OOFMatrix,
    cross_fitted_average,
    cross_fitted_stacking,
    fit_average_weights,
)
from src.models.final import fit_development_model
from src.utils.logging import git_commit, setup_logging, write_json

LOGGER = logging.getLogger(__name__)

CONFIRM_FLAG = "--confirm-final"

_EPSILON = 1e-9


def final_choice(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen ``final`` block, validated against the declared arms."""
    arm = str(get_nested(config, "final.arm", "")).strip()
    calibrator = str(get_nested(config, "final.calibrator", "none")).strip().lower()
    declared_models = [str(name) for name in (get_nested(config, "comparison.models", {}) or {})]
    if not declared_models:
        raise ValueError("comparison.models is empty; the hold-out needs at least one model")

    allowed_arms = [*declared_models, *ensembles(config)]
    if arm not in allowed_arms:
        raise ValueError(f"final.arm {arm!r} is not one of {allowed_arms}")
    if calibrator not in CALIBRATORS:
        raise ValueError(f"final.calibrator must be one of {list(CALIBRATORS)}, got {calibrator!r}")

    seeds = model_seeds(config)
    seed = int(get_nested(config, "final.seed", seeds[0]))
    if seed not in seeds:
        raise ValueError(f"final.seed {seed} is not one of seeds.model {seeds}")
    return {"arm": arm, "calibrator": calibrator, "seed": seed}


def output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every hold-out artifact path, rendering the ``{exp}`` template."""
    exp = get_nested(config, "outputs.exp", "holdout_final")
    paths: dict[str, Path] = {}
    for key in ("table", "delong_table", "figure", "record"):
        rendered = str(resolve_path(config, f"holdout.{key}")).format(exp=exp)
        paths[key] = Path(rendered)
    return paths


def refuse_existing_outputs(paths: Mapping[str, Any]) -> None:
    """Raise if any hold-out artifact already exists.

    A second hold-out run is a protocol violation, so this is deliberately not
    overridable by a flag: deleting the files is a visible, reviewable act.
    """
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "The hold-out was already evaluated; refusing to run again "
            f"(existing artifacts: {existing}). A second run is a protocol violation."
        )


def _arm_development_predictions(
    config: Mapping[str, Any],
    matrix: OOFMatrix,
    arm: str,
    *,
    seed: int,
) -> np.ndarray | None:
    """Return an ensemble arm's development OOF probabilities, or ``None``."""
    if arm == "weighted_average":
        return cross_fitted_average(
            matrix, step=float(get_nested(config, "comparison.weight_step", 0.05))
        ).y_prob
    if arm == "stacking":
        return cross_fitted_stacking(matrix, seed=int(seed)).y_prob
    return None


def _arm_holdout_predictions(
    config: Mapping[str, Any],
    matrix: OOFMatrix,
    arm_probabilities: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Predict every arm on the hold-out from the full-development fits."""
    arms = {
        str(name): arm_probabilities[name].to_numpy(dtype="float64")
        for name in arm_probabilities.columns
    }
    development_target = matrix.meta[matrix.target_column].to_numpy()
    declared = ensembles(config)

    if "stacking" in declared:
        meta = LogisticRegression(max_iter=1000, random_state=int(seed))
        meta.fit(matrix.probabilities, development_target)
        arms["stacking"] = np.asarray(
            meta.predict_proba(arm_probabilities.loc[:, matrix.probabilities.columns])[:, 1],
            dtype="float64",
        )

    if "weighted_average" in declared:
        weights = fit_average_weights(
            development_target,
            matrix.probabilities,
            step=float(get_nested(config, "comparison.weight_step", 0.05)),
        )
        ordered = np.array(
            [weights[str(name)] for name in matrix.probabilities.columns], dtype="float64"
        )
        arms["weighted_average"] = (
            arm_probabilities.loc[:, matrix.probabilities.columns].to_numpy(dtype="float64")
            @ ordered
        )

    return arms


def _metrics_row(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    arm: str,
    calibrator: str,
    is_final: bool,
    n_fit_rows: int,
    thresholds: Sequence[float],
    ece_bins: int,
) -> dict[str, Any]:
    metrics = calibration_metrics(target, probabilities, thresholds=thresholds, ece_bins=ece_bins)
    return {
        "arm": arm,
        "calibrator": calibrator,
        "is_final": bool(is_final),
        "n_rows": int(len(target)),
        "n_fit_rows": int(n_fit_rows),
        **{key: float(value) for key, value in metrics.items()},
    }


def build_b3_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Add deltas against each arm's uncalibrated hold-out row."""
    table = pd.DataFrame(rows)
    reference = (
        table.loc[table["calibrator"] == "none", ["arm", "brier", "ece", "log_loss"]]
        .set_index("arm")
        .rename(columns={"brier": "_b", "ece": "_e", "log_loss": "_l"})
    )
    table = table.join(reference, on="arm")
    table["delta_brier_vs_none"] = table["brier"] - table["_b"]
    table["delta_ece_vs_none"] = table["ece"] - table["_e"]
    table["delta_log_loss_vs_none"] = table["log_loss"] - table["_l"]
    return table.drop(columns=["_b", "_e", "_l"])


def build_b4_table(target: np.ndarray, arms: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise DeLong tests on the hold-out rows.

    Calibration is monotone, so it cannot change these AUCs; the table therefore
    tests the uncalibrated arm probabilities.
    """
    names = list(arms)
    rows: list[dict[str, Any]] = []
    for index, name_a in enumerate(names):
        for name_b in names[index + 1 :]:
            result = delong_test(target, arms[name_a], arms[name_b])
            rows.append(
                {
                    "arm_a": name_a,
                    "arm_b": name_b,
                    "auc_a": result.auc_a,
                    "auc_b": result.auc_b,
                    "delta_auc": result.difference,
                    "z": result.z,
                    "p_value": result.p_value,
                    "significant_at_0_05": bool(result.p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def write_reliability_figure(
    target: np.ndarray,
    arms: Mapping[str, np.ndarray],
    calibrated: Mapping[str, np.ndarray],
    *,
    calibrator: str,
    path,
    ece_bins: int,
) -> Any:
    """Plot one reliability panel per arm, uncalibrated vs the frozen calibrator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(arms)
    columns = min(3, max(1, len(names)))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.6 * rows), squeeze=False)

    for axis, name in zip(axes.ravel(), names, strict=False):
        for label, probabilities, color in (
            ("none", arms[name], "#1f77b4"),
            (calibrator, calibrated[name], "#d62728"),
        ):
            curve = reliability_curve(target, probabilities, ece_bins)
            axis.plot(
                curve["predicted"],
                curve["observed"],
                "o-",
                color=color,
                markersize=3.5,
                linewidth=1.0,
                label=label,
            )
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=0.8, color="grey")
        axis.set_title(name)
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Observed default rate")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    for axis in axes.ravel()[len(names) :]:
        axis.set_visible(False)

    figure.suptitle(f"Final hold-out reliability ({ece_bins} equal-count bins)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_holdout(
    config: Mapping[str, Any],
    *,
    confirm_final: bool = False,
) -> dict[str, Any]:
    """Evaluate the frozen model and calibrator on the hold-out, exactly once."""
    if not confirm_final:
        raise PermissionError(
            f"Refusing to touch the hold-out without {CONFIRM_FLAG}. "
            "This is the single final evaluation; confirm the frozen choice first."
        )

    paths = output_paths(config)
    refuse_existing_outputs(paths)

    setup_logging(get_nested(config, "logging.level", "INFO"))
    choice = final_choice(config)
    seed = int(choice["seed"])
    set_seeds(seed)

    contract = load_contract(resolve_path(config, "dataset.contract"))
    identifier_column = str(contract["identifier_column"])
    target_column = str(contract["target_column"])
    fold_column = str(contract["fold_column"])
    predictors = resolve_feature_columns(config, contract)
    categorical_features = [
        name for name in predictors if name in set(contract["categorical_columns"])
    ]

    development = load_split_frame(
        config,
        "train",
        contract,
        columns=[identifier_column, target_column, fold_column, *predictors],
        n_folds=int(get_nested(config, "split.n_folds", 5)),
    )
    holdout = load_split_frame(
        config,
        "val",
        contract,
        columns=[identifier_column, target_column, *predictors],
        allow_holdout=True,
    )
    development_target = development[target_column].to_numpy(dtype="int8")
    holdout_target = holdout[target_column].to_numpy(dtype="int8")
    LOGGER.warning(
        "FINAL hold-out evaluation: %d development rows fitted, %d hold-out rows scored",
        len(development),
        len(holdout),
    )

    definitions = model_definitions(config)
    matrix = load_seed_oof(
        config,
        list(definitions),
        seed=seed,
        sample=None,
        identifier_column=identifier_column,
        target_column=target_column,
        fold_column=fold_column,
    )

    holdout_probabilities: dict[str, np.ndarray] = {}
    for name, block in definitions.items():
        LOGGER.info("Fitting %s on the full development split (seed=%d)", name, seed)
        model = fit_development_model(
            config,
            block,
            development.loc[:, predictors],
            development_target,
            categorical_features=categorical_features,
            seed=seed,
        )
        holdout_probabilities[name] = model.predict_proba(holdout.loc[:, predictors])

    arm_probabilities = pd.DataFrame(holdout_probabilities, index=holdout.index)
    arms = _arm_holdout_predictions(config, matrix, arm_probabilities, seed=seed)

    thresholds = tuple(get_nested(config, "metrics.thresholds", (0.5,)))
    ece_bins = int(get_nested(config, "metrics.ece_bins", 15))
    n_development = int(len(matrix.meta))

    rows: list[dict[str, Any]] = []
    calibrated_vectors: dict[str, np.ndarray] = {}
    for arm, probabilities in arms.items():
        development_probabilities = _arm_development_predictions(config, matrix, arm, seed=seed)
        if development_probabilities is None:
            development_probabilities = matrix.probabilities[arm].to_numpy(dtype="float64")

        rows.append(
            _metrics_row(
                holdout_target,
                probabilities,
                arm=arm,
                calibrator="none",
                is_final=False,
                n_fit_rows=n_development,
                thresholds=thresholds,
                ece_bins=ece_bins,
            )
        )

        calibrator = build_calibrator(choice["calibrator"]).fit(
            matrix.meta[matrix.target_column].to_numpy(), development_probabilities
        )
        calibrated_vectors[arm] = calibrator.predict(probabilities)
        rows.append(
            _metrics_row(
                holdout_target,
                calibrated_vectors[arm],
                arm=arm,
                calibrator=choice["calibrator"],
                is_final=bool(arm == choice["arm"]),
                n_fit_rows=n_development,
                thresholds=thresholds,
                ece_bins=ece_bins,
            )
        )

    table = build_b3_table(rows)
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table"], index=False)

    delong = build_b4_table(holdout_target, arms)
    paths["delong_table"].parent.mkdir(parents=True, exist_ok=True)
    delong.to_csv(paths["delong_table"], index=False)

    write_reliability_figure(
        holdout_target,
        arms,
        calibrated_vectors,
        calibrator=choice["calibrator"],
        path=paths["figure"],
        ece_bins=ece_bins,
    )

    final_rows = table.loc[table["is_final"]]
    if final_rows.empty:
        raise RuntimeError(f"Frozen arm {choice['arm']!r} is missing from the hold-out table")
    final_row = final_rows.iloc[0].to_dict()

    record = {
        "experiment": get_nested(config, "experiment", "holdout_final"),
        "git_commit": git_commit(get_nested(config, "_meta.project_root", ".")),
        "config_digest": config_digest(config),
        "config": public_config(config),
        "choice": choice,
        "arms": list(arms),
        "n_development_rows": n_development,
        "n_holdout_rows": int(len(holdout)),
        "n_holdout_positives": int(holdout_target.sum()),
        "metrics": {key: float(value) for key, value in final_row.items() if _is_number(value)},
        "final_metrics": final_row,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    record_path = paths["record"]
    write_json(record_path, record)

    LOGGER.info(
        "FINAL hold-out: arm=%s calibrator=%s AUC=%.5f Brier=%.5f ECE=%.5f",
        choice["arm"],
        choice["calibrator"],
        final_row["roc_auc"],
        final_row["brier"],
        final_row["ece"],
    )
    return {"table": paths["table"], "delong_table": paths["delong_table"], "record": record_path}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen model and calibrator on the hold-out, exactly once."
    )
    parser.add_argument("--config", required=True, help="Hold-out experiment YAML config")
    parser.add_argument(
        CONFIRM_FLAG,
        dest="confirm_final",
        action="store_true",
        help="Required. Confirms the frozen model and calibrator are final.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = run_holdout(config, confirm_final=args.confirm_final)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        setup_logging()
        LOGGER.exception("Final hold-out evaluation failed")
        return 1
    for key, path in paths.items():
        LOGGER.info("Hold-out %s: %s", key, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
