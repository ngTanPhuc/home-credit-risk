"""Score ``test.parquet`` with the frozen final model and write a Kaggle submission.

The predictor is exactly the pair frozen in the config (default
``stacking`` + ``isotonic``): the E4-tuned base models refitted on all development
rows, the logistic stack fitted on development OOF predictions, and the
calibrator fitted on the arm's cross-fitted development OOF vector. No test row
and no hold-out label enters the fit, and no metric here is used to choose
anything — model selection ended in Phase 8.

Outputs
-------
``reports/submissions/submission_<exp>.csv``
    Two columns, ``SK_ID_CURR`` and ``TARGET``, one row per test applicant.
``reports/runs/<exp>/final_model.joblib``
    The deployable bundle (``--save-model``), loadable with
    ``FinalModel.load(path)`` and reusable via ``--reuse-model``.
``reports/runs/<exp>/submission.json``
    Record: frozen choice, seeds, row counts, prediction summary, artifacts.

Usage
-----
``uv run python -m src.evaluation.submit --config configs/experiments/holdout_final.yaml``

``--sample N`` fits on N development rows and scores N test rows (smoke run);
``--save-model`` writes the bundle, ``--reuse-model`` skips the ~5 min refit.
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

from src.config import (
    config_digest,
    get_nested,
    load_config,
    public_config,
    resolve_path,
    set_seeds,
)
from src.cv.dataset import load_contract, load_split_frame, resolve_feature_columns
from src.evaluation.holdout import final_choice
from src.evaluation.model_comparison import load_seed_oof, model_definitions
from src.models.ensemble import OOFMatrix
from src.models.final import FinalModel, fit_final_model
from src.utils.logging import git_commit, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _render(config: Mapping[str, Any], key: str) -> Path:
    rendered = str(resolve_path(config, f"submission.{key}")).format(
        exp=get_nested(config, "outputs.exp", "submission")
    )
    return Path(rendered)


def _subset_oof(
    matrix: OOFMatrix,
    mask: np.ndarray,
    *,
    identifier_column: str,
    target_column: str,
    fold_column: str,
) -> OOFMatrix:
    """Return the OOF matrix restricted to ``mask`` rows, keeping its columns."""
    return OOFMatrix(
        meta=matrix.meta.loc[mask].reset_index(drop=True),
        probabilities=matrix.probabilities.loc[mask].reset_index(drop=True),
        identifier_column=identifier_column,
        target_column=target_column,
        fold_column=fold_column,
    )


def _smoke_development(
    development: pd.DataFrame,
    sample: int,
    *,
    fold_column: str,
    seed: int,
) -> pd.DataFrame:
    """Take ``sample`` development rows spread over the folds."""
    if sample >= len(development):
        return development
    per_fold = max(1, sample // max(1, development[fold_column].nunique()))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for fold in sorted(development[fold_column].unique().tolist()):
        positions = np.flatnonzero(development[fold_column].to_numpy() == fold)
        take = min(per_fold, positions.size)
        chosen.append(rng.choice(positions, size=take, replace=False))
    selected = development.iloc[np.sort(np.concatenate(chosen))].reset_index(drop=True)
    LOGGER.warning(
        "Smoke mode: fitting on %d development rows and scoring %d test rows",
        len(selected),
        sample,
    )
    return selected


def run_submission(
    config: Mapping[str, Any],
    *,
    sample: int | None = None,
    out: str | Path | None = None,
    save_model: str | Path | None = None,
    reuse_model: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit (or load) the frozen model, score the test split and write the CSV."""
    setup_logging(get_nested(config, "logging.level", "INFO"))
    choice = final_choice(config)
    seed = int(choice["seed"])
    set_seeds(seed)

    contract = load_contract(resolve_path(config, "dataset.contract"))
    identifier_column = str(contract["identifier_column"])
    target_column = str(contract["target_column"])
    fold_column = str(contract["fold_column"])
    predictors = resolve_feature_columns(config, contract)
    categorical = [name for name in predictors if name in set(contract["categorical_columns"])]

    out_path = Path(out) if out is not None else _render(config, "out")
    record_path = _render(config, "record")
    if sample is not None:
        out_path = out_path.with_name(f"{out_path.stem}_sample{sample}{out_path.suffix}")
        record_path = record_path.with_name(
            f"{record_path.stem}_sample{sample}{record_path.suffix}"
        )

    existing = [str(path) for path in (out_path, record_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"Submission artifacts exist (use --force): {existing}")

    development: pd.DataFrame | None = None
    bundle_path: Path | None = None
    if reuse_model is not None:
        model = FinalModel.load(reuse_model)
        bundle_path = Path(reuse_model)
        LOGGER.info("Reusing the saved bundle: %s", bundle_path)
    else:
        definitions = model_definitions(config)
        development = load_split_frame(
            config,
            "train",
            contract,
            columns=[identifier_column, target_column, fold_column, *predictors],
            n_folds=int(get_nested(config, "split.n_folds", 5)),
        )
        oof = load_seed_oof(
            config,
            list(definitions),
            seed=seed,
            sample=None,
            identifier_column=identifier_column,
            target_column=target_column,
            fold_column=fold_column,
        )
        if sample is not None:
            development = _smoke_development(
                development, int(sample), fold_column=fold_column, seed=seed
            )
            mask = oof.meta[identifier_column].isin(development[identifier_column]).to_numpy()
            oof = _subset_oof(
                oof,
                mask,
                identifier_column=identifier_column,
                target_column=target_column,
                fold_column=fold_column,
            )

        model = fit_final_model(
            config,
            definitions,
            development.loc[:, predictors],
            development[target_column].to_numpy(dtype="int8"),
            oof=oof,
            arm=str(choice["arm"]),
            calibrator=str(choice["calibrator"]),
            predictor_columns=predictors,
            categorical_features=categorical,
            seed=seed,
            weight_step=float(get_nested(config, "comparison.weight_step", 0.05)),
        )
        if save_model is not None:
            bundle_path = model.save(Path(save_model))
            LOGGER.info("Saved the final model bundle: %s", bundle_path)

    test = load_split_frame(config, "test", contract, columns=[identifier_column, *predictors])
    if sample is not None and sample < len(test):
        test = test.head(int(sample)).reset_index(drop=True)
    if test[identifier_column].duplicated().any():
        raise ValueError("test.parquet has duplicated identifiers")

    probabilities = np.asarray(model.predict_proba(test.loc[:, predictors]), dtype="float64")
    if probabilities.shape[0] != len(test):
        raise ValueError("prediction length does not match the test frame")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("submission contains non-finite probabilities")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(
            f"probabilities outside [0, 1]: {probabilities.min():.6f}..{probabilities.max():.6f}"
        )

    submission = pd.DataFrame(
        {identifier_column: test[identifier_column].to_numpy(), target_column: probabilities}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    summary = {
        "n_rows": int(len(submission)),
        "mean_probability": float(probabilities.mean()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
        "share_above_0_5": float((probabilities >= 0.5).mean()),
        "share_above_0_2": float((probabilities >= 0.2).mean()),
    }
    record = {
        "experiment": get_nested(config, "experiment", "submission"),
        "git_commit": git_commit(get_nested(config, "_meta.project_root", ".")),
        "config_digest": config_digest(config),
        "config": public_config(config),
        "choice": choice,
        "n_features": len(predictors),
        "n_development_rows": None if development is None else int(len(development)),
        "n_test_rows": int(len(test)),
        "sample": sample,
        "model_bundle": None if bundle_path is None else str(bundle_path),
        "artifacts": {"submission": str(out_path), "record": str(record_path)},
        "prediction_summary": summary,
        "development_positive_rate": (
            None
            if development is None
            else float(development[target_column].astype("float64").mean())
        ),
        "note": (
            "Test-split predictions of the frozen model; no test label exists. "
            "The development positive rate is the expected mean; a large shift means "
            "the feature pipeline changed, not that the model improved."
        ),
    }
    write_json(record_path, record)

    LOGGER.info(
        "Submission: %d rows, mean p=%.5f, min=%.5f, max=%.5f -> %s",
        summary["n_rows"],
        summary["mean_probability"],
        summary["min_probability"],
        summary["max_probability"],
        out_path,
    )
    return {"submission": str(out_path), "record": str(record_path), **summary}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score test.parquet with the frozen model.")
    parser.add_argument("--config", required=True, help="Hold-out/submission YAML config")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Smoke run: fit on N development rows and score N test rows.",
    )
    parser.add_argument("--out", default=None, help="Submission CSV path (default: config).")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also write the joblib bundle (no value: the configured submission.model path).",
    )
    parser.add_argument(
        "--reuse-model",
        default=None,
        help="Load a saved bundle instead of refitting (skips the multi-minute fit).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = run_submission(
            config,
            sample=args.sample,
            out=args.out,
            save_model=(
                None if args.save_model is None else (args.save_model or _render(config, "model"))
            ),
            reuse_model=args.reuse_model,
            force=args.force,
        )
    except Exception:
        setup_logging()
        LOGGER.exception("Submission failed")
        return 1
    for key in ("submission", "record"):
        LOGGER.info("Submission %s: %s", key, paths[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
