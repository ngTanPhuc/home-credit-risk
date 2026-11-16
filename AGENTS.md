# AGENTS.md

`INIT.md` is the authoritative spec (phases, ownership, schema, hard rules); `README.md` covers setup and the intended CLI. Read `INIT.md` before changing anything and ask the human when a request conflicts with it.

## Current state (read this first)

- Person A's pipeline is `notebooks/01..04`. Those notebooks are standalone by design ("Notebook độc lập, không gọi `src/`") and own everything under `data/`. INIT's "notebooks are analysis-only and import from `src/`" rule describes later analysis notebooks, not these four.
- The frozen modelling input exists on this machine: `data/processed/datasets/{train,val,test}.parquet` (246,008 / 61,503 / 48,744 rows), `feature_contract.json` (version `datasets_v1`, 180 predictors, 16 categorical), `dataset_manifest.json`. They are gitignored and absent from a clone.
- Implemented in `src/`: `config.py`, `utils/{logging,memory}.py`, `cv/{dataset,splits,runner}.py`, `models/{base,lgbm}.py`, `evaluation/metrics.py`. `tests/` has 28 passing tests on synthetic data.
- Still 0-byte stubs on purpose, filled only when their phase is requested: `calibration.py`, `imbalance.py`, `tuning/optuna_search.py`, `evaluation/{delong,holdout}.py`, `explain/*`, `models/{xgb,catboost_model,ensemble}.py`.
- `graphify-out/` predates the rewrite of `src/`. Don't trust it for architecture.
- Notebook prose, comments and print strings are Vietnamese; `src/` docstrings are English. Match the file you are editing; do not mass-translate.

## Data contract and split protocol

- Canonical split: `data/splits/train.parquet` (`SK_ID_CURR`, `fold`), `data/splits/val.parquet` (`SK_ID_CURR`) and `data/splits/train_val_metadata.json` (hold-out seed **2026**, CV seed 42, 20% hold-out). Notebook 01 creates it; notebook 04 copies `fold` into `data/processed/datasets/train.parquet`.
- `splits/folds.parquet` no longer exists. The notebook protocol is the only one. Never re-split, reshuffle or regenerate it; the notebooks raise when a manifest or `source_sha256` does not match.
- `src/cv/splits.py` only validates `train_val_metadata.json` against dataset row counts. It cannot create a split.
- `data/processed/datasets/feature_contract.json` is the schema source of truth: predictor order, categorical columns, per-split column lists, excluded keys. `src/cv/dataset.py` validates every load against it.
- `configs/features.yaml` maps group to columns and defines the cumulative ablation sets. It is generated from `reports/tables/feature_catalog.csv` in contract order; regenerate it when the dataset changes, don't hand-edit.
- `load_split_frame(..., split="val")` raises `PermissionError` unless `allow_holdout=True`. Only `src/evaluation/holdout.py` may pass it.
- Delivered groups: `application_original` 120, `application` 10, `bureau` 17, `previous` 10, `pos` 6, `credit_card` 8, `installments` 9 = 180. No time-window (`WIN_`) or in-fold selection group was delivered; do not invent them.
- E1 uses set `main` (the 120 raw application columns). The 10 ratios enter E2 at the second rung of the ladder.

## Notebook data flow

- `01_data_overview_eda`: `data/raw/*.csv` -> `data/interim/<table>.parquet`, and creates the split in `data/splits/`.
- `02_data_cleaning`: -> `data/processed/<table>.parquet` + `cleaning_manifest.json` (rule signature + per-file SHA-256).
- `03_feature_engineering`: -> `data/processed/features/{application_train,application_test,bureau,previous,installment}_features.parquet` + `feature_manifest.json`, `feature_catalog.csv`.
- `04_dataset_assembly_validation`: -> `data/processed/datasets/{train,val,test}.parquet` + `dataset_manifest.json`, `feature_contract.json`.
- Every notebook locates the repo root via `HOME_CREDIT_ROOT`, else walks up until it finds `data/raw/`; run them from the repo root. They abort when a manifest/hash does not match, so never hand-edit files under `data/interim` or `data/processed`.
- CSV reports go to `reports/tables/`; notebooks display figures with matplotlib but do not write files to `reports/figures/`.

## Commands

```bash
~/.local/bin/uv sync                 # uv may be missing from PATH in non-interactive shells
~/.local/bin/uv run pytest -q
~/.local/bin/uv run pytest tests/test_metrics.py -q
~/.local/bin/uv run pytest tests/test_metrics.py -k ece -q
~/.local/bin/uv run ruff check .     # lints notebooks too
~/.local/bin/uv run ruff format --check .
~/.local/bin/uv run python -m src.cv.runner --config configs/experiments/E1_baseline.yaml --sample 500
```

- Use `uv run` / `uv add`; never `pip` or a bare `python`. `pyproject.toml` sets `package = false`, so modules run as `python -m src.<module>` from the repo root.
- `pytest` runs with `--strict-markers --strict-config`: an unregistered marker (`slow`, `integration`) or stray config key is an error.
- Ruff is configured for `py313`, line length 100, rules `E,F,I,UP,B` (E501 ignored), `src` and `tests` first-party. `ruff check .` still reports findings inside the notebooks; don't blanket `--fix` files you are not working on.
- `--sample N` writes `*_sampleN`-suffixed outputs; delete them after a smoke run.

## Hard stops

- **No long jobs.** Never launch full feature builds, tuning, full training, or SHAP over the full dataset. Write the code and an exact command, then ask the human to run it on Kaggle. State expected runtime and memory.
- **No git writes.** Never commit, push, merge, branch, or change git config. Suggest a branch name and commit message instead.
- **No data acquisition.** Never download data, install system packages, or modify `data/raw/`.
- **Hold-out is locked.** Only `src/evaluation/holdout.py --confirm-final` may read hold-out labels; all development uses CV.
- **No silent dependencies.** Use `uv add` and tell the human why.
- **No deep learning** (MLP, TabNet, FT-Transformer) unless explicitly approved.

## Invariants that are easy to break

- **Feature schema contract:** each group produces one row per `SK_ID_CURR`, keeps its prefix (`APP_`, `BUREAU_`, `PREV_`, `POS_`, `CC_`, `INS_`, `HAS_`), contains no `TARGET`, and uses no fold information. Any column-name or key change is a schema change: update `DECISIONS.md` and tell the human.
- **Per-fold fit only:** target encoding, SMOTE/undersampling, scaling, feature selection and calibration are fit on the training part of each fold and applied to validation. Enforce it with fit/transform objects; never fit on the full development set.
- **Memory:** 16 GB is a hard ceiling. `usecols`, explicit dtypes, chunked reads and per-key aggregation for the 10M+ row tables (`bureau_balance`, `POS_CASH_balance`, `installments_payments`), `float32`/`int32`, free intermediates, Parquet between steps.
- **Reproducibility:** seed from config, log the resolved config and a read-only git commit to MLflow, deterministic output paths. No magic numbers, paths or file names.
- **Metrics:** ROC-AUC primary; also Gini, KS, PR-AUC, Brier, ECE, threshold precision/recall. `compute_metrics` wraps the human's `all_metrics`; do not rewrite that function's body.
- **Kaggle paths differ** (`/kaggle/input/...`, `/kaggle/working/...`). Resolve through configs, never hardcode.
- `data/raw/`, `data/interim/`, `data/processed/`, `mlruns/`, `reports/runs/` and all `*.parquet` are gitignored. `data/splits/train_val_metadata.json` and the manifests are trackable.

## Ownership and scope

- Person A: `notebooks/01..04`, `data/`, EDA, E2. Person B: `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`, E1, E3-E7.
- Implement only the phase/step requested; don't fill unrelated stubs or opportunistically rework a teammate's layer.
- New modules need at least one test; leakage and feature-schema regressions are the highest-risk failures.

## Validation before handing back

Run only cheap checks: `ruff`, `pytest` (28 tests, synthetic data), config loading, `--help`, `--sample`. State what passed, what could not be run on this branch, and the exact Kaggle command plus expected runtime/memory for anything heavy.
