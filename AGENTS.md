# AGENTS.md

`INIT.md` is the authoritative spec (phases, ownership, schema, hard rules). Read it before changing anything. If a request conflicts with it, ask the human instead of guessing. `README.md` covers setup and the intended CLI.

## Current state

- Phase 0 scaffold: most files under `src/`, `tests/`, and `configs/` are **0-byte stubs**. Read a file before assuming it exists or works; do not infer behavior from its name.
- `pyproject.toml`, `.gitignore`, and all `configs/*.yaml` are also empty. Do not treat missing config as permission to hardcode values.
- `.python-version` pins `3.14.4`; `uv` owns `.venv/`. Changing the pin requires re-locking and affects the whole team.

## Commands

```bash
uv sync                                  # install/refresh the env
uv run pytest -q                         # full suite (empty until tests are written)
uv run pytest tests/test_metrics.py -q   # one file
uv run pytest tests/test_metrics.py -k ece -q
uv run ruff check . && uv run ruff format --check .
uv run python -m src.<module> --config configs/<file>.yaml [--sample N] [--force]
```

Use `uv run` (or `uv add`); never call `pip` or a bare `python` directly. Every script must accept `--config`, plus `--sample N` for smoke checks and `--force` to recompute. Scripts are resumable: skip existing outputs by default.

## Hard stops

- **No long jobs.** Never launch full feature builds, tuning, full training, or SHAP over the full dataset. Write the code and an exact command, then ask the human to run it on Kaggle. State expected runtime and memory.
- **No git writes.** Never commit, push, merge, branch, or change git config. Suggest a branch name and commit message instead.
- **No data acquisition.** Never download data, install system packages, or modify `data/raw/`. Data is absent from a fresh clone.
- **Hold-out is locked.** Only `src/evaluation/holdout.py --confirm-final` may read hold-out labels. All development uses CV on `splits/folds.parquet`.
- **Never regenerate `splits/folds.parquet`** once it exists, and never re-split or reshuffle.
- **No silent dependencies.** Use `uv add` and tell the human why.
- **No deep learning** (MLP, TabNet, FT-Transformer) unless explicitly approved.

## Invariants that are easy to break

- **Feature schema contract:** each group returns one row per `SK_ID_CURR`, uses its prefix (`MAIN_`, `BUR_`, `PREV_`, `REPAY_`, `WIN_`), contains no `TARGET`, uses no fold information, and is written to `data/processed/features_<group>.parquet`. Any column-name or key change is a schema change: update `DECISIONS.md` and tell the human.
- **Per-fold fit only:** target encoding, SMOTE/undersampling, scaling, feature selection, and calibration must be fit on the training portion of each fold and applied to validation. Enforce this structurally with fit/transform objects; never fit on the full development set.
- **Memory:** 16 GB is a hard ceiling. Use `usecols`, explicit dtypes, chunked reads/aggregation for the 10M+ row tables, `float32`/`int32`, and free intermediates. Save to Parquet between steps.
- **Reproducibility:** seed everything from config (split seed `42`), log the resolved config and read-only git commit to MLflow, write to deterministic paths. No magic numbers, paths, or file names.
- **Metrics:** ROC-AUC is primary; also Gini, KS, PR-AUC, Brier, ECE, and threshold precision/recall. All models use the same folds, features, metrics, and tuning budget.
- **Kaggle paths differ** (`/kaggle/input/...`, `/kaggle/working/...`). Resolve them through configs, never hardcode.
- **Style:** type hints on public functions, docstrings that state inputs/outputs/leakage notes, English everywhere, `logging` not `print`. Notebooks are analysis-only and import from `src/`.

## Ownership and scope

- Person A: `src/data`, `src/features`. Person B: `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`.
- Implement only the phase/step requested. Do not fill unrelated stubs or "improve" a teammate's layer opportunistically.
- New modules need at least one test; leakage and feature-schema regressions are the highest-risk failures.

## Validation before handing back

Run cheap checks only: `ruff`, targeted tests on tiny synthetic data, config validation, `--help`, or `--sample`. Report what passed, what could not be run, and the exact Kaggle command for anything heavy.
