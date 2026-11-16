# INIT.md — Home Credit Credit Risk Scoring

Instructions for AI coding agents initializing and working on this repository. Read this file fully before writing any code. If anything here conflicts with a request from the human, ask before proceeding.

---

## 1. Project summary

**Title:** Building a Credit Risk Scoring Pipeline on Home Credit Data: Gradient-Boosted Trees with Feature Engineering, Tuning, Calibration and Explanation

**Task:** Binary classification on multi-table tabular data with class imbalance (~8% positives). Predict `P(TARGET = 1 | x)` for each applicant in `application_train`, then calibrate and explain it.

**Data:** Kaggle "Home Credit Default Risk" (7 tables, ~2.5 GB CSV). Key tables and keys:

| Table | Approx. rows | Keys |
|---|---|---|
| application_train / application_test | 307k / 49k | SK_ID_CURR |
| bureau | 1.7M | SK_ID_CURR, SK_ID_BUREAU |
| bureau_balance | 27M | SK_ID_BUREAU |
| previous_application | 1.7M | SK_ID_CURR, SK_ID_PREV |
| POS_CASH_balance | 10M | SK_ID_PREV |
| installments_payments | 13.6M | SK_ID_PREV |
| credit_card_balance | 3.8M | SK_ID_PREV |

Figures are approximate; verify after download.

**Models (decided):** XGBoost, CatBoost, LightGBM, plus an ensemble of them. A plain logistic regression is an optional sanity baseline. Deep Learning (MLP, TabNet, FT-Transformer) is **optional stretch only**; do not implement it unless the human asks.

**Primary metric:** ROC-AUC. Also report Gini (`2*AUC - 1`), KS, PR-AUC, Brier score, ECE, and recall/precision at chosen thresholds.

### Research questions (adapted to GBDT-only scope)

- **RQ1:** How much does feature engineering from the auxiliary tables improve performance over the main application table alone?
- **RQ2:** After tuning, how do XGBoost, CatBoost, LightGBM and their ensemble differ in AUC, KS and calibration?
- **RQ3:** Which imbalance-handling technique and tuning strategy works best per model?
- **RQ4:** Do the models agree on the most important features (SHAP / gain importance; top-k Jaccard, Spearman rank correlation)?

### Experiments

| ID | Experiment | Answers |
|---|---|---|
| E1 | Baseline: LightGBM (and optional LR) with default params, main table only | reference point |
| E2 | Feature-engineering ablation: add groups one by one | RQ1 |
| E3 | Imbalance handling: none, class weight, undersampling, SMOTE (inside folds only) | RQ3 |
| E4 | Tuning strategy: default vs Optuna TPE; AUC vs trial count | RQ3 |
| E5 | Model comparison on the full tuned feature set | RQ2 |
| E6 | Calibration: before vs after Platt / isotonic | RQ2 |
| E7 | Explanation and agreement across models | RQ4 |

---

## 2. Team, roles and agent permissions

The team has 2 humans. Roles are by pipeline layer so people rarely edit the same files.

- **Person A — data and features:** `notebooks/01..04` (ingestion, cleaning, feature engineering, dataset assembly), `data/`, EDA, E2.
- **Person B — modeling and evaluation:** `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`, E1, E3-E7.
- **Shared contract:** the dataset schema `data/processed/datasets/feature_contract.json` (Section 6) and the frozen split (Section 5).

### What the agent MAY do

- Write and edit source code, configs, tests, notebooks and documentation.
- Run quick, cheap checks locally: linting, unit tests on tiny synthetic data, `--help`, config validation, dry runs on a small sample.
- Propose commands for the human to run, with expected runtime and memory.

### What the agent MUST NOT do

- **Do not launch long jobs** (full feature building, tuning, full training, SHAP over the full set). Write the code and a ready-to-run command, then ask the human to launch it on Kaggle.
- **Do not run git commit, push, merge or branch operations.** The human commits to branches. The agent may suggest a branch name and a commit message.
- Do not download data, install system packages, or modify files under `data/raw/`.
- Do not touch the hold-out set except through the single final evaluation script (Section 7, rule 3).
- Do not regenerate the frozen split (`data/splits/` and the `fold` column) once it exists.
- Do not add dependencies without telling the human; use `uv add` and say why.

When unsure, stop and ask. Prefer a short question over a large wrong change.

---

## 3. Environment

- **Python environment:** `uv`. Pin Python in `.python-version`; dependencies in `pyproject.toml`; lock file `uv.lock` committed.
- **Core libraries:** pandas, pyarrow, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, SHAP, MLflow, imbalanced-learn, matplotlib, PyYAML, pytest, ruff.
- **Optional (stretch):** PyTorch, pytorch-tabnet, rtdl (only if DL is approved).
- **Compute:** Kaggle Notebooks (GPU T4 optional, ~16 GB RAM, limited session time and disk). Treat 16 GB RAM as a hard ceiling.
- **Tracking:** MLflow with a local file store at `./mlruns` (gitignored). On Kaggle, export `mlruns/` as notebook output and download it, or log to `reports/runs/` as JSON fallback.
- **Code language:** English everywhere (code, comments, docstrings, commit suggestions, docs).

### Kaggle workflow

1. Code lives in the GitHub repo; the Kaggle notebook clones or pulls the repo (or the repo is attached as a Kaggle dataset) and runs scripts via `python -m src.<module> --config configs/<file>.yaml`.
2. Raw data comes from the Kaggle competition input; processed Parquet files are saved to `/kaggle/working` and re-attached as a Kaggle dataset so later sessions do not recompute.
3. Every script must be **resumable**: write intermediate outputs to disk and skip steps whose outputs already exist (unless `--force`).
4. Keep a tiny launcher notebook `notebooks/kaggle_launcher.ipynb` that only installs deps, sets paths, and calls scripts. No pipeline logic in notebooks.

---

## 4. Repository structure

```
.
├── INIT.md                  # this file
├── README.md                # human-facing setup and reproduction guide
├── DECISIONS.md             # log of design decisions and why (append-only)
├── AGENTS.md                # agent instructions for this repo
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore               # data/, mlruns/, *.parquet outputs, .venv
├── configs/
│   ├── base.yaml            # paths, seeds, fold count, metrics settings
│   ├── features.yaml        # feature groups + cumulative ablation sets
│   ├── models/
│   │   ├── lightgbm.yaml    # default params + Optuna search space
│   │   ├── xgboost.yaml
│   │   └── catboost.yaml
│   └── experiments/         # one file per experiment E1..E7
├── data/                    # gitignored; manifests and split metadata tracked
│   ├── raw/                 # untouched Kaggle CSVs
│   ├── interim/             # per-table Parquet              (notebook 01)
│   ├── processed/           # cleaned tables, features/, datasets/ (notebooks 02..04)
│   └── splits/              # train.parquet, val.parquet, train_val_metadata.json
├── notebooks/               # Person A pipeline, standalone, Vietnamese prose
│   ├── 01_data_overview_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_dataset_assembly_validation.ipynb
│   └── kaggle_launcher.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            # load/merge YAML, resolve paths, set seeds
│   ├── cv/
│   │   ├── dataset.py       # contract + dataset loading, feature-set resolution
│   │   ├── splits.py        # split metadata loader/validator (never creates a split)
│   │   └── runner.py        # fold loop, OOF predictions, leakage-safe steps
│   ├── models/
│   │   ├── base.py          # common fit/predict_proba/save interface
│   │   ├── lgbm.py
│   │   ├── xgb.py
│   │   ├── catboost_model.py
│   │   └── ensemble.py      # weighted average and stacking
│   ├── imbalance.py         # class weight, undersampling, SMOTE (in-fold)
│   ├── tuning/
│   │   └── optuna_search.py # TPE + MedianPruner, mean 5-fold AUC objective
│   ├── calibration.py       # Platt / isotonic fit on OOF, apply to hold-out
│   ├── evaluation/
│   │   ├── metrics.py       # AUC, Gini, KS, PR-AUC, Brier, ECE, threshold metrics
│   │   ├── delong.py        # DeLong test for AUC differences
│   │   └── holdout.py       # the ONLY script allowed to read hold-out labels
│   ├── explain/
│   │   ├── shap_analysis.py # global + local SHAP, reason codes
│   │   └── agreement.py     # top-k Jaccard, Spearman across models
│   └── utils/
│       ├── memory.py        # downcast helpers, memory logging
│       └── logging.py       # logger + MLflow helpers
├── tests/                   # contract, split, metrics and leakage tests
├── scripts/                 # thin CLI entry points / shell runners
└── reports/
    ├── tables/              # B1..B4 as CSV (+ LaTeX/Markdown optional)
    ├── predictions/         # OOF Parquet per run
    ├── runs/                # JSON run records (MLflow fallback)
    └── figures/             # ROC/PR, Optuna history, reliability, SHAP, heatmap
```

Modelling code never rebuilds data: it reads the frozen dataset and validates it
against the contract. Person A's notebooks own everything under `data/`.

---

## 5. Data splitting protocol (fixed, shared)

- The split is created **once** by `notebooks/01_data_overview_eda.ipynb`: 20% of `application_train` becomes the **hold-out** (stratified on `TARGET`, seed `2026`), the remaining 80% is the development set with **5-fold stratified CV** membership (seed `42`).
- It is stored as `data/splits/train.parquet` (`SK_ID_CURR`, `fold`), `data/splits/val.parquet` (`SK_ID_CURR`) and `data/splits/train_val_metadata.json` (seeds, row counts, `source_sha256`).
- `notebooks/04_dataset_assembly_validation.ipynb` copies `fold` into `data/processed/datasets/train.parquet`. **That dataset is the modelling input**: every model and experiment reads the `fold` column from it and never re-splits.
- `src/cv/splits.py` only validates the metadata against the dataset row counts. It cannot create or regenerate a split.
- Never re-split, never reshuffle, never hand-edit files under `data/splits/`, `data/interim/` or `data/processed/`; the notebooks abort when a manifest hash does not match.
- Random split is intentional: the data has no application date, so a temporal split is impossible. Mention this limitation in the report.
- Repeat final experiments with 3 random seeds for model seeds (the split stays fixed); report mean and standard deviation.

---

## 6. Feature groups and schema contract

The feature groups are built by Person A in `notebooks/03_feature_engineering.ipynb` and assembled into one table per split by `notebooks/04_dataset_assembly_validation.ipynb`. Modelling code does not build features.

Each group:

- produces one row per `SK_ID_CURR`, keyed on the identifier,
- contains no `TARGET` column and uses no fold information,
- uses its own column prefix: `APP_` (engineered application ratios), `BUREAU_` (bureau + bureau_balance), `PREV_` (previous_application), `POS_`, `CC_`, `INS_` (repayment tables), `HAS_` (coverage flags). The 120 untouched application columns keep their Kaggle names.

Delivered groups (this is also the E2 ablation order):

| Group | Columns | Source |
|---|---|---|
| `application_original` | 120 | `application_train` raw columns |
| `application` | 10 | ratios derived from the application table |
| `bureau` | 17 | `bureau` + `bureau_balance` |
| `previous` | 10 | `previous_application` |
| `pos` | 6 | `POS_CASH_balance` |
| `credit_card` | 8 | `credit_card_balance` |
| `installments` | 9 | `installments_payments` |

180 predictors in total. The time-window variants (`WIN_`) and the in-fold feature-selection group were **not** delivered; E2 answers RQ1 with the groups above and does not invent the missing ones.

Where the contract lives:

- `data/processed/datasets/feature_contract.json` is the schema source of truth: version, predictor order, categorical columns, per-split column lists, excluded keys. `src/cv/dataset.py` validates every load against it.
- `reports/tables/feature_catalog.csv` documents every column: group, source, definition, dtype, role.
- `configs/features.yaml` maps group to columns and defines the cumulative ablation sets (`main` = 120, `main_plus_ratios` = 130, `plus_bureau` = 147, `plus_previous` = 157, `plus_pos` = 163, `plus_credit_card` = 171, `plus_installments` = 180). It is generated from the catalog in contract order; regenerate it when the dataset changes.

Any change to column names or the merge key is a schema change: update `DECISIONS.md` and tell the human, because both teammates depend on it.

---

## 7. Hard rules (non-negotiable)

1. **No leakage.** Any step that learns from data — target encoding, SMOTE or undersampling, feature selection, scaling, calibration — is fit on the training portion of a fold only, then applied to the validation portion. Wrap such steps so this is structurally enforced (fit/transform objects created per fold), not left to discipline.
2. **Aggregations that use no labels** (counts, sums, ratios from auxiliary tables) may be computed once globally. Anything touching `TARGET` may not.
3. **Hold-out is touched once, at the end.** Only `src/evaluation/holdout.py` reads hold-out labels, and it requires an explicit `--confirm-final` flag. Model selection and tuning use CV only.
4. **Calibrators** are fit on out-of-fold predictions from the development set, then applied to hold-out predictions.
5. **Reproducibility:** every script takes a config, sets all seeds, logs config and git commit hash (read-only) to MLflow, and writes outputs to deterministic paths.
6. **Memory discipline:** never load a large table with default dtypes. Use `usecols`, `dtype`, chunked reads, `float32`/`int32`/category types, aggregate per table, free intermediates, and save to Parquet. Log peak memory in heavy scripts.
7. **Same protocol for all models:** same folds, same features, same metrics, same tuning budget. Differences in results must come from the methods, not from the setup.

---


## 9. Coding standards

- Python 3.13+, type hints on public functions, docstrings stating inputs, outputs and any leakage considerations.
- `ruff` for lint and format; `pytest` for tests.
- No hard-coded paths or magic numbers: use configs.
- Keep functions small and pure where possible; side effects (disk I/O, MLflow) at the edges.
- Scripts expose `python -m src.<module> --config <yaml>` and support `--sample N` for quick checks and `--force` to recompute.
- Logging via the `logging` module, not `print`.
- Notebooks 01..04 are Person A's standalone data pipeline (Vietnamese prose, no `src/` imports). Analysis notebooks from 05 on import from `src/` and hold no pipeline logic.

## 10. Experiment tracking conventions

- One MLflow experiment per ID (`E1_baseline`, `E2_ablation`, ...).
- Run name: `<model>_<variant>_seed<k>`.
- Log: full resolved config, feature group list and count, fold AUCs, mean and std, OOF predictions path, training time, inference time, peak memory.
- Tag each run with `git_commit` (read from the repo, never created by the agent) and `data_version`.

## 11. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Out of memory on `bureau_balance` / `installments_payments` | Downcast dtypes, `usecols`, chunked aggregation by key, write Parquet per group |
| Leakage via target encoding, SMOTE, selection | Per-fold fit/transform objects; `tests/test_cv_leakage.py`; hold-out locked behind `--confirm-final` |
| Kaggle session limits (time, disk) | Resumable scripts, Optuna SQLite storage, persist outputs as Kaggle datasets |
| Overfitting to CV during heavy tuning | Compare CV vs hold-out gap; multiple seeds; limit trial counts |
| Tuning takes too long | MedianPruner, tune on a subsample first, cap trials |
| Teammates break each other's schema | Feature schema test; changes logged in `DECISIONS.md` |

## 12. How the agent should respond on each task

1. State briefly which phase and step the task belongs to.
2. List the files it will create or change.
3. Write the code and tests.
4. Give the exact command(s) the human should run, with expected runtime, memory use, and where outputs will appear.
5. Mention any decision that should be added to `DECISIONS.md`, and suggest a branch name and commit message (but do not run git).
6. Flag anything uncertain, and any assumption about the data that must be verified after download.

## 13. Out of scope

Production deployment, reject inference, in-depth fairness analysis (mention only in the discussion section), and sequence models (RNN/Transformer on transaction histories) unless explicitly approved.
