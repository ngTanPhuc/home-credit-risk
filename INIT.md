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
| E4 | Tuning strategy: default vs random search vs Optuna TPE; AUC vs trial count | RQ3 |
| E5 | Model comparison on the full tuned feature set | RQ2 |
| E6 | Calibration: before vs after Platt / isotonic | RQ2 |
| E7 | Explanation and agreement across models | RQ4 |

---

## 2. Team, roles and agent permissions

The team has 2 humans. Roles are by pipeline layer so people rarely edit the same files.

- **Person A — data and features:** `src/data`, `src/features`, EDA, E2.
- **Person B — modeling and evaluation:** `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`, E1, E3-E7.
- **Shared contract:** the feature table schema (Section 6) and the fold file (Section 5).

### What the agent MAY do

- Write and edit source code, configs, tests, notebooks and documentation.
- Run quick, cheap checks locally: linting, unit tests on tiny synthetic data, `--help`, config validation, dry runs on a small sample.
- Propose commands for the human to run, with expected runtime and memory.

### What the agent MUST NOT do

- **Do not launch long jobs** (full feature building, tuning, full training, SHAP over the full set). Write the code and a ready-to-run command, then ask the human to launch it on Kaggle.
- **Do not run git commit, push, merge or branch operations.** The human commits to branches. The agent may suggest a branch name and a commit message.
- Do not download data, install system packages, or modify files under `data/raw/`.
- Do not touch the hold-out set except through the single final evaluation script (Section 7, rule 3).
- Do not regenerate `splits/folds.parquet` once it exists.
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
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore               # data/, mlruns/, *.parquet outputs, .venv
├── configs/
│   ├── base.yaml            # paths, seeds, fold count, hold-out fraction
│   ├── features.yaml        # which feature groups are enabled
│   ├── models/
│   │   ├── lightgbm.yaml    # default params + Optuna search space
│   │   ├── xgboost.yaml
│   │   └── catboost.yaml
│   └── experiments/         # one file per experiment E1..E7
├── data/                    # gitignored
│   ├── raw/                 # untouched Kaggle CSVs
│   ├── interim/             # cleaned/downcast Parquet per table
│   └── processed/           # per-group feature tables + merged matrix
├── splits/
│   └── folds.parquet        # SK_ID_CURR, is_holdout, fold  (committed, small)
├── src/
│   ├── __init__.py
│   ├── config.py            # load/merge YAML, resolve paths, set seeds
│   ├── data/
│   │   ├── load.py          # chunked CSV -> Parquet, dtype downcasting
│   │   └── clean.py         # DAYS_EMPLOYED=365243, XNA, sign conventions
│   ├── features/
│   │   ├── base.py          # FeatureGroup interface + registry
│   │   ├── main_ratios.py   # group 1: ratios from application table
│   │   ├── bureau.py        # group 2: bureau + bureau_balance
│   │   ├── previous.py      # group 3: previous_application
│   │   ├── repayment.py     # group 4: installments, POS, credit card
│   │   ├── time_windows.py  # group 5: last 6 / 12 months variants
│   │   ├── build.py         # builds enabled groups, merges, validates schema
│   │   └── select.py        # group 6: feature selection (inside folds)
│   ├── cv/
│   │   ├── splits.py        # create_splits() (run once), load_splits()
│   │   └── runner.py        # fold loop, OOF predictions, leakage-safe steps
│   ├── models/
│   │   ├── base.py          # common fit/predict_proba/save interface
│   │   ├── lgbm.py
│   │   ├── xgb.py
│   │   ├── catboost_model.py
│   │   └── ensemble.py      # weighted average and stacking
│   ├── imbalance.py         # class weight, undersampling, SMOTE (in-fold)
│   ├── tuning/
│   │   ├── optuna_search.py # TPE + MedianPruner, mean 5-fold AUC objective
│   │   └── random_search.py # comparison baseline for E4
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
├── scripts/                 # thin CLI entry points / shell runners
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results_analysis.ipynb
│   └── kaggle_launcher.ipynb
├── tests/
│   ├── test_no_leakage.py
│   ├── test_splits.py
│   ├── test_features_schema.py
│   └── test_metrics.py
└── reports/
    ├── tables/              # B1..B4 as CSV (+ LaTeX/Markdown optional)
    └── figures/             # ROC/PR, Optuna history, reliability, SHAP, heatmap
```

Create empty `__init__.py` files and stub modules with docstrings and `NotImplementedError` where appropriate during initialization. Do not fill in full implementations during scaffolding.

---

## 5. Data splitting protocol (fixed, shared)

- Create the split **once** with a fixed seed (`seed: 42` in `configs/base.yaml`).
- 20% of `application_train` is the **hold-out**, stratified on `TARGET`. 80% is the development set.
- The development set gets **5-fold stratified CV**; fold ids are stored in `splits/folds.parquet` with columns `SK_ID_CURR`, `is_holdout`, `fold` (`-1` for hold-out rows).
- All models and all experiments read this file. Never re-split, never reshuffle.
- Random split is intentional: the data has no application date, so a temporal split is impossible. Mention this limitation in the report.
- Repeat final experiments with 3 random seeds for model seeds (the split stays fixed); report mean and standard deviation.

---

## 6. Feature groups and schema contract

Each group is a class or function registered in `src/features/base.py` that:

- takes cleaned Parquet tables as input,
- returns a DataFrame indexed or keyed by `SK_ID_CURR`, one row per applicant,
- uses a unique column prefix per group (`MAIN_`, `BUR_`, `PREV_`, `REPAY_`, `WIN_`),
- contains no `TARGET` column and uses no fold information,
- is saved to `data/processed/features_<group>.parquet`.

Groups (this is also the E2 ablation order):

1. **Main ratios:** credit/income, annuity/income, credit/annuity (approximate term), goods price/credit, days employed/age; mean, product and std of `EXT_SOURCE_1..3`.
2. **Bureau and bureau_balance:** loan counts, total debt, overdue ratios; active vs closed split; month-status aggregates.
3. **Previous applications:** approved/refused counts, approved-to-requested amount ratio.
4. **Repayment behavior** (installments, POS, credit card): days past due, early payment days, paid/due ratio, credit utilization.
5. **Time windows:** recompute groups 2-4 for the last 6 and 12 months.
6. **Feature selection:** drop mostly-missing or constant columns and highly correlated pairs; null importance or SHAP-based pruning (**fit inside folds only**).

Enabling/disabling groups is controlled only through `configs/features.yaml`. Any change to column names or the merge key is a schema change: update `DECISIONS.md` and tell the human, because both teammates depend on it.

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

## 8. Workflow, step by step

Each step lists the owner, the work the agent does, and what the human launches. The agent writes code; the human runs it.

### Phase 0 — Scaffold
1. **Agent:** create the repository structure in Section 4, `pyproject.toml` (via `uv`), `.gitignore`, `README.md` skeleton, `DECISIONS.md` with the decisions already made (models, tools, scope), stub modules, and `configs/base.yaml`.
2. **Agent:** create `src/config.py`, `src/utils/memory.py`, and `src/utils/logging.py` fully.
3. **Human:** `uv sync`, run `pytest`, commit to a branch, push.

### Phase 1 — Data ingestion and EDA — Person A
4. **Agent:** write `src/data/load.py` to convert each CSV to downcast Parquet in `data/interim/`, chunked for the large tables (`bureau_balance`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`).
5. **Agent:** write `src/data/clean.py`: replace `DAYS_EMPLOYED == 365243` with NaN plus a binary flag, handle `XNA` in `CODE_GENDER` and other columns, keep `DAYS_*` sign conventions documented.
6. **Human:** run ingestion and cleaning on Kaggle; verify row counts against Section 1.
7. **Agent:** produce `notebooks/01_eda.ipynb`: missingness, target rate, anomalies, key cardinalities, join coverage.

### Phase 2 — Splits and baseline — Person B
8. **Agent:** write `src/cv/splits.py` and `tests/test_splits.py` (stratification preserved, no overlap, hold-out size).
9. **Human:** run split creation once; commit `splits/folds.parquet`.
10. **Agent:** write `src/cv/runner.py`, `src/models/base.py`, `src/models/lgbm.py`, `src/evaluation/metrics.py`, and an E1 config. Baseline = LightGBM default params on main-table columns only.
11. **Human:** run E1; confirm MLflow logging and CV AUC (expect roughly the 0.74-0.76 range on main table only; flag large deviations).

### Phase 3 — Feature engineering and ablation E2 — Person A, with B running the ablation
12. **Agent:** implement feature groups one at a time (Section 6), each with a schema test and a small synthetic-data unit test. Aggregate large tables in pieces and write Parquet per group.
13. **Human:** run `features/build.py` per group on Kaggle; re-attach outputs as a Kaggle dataset.
14. **Agent:** write the E2 ablation runner that adds groups cumulatively and logs AUC (CV and hold-out-free), number of features, and gain per group to `reports/tables/B1_ablation.csv`.
15. **Human:** run E2.

### Phase 4 — Imbalance and tuning, E3 and E4  — Person B
16. **Agent:** implement `src/imbalance.py` (none, class weight / `scale_pos_weight`, random undersampling, SMOTE) applied strictly within training folds. Add a leakage test.
17. **Agent:** implement `src/tuning/optuna_search.py` (TPE sampler, MedianPruner, objective = mean 5-fold AUC, 50-100 trials, early stopping on the validation fold) and `random_search.py` for comparison. Search spaces in `configs/models/*.yaml`:
    - LightGBM/XGBoost: `num_leaves`/`max_depth`, `learning_rate`, `min_child_samples`/`min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
    - CatBoost: `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `border_count`.
18. **Agent:** add XGBoost and CatBoost wrappers behind the same interface. Use native categorical handling for LightGBM and CatBoost; out-of-fold target encoding for XGBoost.
19. **Human:** run E3 and E4 (start with a subsample to check runtime, then full). Keep Kaggle session limits in mind: the agent should make tuning resumable via Optuna's SQLite storage.

### Phase 5 — Comparison, ensemble, calibration, statistics — Person B
20. **Agent:** implement the ensemble (weighted average and stacking with OOF predictions), calibration (Platt, isotonic, fit on OOF), DeLong test, and the B2/B3/B4 table generators plus ROC/PR and reliability plots.
21. **Human:** run E5 and E6 with 3 seeds; then run `holdout.py --confirm-final` **once**.
22. **Agent:** produce B2 (model comparison), B4 (DeLong pairs), and figures. Compare CV vs hold-out gap to check for overfitting to CV.

### Phase 6 — Explanation, E7
23. **Agent:** implement SHAP global (summary, dependence), local (waterfall for one rejected and one approved applicant), 3-5 reason codes per rejected applicant, and `agreement.py` (top-k Jaccard, Spearman across LightGBM, XGBoost, CatBoost).
24. **Human:** run on a sampled subset of rows if SHAP is slow.

### Phase 7 — Writing and delivery
25. **Agent:** help with `notebooks/02_results_analysis.ipynb`, final README with reproduction steps, and report skeleton (introduction, related work, method, experiments, conclusion).
26. **Optional milestones** (only if time and the human agrees): Kaggle submission on `application_test`; a Streamlit demo returning probability and reason codes; one DL baseline (MLP or TabNet) under the same protocol.

---

## 9. Coding standards

- Python 3.13+, type hints on public functions, docstrings stating inputs, outputs and any leakage considerations.
- `ruff` for lint and format; `pytest` for tests. New modules need at least one test.
- No hard-coded paths or magic numbers: use configs.
- Keep functions small and pure where possible; side effects (disk I/O, MLflow) at the edges.
- Scripts expose `python -m src.<module> --config <yaml>` and support `--sample N` for quick checks and `--force` to recompute.
- Logging via the `logging` module, not `print`.
- Notebooks contain analysis and plots only; they import from `src/`.

## 10. Experiment tracking conventions

- One MLflow experiment per ID (`E1_baseline`, `E2_ablation`, ...).
- Run name: `<model>_<variant>_seed<k>`.
- Log: full resolved config, feature group list and count, fold AUCs, mean and std, OOF predictions path, training time, inference time, peak memory.
- Tag each run with `git_commit` (read from the repo, never created by the agent) and `data_version`.

## 11. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Out of memory on `bureau_balance` / `installments_payments` | Downcast dtypes, `usecols`, chunked aggregation by key, write Parquet per group |
| Leakage via target encoding, SMOTE, selection | Per-fold fit/transform objects; `test_no_leakage.py`; hold-out locked behind `--confirm-final` |
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
