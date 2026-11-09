# Home Credit Risk Scoring

Building a credit risk scoring pipeline on the Kaggle **Home Credit Default Risk** dataset: gradient-boosted trees with feature engineering, tuning, calibration, and explanation.

The task is binary classification on seven relational tables. For each applicant in `application_train`, predict `P(TARGET = 1 | x)`, then calibrate the probability and explain the drivers. Positives are roughly 8% of the data.

> **Project status: Phase 0 (scaffold).** The directory tree exists, but most modules are empty stubs and `pyproject.toml`, `.gitignore`, and `configs/*.yaml` are not populated yet. Commands below describe the intended interface; they will work as the corresponding phases land. See `INIT.md` for the authoritative plan.

## Scope

- **Models:** LightGBM, XGBoost, CatBoost, plus a weighted-average / stacked ensemble. A logistic-regression sanity baseline is optional. Deep learning is out of scope unless explicitly approved.
- **Primary metric:** ROC-AUC. Also reported: Gini (`2*AUC - 1`), KS, PR-AUC, Brier score, ECE, and recall/precision at fixed thresholds.
- **Research questions:**
  - **RQ1** — How much do features from the auxiliary tables add over the main application table alone?
  - **RQ2** — After tuning, how do the three GBDTs and their ensemble differ in AUC, KS, and calibration?
  - **RQ3** — Which imbalance-handling technique and tuning strategy works best per model?
  - **RQ4** — Do the models agree on the most important features (SHAP / gain; top-k Jaccard, Spearman)?

## Setup on a new machine

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| Git | Any recent version |
| [uv](https://docs.astral.sh/uv/) | Manages Python, the virtualenv, and dependencies. Do not use `pip` directly. |
| Python | Pinned in `.python-version` (currently `3.14.4`); `uv` installs it automatically. |
| Kaggle account | You must accept the competition rules before downloading data. |
| Disk / RAM | ~3 GB for raw CSVs, several GB for Parquet outputs; 16 GB RAM recommended. |

Install `uv` if it is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone git@github.com:ngTanPhuc/home-credit-risk.git
cd home-creadit-risk
uv sync
```

`uv sync` creates `.venv/` from `.python-version` and installs the locked dependencies from `pyproject.toml` / `uv.lock`. Commit `uv.lock`; never hand-edit it.

Verify the environment:

```bash
uv run python --version
uv run pytest -q
uv run ruff check .
```


### 3. Get the data

Accept the competition rules at <https://www.kaggle.com/c/home-credit-default-risk/rules>, then configure credentials:

```bash
mkdir -p ~/.kaggle
# Put your kaggle.json here (Kaggle -> Account -> Create New API Token)
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download and unpack into `data/raw/` (the only place raw data may live; never modify it):

```bash
uvx kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw
unzip home-credit-default-risk.zip
rm home-credit-default-risk.zip
cd ../..
```

Alternatively, download the zip manually from the competition's **Data** tab and unzip it into `data/raw/` (Recommend cách này cho đỡ rối).

Expected files (row counts are approximate — verify after download):

| File | Rows | Key(s) |
|---|---|---|
| `application_train.csv` | 307k | `SK_ID_CURR` |
| `application_test.csv` | 49k | `SK_ID_CURR` |
| `bureau.csv` | 1.7M | `SK_ID_CURR`, `SK_ID_BUREAU` |
| `bureau_balance.csv` | 27M | `SK_ID_BUREAU` |
| `previous_application.csv` | 1.7M | `SK_ID_CURR`, `SK_ID_PREV` |
| `POS_CASH_balance.csv` | 10M | `SK_ID_PREV` |
| `installments_payments.csv` | 13.6M | `SK_ID_PREV` |
| `credit_card_balance.csv` | 3.8M | `SK_ID_PREV` |

Data is never committed. Only `splits/folds.parquet` (a few MB) is tracked.

### 4. Optional: MLflow UI

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

`mlruns/` is local and gitignored. On Kaggle, export it as notebook output, or use the JSON fallback under `reports/runs/`.

## Intended pipeline

All scripts take a YAML config and expose `--sample N` (quick checks) and `--force` (recompute). Outputs are deterministic and resumable.

```bash
# 1. Ingest CSV -> downcast Parquet in data/interim/
uv run python -m src.data.load --config configs/base.yaml

# 2. Clean (DAYS_EMPLOYED sentinel, XNA, sign conventions)
uv run python -m src.data.clean --config configs/base.yaml

# 3. Create splits ONCE (20% hold-out, 5-fold CV) -> splits/folds.parquet
uv run python -m src.cv.splits --config configs/base.yaml

# 4. Build feature groups enabled in configs/features.yaml
uv run python -m src.features.build --config configs/features.yaml

# 5. Run an experiment (E1 baseline, E2 ablation, ...)
uv run python -m src.cv.runner --config configs/experiments/E1_baseline.yaml

# 6. Tune (resumable via Optuna SQLite storage)
uv run python -m src.tuning.optuna_search --config configs/experiments/E4_tuning.yaml

# 7. Final hold-out evaluation — run ONCE, deliberately
uv run python -m src.evaluation.holdout --config configs/experiments/E5_models.yaml --confirm-final

# 8. Explanation
uv run python -m src.explain.shap_analysis --config configs/experiments/E7_explain.yaml
uv run python -m src.explain.agreement --config configs/experiments/E7_explain.yaml
```

## Repository layout

```
configs/              YAML configs: base, features, per-model, per-experiment
data/raw/             Kaggle CSVs, untouched             (gitignored)
data/interim/         Cleaned/downcast Parquet per table (gitignored)
data/processed/       Feature tables + merged matrix     (gitignored)
splits/folds.parquet  SK_ID_CURR, is_holdout, fold       (committed)
src/data/             Ingestion and cleaning
src/features/         Feature groups, build, selection
src/cv/               Split creation and fold runner
src/models/           Common model interface, LGBM/XGB/CatBoost, ensemble
src/tuning/           Optuna and random search
src/calibration.py    Platt / isotonic on OOF predictions
src/evaluation/       Metrics, DeLong, locked hold-out script
src/explain/          SHAP analysis and model agreement
src/utils/            Memory downcasting and logging/MLflow helpers
scripts/              Thin CLI wrappers
notebooks/            EDA, results analysis, Kaggle launcher (no pipeline logic)
tests/                Unit, schema, and leakage tests
reports/tables/       B1-B4 CSV/Markdown/LaTeX tables
reports/figures/      ROC/PR, tuning history, reliability, SHAP plots
```

## Splits and evaluation protocol

- One fixed split, seed `42`: 80% development, 20% stratified hold-out.
- The development set uses 5-fold stratified CV; fold ids live in `splits/folds.parquet`.
- **Never re-split or regenerate `folds.parquet`** once it exists. Every model and experiment uses the same folds.
- The hold-out is read exactly once, by `src/evaluation/holdout.py --confirm-final`. No tuning or model selection may touch it.
- The split is random by design: the data has no application date, so a temporal split is impossible. This is a documented limitation.
- Final experiments repeat with 3 model seeds (the split stays fixed) and report mean ± std.

## Experiments

| ID | Experiment | Answers |
|---|---|---|
| E1 | LightGBM baseline (optional LR), main table only | Reference point |
| E2 | Cumulative feature-group ablation | RQ1 |
| E3 | Imbalance handling: none, class weight, undersampling, SMOTE (in-fold) | RQ3 |
| E4 | Tuning: default vs random search vs Optuna TPE | RQ3 |
| E5 | Model comparison on the full tuned feature set | RQ2 |
| E6 | Calibration: none vs Platt vs isotonic | RQ2 |
| E7 | SHAP explanation and cross-model agreement | RQ4 |

One MLflow experiment per ID (`E1_baseline`, ...). Run name format: `<model>_<variant>_seed<k>`. Log the resolved config, feature count, fold AUCs, mean/std, artifact paths, timing, peak memory, and git commit.

## Kaggle workflow

Kaggle sessions are time- and memory-limited (16 GB RAM ceiling):

1. Push code to GitHub, then clone/pull it in the notebook, or attach the repo as a Kaggle dataset.
2. Raw data is mounted read-only under `/kaggle/input/home-credit-default-risk/`.
3. Write processed Parquet and `mlruns/` to `/kaggle/working/`, then re-attach them as a Kaggle dataset so later sessions skip recomputation.
4. Keep `notebooks/kaggle_launcher.ipynb` thin: install/sync deps, set paths, call `python -m src.<module> --config ...`.
5. Heavy steps are resumable: skip existing outputs unless `--force`; use Optuna SQLite storage for tuning.

## Outputs

- `reports/tables/B1_ablation.csv`, `B2_model_comparison.csv`, `B3_calibration.csv`, `B4_delong_pairs.csv`
- `reports/figures/`: ROC/PR curves, Optuna history, reliability diagrams, SHAP summary/dependence/waterfall, importance-agreement heatmap
- `mlruns/`: local MLflow tracking (gitignored)

## Team

| Role | Owns |
|---|---|
| Person A — data & features | `src/data`, `src/features`, EDA, E2 |
| Person B — modeling & evaluation | `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`, E1, E3-E7 |

Shared contracts: the feature-table schema and `splits/folds.parquet`. Any column-name or key change must be recorded in `DECISIONS.md` and communicated.

## Documentation

- `INIT.md` — authoritative specification: phases, ownership, schema, hard rules.
- `DECISIONS.md` — append-only log of design decisions and why.
- `AGENTS.md` — instructions for AI coding agents working in this repo.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync` fails during Phase 0 | Dependencies are not declared yet; wait for Phase 0 step 2. |
| A package has no wheel for Python 3.14 | Lower `.python-version` (e.g. to `3.13`), then `uv lock` and `uv sync`. Check which interpreter the Kaggle image provides before matching it. |
| Kaggle download returns 403 | Accept the competition rules on the Kaggle website first. |
| Kaggle CLI complains about credentials | `chmod 600 ~/.kaggle/kaggle.json`; the file must be valid JSON. |
| Out-of-memory on `bureau_balance` / `installments_payments` | Use chunked reads, `usecols`, downcasting, and per-group aggregation; never load with default dtypes. |
| Hold-out script refuses to run | It requires `--confirm-final`. That is intentional; use CV for all development. |
