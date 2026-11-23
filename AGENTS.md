# AGENTS.md

`AGENTS.md`(local) is the authoritative source for project workflow, phases, ownership, implementation responsibilities, coding constraints, and validation requirements.
`AGENTS.md`(global at ~/.config/opencode/AGENTS.md) contains the global rules for agents. **Read this file once at the start of every session**.

`INIT.md` describes the original project objectives, research questions, dataset specifications, methodology, and technical requirements. It provides project context but does not define the execution roadmap.

`README.md` is the human-facing project guide. It covers environment setup, installation, commands, experiment execution, and reproduction instructions. It does not define phases or override the instructions in `AGENTS.md`.

Read `AGENTS.md` before making changes. Consult `INIT.md` for research and technical context when needed. If instructions conflict, follow `AGENTS.md` for workflow and implementation decisions, and ask the human when a conflict affects the research methodology or project requirements.

## Data contract and split protocol

- Canonical split: `data/splits/train.parquet` (`SK_ID_CURR`, `fold`), `data/splits/val.parquet` (`SK_ID_CURR`) and `data/splits/train_val_metadata.json` (hold-out seed **2026**, CV seed 42, 20% hold-out). Notebook 01 creates it; notebook 04 copies `fold` into `data/processed/datasets/train.parquet`.
- `splits/folds.parquet` no longer exists. The notebook protocol is the only one. Never re-split, reshuffle or regenerate it; the notebooks raise when a manifest or `source_sha256` does not match.
- `src/cv/splits.py` only validates `train_val_metadata.json` against dataset row counts. It cannot create a split.
- `data/processed/datasets/feature_contract.json` is the schema source of truth: predictor order, categorical columns, per-split column lists, excluded keys. `src/cv/dataset.py` validates every load against it.
- `configs/features.yaml` maps group to columns and defines the cumulative ablation sets. It is generated from `reports/tables/feature_catalog.csv` in contract order; regenerate it when the dataset changes, don't hand-edit.
- `load_split_frame(..., split="val")` raises `PermissionError` unless `allow_holdout=True`. Only `src/evaluation/holdout.py` may pass it.
- Delivered groups: `application_original` 120, `application` 10, `bureau` 17, `previous` 10, `pos` 6, `credit_card` 8, `installments` 9 = 180. No time-window (`WIN_`) or in-fold selection group was delivered; do not invent them.
- E1 uses set `main` (the 120 raw application columns). The 10 ratios enter E2 at the second feature set of the ladder.

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

- **Person A:** Owns `notebooks/01..04`, `data/`, EDA, feature engineering, dataset assembly, and feature-group preparation for E2.
- **Person B:** Owns `src/cv`, `src/models`, `src/tuning`, `src/calibration`, `src/evaluation`, `src/explain`, E1, and E3-E7.
- **E2 (shared):** Person A prepares and validates feature groups and the feature catalog. Person B implements and executes the cumulative ablation experiments, evaluates the results, and produces `B1_ablation.csv`.

Implement only the phase or step requested. Do not fill unrelated stubs or opportunistically modify another person's layer.

## Validation before handing back

Run only cheap checks: `ruff`, `pytest` (45 tests, synthetic data), config loading, `--help`, `--sample`. State what passed, what could not be run on this branch, and the exact Kaggle command plus expected runtime/memory for anything heavy.

## Phases

Format: `Phase n - <name> - <PIC>`. Each step names who does it. **Agent** writes code, tests and docs, and runs only cheap checks. **Human** runs anything heavy (Kaggle), commits, and makes decisions. Implement only the phase/step requested (see "Ownership and scope"). Status tags show where the repo stands now.

### Phase 0 - Contract and repo cleanup - Person B (Person A supplies artifacts) [DONE]
1. **Agent:** verify Person A's artifacts (`data/processed/datasets/{train,val,test}.parquet`, `feature_contract.json`, `dataset_manifest.json`) and the frozen split in `data/splits/`. Record in `DECISIONS.md`: single split protocol (`data/splits` + `fold` column), hold-out seed 2026, CV seed 42, Optuna only (no random search).
2. **Agent:** delete `src/data/`, `src/features/`, `src/tuning/random_search.py`. Write `src/config.py`, `src/utils/{logging,memory}.py`, `src/cv/{dataset,splits}.py`, `configs/{base,features}.yaml`. Loading must check `feature_contract.json` and fail loudly if `data/` is missing. Write `tests/` for contract, fold disjointness, no hold-out leakage, and the per-fold fit/transform guard.
3. **Agent:** fix stale docs (`folds.parquet` references in README, INIT). Flag but do not delete the stale `reports/predictions/E1_baseline_lgbm_seed42_oof.parquet` and `data/raw/mlflow.db`.
4. **Human:** `uv sync`, `uv run pytest -q`, `uv run ruff check .`. Remove the flagged stale files. Commit to a branch and push.

### Phase 1 - Data ingestion, cleaning and EDA - Person A [DONE]
5. **Agent:** write `notebooks/01_data_overview_eda.ipynb`: CSV to downcast Parquet in `data/interim/` (chunked for the 10M+ row tables), then create the split in `data/splits/`.
6. **Agent:** write `notebooks/02_data_cleaning.ipynb`: `DAYS_EMPLOYED == 365243` to NaN plus flag, `XNA` handling, `DAYS_*` sign conventions, `cleaning_manifest.json`.
7. **Human:** run notebooks 01 and 02 on Kaggle. Verify row counts against INIT. Keep `data/splits/train_val_metadata.json` and never regenerate it.
8. **Agent:** review EDA outputs in `reports/tables/` (missingness, target rate, anomalies, join coverage).

### Phase 2 - Baseline (E1) - Person B [DONE]
9. **Agent:** write `src/cv/runner.py`, `src/models/{base,lgbm}.py`, `src/evaluation/metrics.py` (`compute_metrics` wraps the human's `all_metrics`; do not rewrite its body), and `configs/models/lightgbm.yaml`, `configs/experiments/E1_baseline.yaml`. E1 = LightGBM default params on set `main` (120 raw application columns). Cast `string` to `category` with categories fixed from fold-training. Add metrics tests on known arrays.
10. **Agent:** run only cheap checks: `ruff`, `pytest`, `--help`, `--sample 500` (delete `*_sample500` outputs afterwards).
11. **Human:** run E1 (expected about 5 min, under 1 GB). Confirm MLflow logging and CV AUC in roughly 0.74-0.76; flag large deviations.
12. **Agent:** check outputs: OOF Parquet in `reports/predictions/`, fold metrics and importance CSVs in `reports/tables/`.

### Phase 3 - Feature engineering and dataset assembly - Person A [DONE]
13. **Agent:** write `notebooks/03_feature_engineering.ipynb` (one row per `SK_ID_CURR`, correct prefixes, no `TARGET`, no fold info) and `notebooks/04_dataset_assembly_validation.ipynb` (copies `fold` into `train.parquet`, writes `feature_contract.json`).
14. **Human:** run notebooks 03 and 04 on Kaggle. Re-attach outputs as a Kaggle dataset and copy `data/processed/datasets/` and `data/splits/` to the local machine.
15. **Agent:** confirm the delivered contract (`datasets_v1`, 180 predictors, 16 categoricals) loads cleanly through `src/cv/dataset.py`.

### Phase 4 - Feature-group ablation (E2) - Person A (Person B runs it) [DONE]
16. **Agent:** regenerate `configs/features.yaml` from `reports/tables/feature_catalog.csv` in contract order. Never hand-edit it.
17. **Agent:** write the E2 configs for the seven cumulative sets (120, 130, 147, 157, 163, 171, 180). The 10 ratios enter at the second feature set. Same folds, same seeds.
18. **Agent:** write the B1 generator: `reports/tables/B1_ablation.csv` with set, n_features, fold AUCs, mean/std, delta vs previous set, and gain per group.
19. **Human:** run E2 (start with `--sample`, then full; runtime grows with feature count).
20. **Agent:** review B1. State in the notes that RQ1 is answered without time-window (`WIN_`) features, which were not delivered.

### Phase 5 - Imbalance handling (E3) - Person B [DONE]
21. **Agent:** implement `src/imbalance.py`: none, class weight / `scale_pos_weight`, random undersampling, SMOTENC on the 16 contract categoricals. Each variant is a fit/resample object created per fold and applied only to the training part. After SMOTENC, cast categoricals back to `category` using fold-training categories.
22. **Agent:** add tests that resampling never touches validation rows (`tests/test_cv_leakage.py`). Write the E3 configs.
23. **Human:** run E3 on a subsample first, then full (expected 30-60 min).
24. **Agent:** produce the E3 comparison table in `reports/tables/`; propose a `DECISIONS.md` entry on the chosen imbalance strategy.
- Result: AUC 0.77836 (`none`), 0.77920 (`class_weight`, +0.00084), 0.77753 (`undersample`, −0.00083), 0.69120 (`smotenc`, −0.08716) on `plus_installments`. `class_weight` is the chosen variant for E4-E7, and its ECE 0.29946 makes E6 calibration mandatory before any probability is reported. Artifacts: `src/imbalance.py` (four variants), `src/evaluation/imbalance_comparison.py`, `configs/experiments/E3_imbalance.yaml`, `reports/tables/E3_imbalance_comparison.csv`, `reports/E3_report.md`.

### Phase 6 - Model wrappers and tuning (E4) - Person B [DONE]
25. **Agent:** add `src/models/xgb.py` (out-of-fold target encoding, fit per fold) and `src/models/catboost.py` (native categoricals) behind the `BaseModel` interface, with tests. LightGBM keeps native categoricals.
26. **Agent:** implement `src/tuning/optuna_search.py`: TPE sampler, MedianPruner, objective = mean 5-fold AUC, early stopping on the validation fold. Search spaces go in `configs/models/*.yaml` (LightGBM/XGBoost: `num_leaves`/`max_depth`, `learning_rate`, `min_child_samples`/`min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`; CatBoost: `depth`, `learning_rate`, `l2_leaf_reg`, `bagging_temperature`, `border_count`). Write the E4 config and a smoke test on synthetic data.
27. **Human:** time a subsample run first, then run full on Kaggle. Expect hours for 3 models x 5 folds; the agent suggests about 40 trials per model.
28. **Agent:** produce default vs tuned AUC and the AUC vs trial-count curve from the Optuna history. Save best params to a deterministic path for E5.
- Result (run `E4_26-11-18_exp01`): default → tuned AUC 0.77920 → 0.78424 (LightGBM, +0.00504), 0.77528 → 0.78717 (XGBoost, +0.01189), 0.78259 → 0.78607 (CatBoost, +0.00348) on `plus_installments` with `class_weight`, seed 42, 40 trials per model (113 complete, 7 pruned, 0 failed). LightGBM's default run reproduces the E3 `class_weight` AUC exactly, and the hold-out stayed locked. Tuning cost 47.8 h total and 5.5 GB peak. The best-so-far curve is flat from trial ~4, and the 2000-round budget binds for the slow-learning winners, so tuned AUC is if anything slightly under-fitted. `fixed_iterations` (2000 / 1998 / 2000) and `final_params` are saved for E5; E5 must refit with them and no validation pair. Artifacts: `src/models/{xgb,catboost}.py`, `src/models/__init__.py`, `src/tuning/optuna_search.py`, `configs/experiments/E4_tuning.yaml`, `tests/test_{models,tuning}.py`, `reports/tables/E4_tuning_comparison.csv`, `reports/tables/E4_tuning_history.csv`, `reports/runs/E4_26-11-18_exp01/E4_best_params.json`, `reports/figures/E4_optuna_history.png`.

### Phase 7 - Model comparison and ensemble (E5) - Person B
29. **Agent:** implement `src/models/ensemble.py`: weighted average and stacking on OOF predictions (meta-learner evaluated without leaking its own training rows). Implement `src/evaluation/delong.py` with a unit test on known arrays.
30. **Agent:** write the E5 config: three GBDTs on the tuned full set, seeds `[42, 43, 44]`, same folds/features/metrics/tuning budget for every model.
31. **Human:** run E5.
32. **Agent:** produce `reports/tables/B2_model_comparison.csv` (mean ± std over seeds) and ROC/PR figures in `reports/figures/`.

### Phase 8 - Calibration (E6) - Person B
33. **Agent:** implement `src/calibration.py`: Platt and isotonic, fit on development OOF predictions only. Evaluate without in-sample fitting (fit on other folds' OOF, apply to the held-out fold).
34. **Agent:** write the E6 config, reliability diagrams, and Brier and ECE (15 bins) before vs after. Add tests.
35. **Human:** run E6. Then freeze the final model choice and calibrator, and record the decision in `DECISIONS.md`. This decision must be made before Phase 9.

### Phase 9 - Final hold-out evaluation - Person B
36. **Agent:** implement `src/evaluation/holdout.py`, the only module allowed to pass `allow_holdout=True`. It requires `--confirm-final` and emits `B3_calibration.csv`, `B4_delong_pairs.csv` and final metrics together. Test on synthetic data that it refuses to run without the flag. The agent never runs it on real data.
37. **Human:** review the frozen choice, then run `holdout.py --confirm-final` **once**. A second run is a protocol violation.
38. **Agent:** compare CV vs hold-out gap to check overfitting to CV; produce B3, B4 and the final figures.

### Phase 10 - Explanation (E7) - Person B
39. **Agent:** implement `src/explain/shap_analysis.py`: global (summary, dependence), local (waterfall for one approved and one rejected applicant), 3-5 reason codes per rejected applicant. Read hold-out features only, never labels.
40. **Agent:** implement `src/explain/agreement.py`: top-k Jaccard and Spearman across LightGBM, XGBoost, CatBoost.
41. **Human:** run on a 10k-row sample (full SHAP is a long job).

### Phase 11 - Writing, consolidation and delivery - Person A and Person B
42. **Agent:** consolidate B1-B4 and figures; write `notebooks/05_results_analysis.ipynb` (imports from `src/`, no pipeline logic); finish the README and a report skeleton (introduction, related work, method, experiments, conclusion). Note the random-split limitation and the missing `WIN_` features.
43. **Human:** review the report, final `DECISIONS.md` pass, commit to a branch.
44. **Optional (only if the human agrees):** Kaggle submission from `test.parquet`; Streamlit demo returning probability and reason codes; one DL baseline (MLP or TabNet) under the same protocol.
