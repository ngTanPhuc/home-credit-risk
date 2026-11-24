# E4 — Tuning (default vs Optuna TPE)

Báo cáo kết quả E4 (RQ3, phần tuning): tuning có cải thiện năng lực xếp hạng của ba GBDT so với tham số mặc định hay không, và cái giá phải trả là bao nhiêu.

## 1. Mục tiêu

Trả lời phần tuning của **RQ3**: trên `plus_installments` (180 predictor), cùng 5 fold, cùng seed và cùng variant `class_weight` của E3, ba model (LightGBM, XGBoost, CatBoost) có tăng ROC-AUC khi tune bằng Optuna TPE so với chạy mặc định hay không; nếu có thì tăng bao nhiêu, ở model nào, và tune có cần đủ 40 trial hay không.

Cách đo: mỗi model chạy hai lần trên cùng fold. Lần 1 là **default** (`params: {}` trong `configs/models/*.yaml`, đúng cấu hình E1–E3). Lần 2 là **tuned**: Optuna TPE tối đa hoá AUC trung bình 5 fold, có MedianPruner, early stopping 100 vòng trên validation của fold đang tune. Không calibrate, không chạm hold-out.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E4_tuning.yaml` |
| Model | `configs/models/{lightgbm,xgboost,catboost}.yaml` |
| Feature set | `plus_installments` = 180 predictor, 16 cột categorical |
| Imbalance | `class_weight` (quyết định E3) |
| Dataset | `data/processed/datasets/train.parquet`, contract `datasets_v1` |
| Split | 5 fold, seed 42; hold-out 61.503 dòng **không đọc** |
| Model seed | 42 (sampler seed 42) |
| Trial budget | 40 trial/model, `timeout_seconds: null` |
| Sampler | TPE, `seed = 42` |
| Pruner | Median, `n_startup_trials: 10`, `n_warmup_steps: 1` |
| Early stopping | 100 vòng, chỉ trên validation của fold đang tune |
| Metric mục tiêu | mean ROC-AUC 5 fold |
| Metric báo cáo | ROC-AUC (chính), Gini, KS, PR-AUC, Brier, ECE 15 bins |

Lệnh chạy:

```bash
uv run python3 -m src.tuning.optuna_search --config configs/experiments/E4_tuning.yaml
```

Study lưu ở `reports/runs/E4_26-11-18_exp01/E4_optuna.db` nên chạy lại chỉ chạy phần trial còn thiếu. `--models a,b` và `--trials N` giới hạn phạm vi; `--force` xoá study nháp.

Không gian tìm kiếm (7 chiều mỗi model, `configs/models/*.yaml`):

| Model | Tham số |
|---|---|
| LightGBM | `num_leaves` 15–255 (log), `learning_rate` 0,01–0,3 (log), `min_child_samples` 5–100, `subsample` 0,6–1,0, `colsample_bytree` 0,6–1,0, `reg_alpha` 1e-8–10 (log), `reg_lambda` 1e-8–10 (log) |
| XGBoost | `max_depth` 3–10, `learning_rate` 0,01–0,3 (log), `min_child_weight` 1–50 (log), `subsample` 0,6–1,0, `colsample_bytree` 0,6–1,0, `reg_alpha` 1e-8–10 (log), `reg_lambda` 1e-8–10 (log) |
| CatBoost | `depth` 4–10, `learning_rate` 0,01–0,3 (log), `l2_leaf_reg` 1e-3–30 (log), `bagging_temperature` 0–1, `border_count` 64–255 |

## 3. Dữ liệu vào

- Development: 246.008 dòng, 180 predictor, tỷ lệ positive 0,08073, giống E1–E3.
- Hold-out: 61.503 dòng, bị khoá; cả 3 run default ghi `n_holdout_rows = 61.503` và không load `val.parquet`.
- Missing: không điền. LightGBM/CatBoost nhận NaN native; XGBoost target-encode categorical trong `fit` bằng nested 5-fold trên dòng training của fold, unseen/missing rơi về prior của fold.
- Số dòng fit mỗi fold: 196.806 / 196.807, validate 49.202 / 49.201.

## 4. Kết quả: default vs tuned

| Model | Default AUC | Tuned AUC | ΔAUC | Std tuned (5 fold) | Trial | Complete | Pruned | Failed | Vòng lặp cố định |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0,77920 ± 0,00120 | **0,78424** | **+0,00504** | 0,00126 | 40 | 38 | 2 | 0 | 2.000 |
| XGBoost | 0,77528 ± 0,00161 | **0,78717** | **+0,01189** | 0,00136 | 40 | 38 | 2 | 0 | 1.998 |
| CatBoost | 0,78259 ± 0,00205 | **0,78607** | **+0,00348** | 0,00141 | 40 | 37 | 3 | 0 | 2.000 |

Ba điểm cần đọc đúng:

- **LightGBM default tái lập E3 `class_weight` chính xác** (0,77920, cùng 5 giá trị AUC theo fold). Vậy harness E4 chạy đúng protocol E1–E3, và mọi chênh lệch còn lại đến từ tuning chứ không từ dữ liệu hay fold.
- **XGBoost được lợi nhiều nhất** (+0,01189), CatBoost ít nhất (+0,00348). Nhưng đây là **một seed**; thứ hạng giữa ba model (XGBoost > CatBoost > LightGBM, spread 0,0029) nằm trong khoảng hơn 2 std fold, nên E4 **không** kết luận model nào thắng. Việc đó thuộc E5 (3 seed + DeLong).
- **ΔAUC của tuned là hơi lạc quan** vì tuned dùng early stopping trên chính validation fold đang chấm. Default thì không. Mức lạc quan này nhỏ vì early stopping gần như không kích hoạt (mục 7), nhưng vẫn phải nêu.

Metric đầy đủ chỉ có ở nhánh **default** (nhánh tuned chỉ ghi AUC từng fold, không retrain nên không có OOF/Brier/ECE):

| Model (default) | AUC | Gini | KS | PR-AUC | Brier | ECE | Fit (s/fold) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0,77920 | 0,55840 | 0,42003 | 0,26894 | 0,17736 | 0,29946 | 8,16 | 3.042 |
| XGBoost | 0,77528 | 0,55056 | 0,41556 | 0,26312 | 0,15385 | 0,25484 | 19,26 | 3.173 |
| CatBoost | 0,78259 | 0,56518 | 0,42677 | 0,27287 | 0,17822 | 0,30222 | 72,45 | 3.738 |

`class_weight` vẫn làm hỏng calibration như E3 (Brier 0,154–0,178, ECE 0,255–0,302 so với 0,0681 / 0,0046 của E1 `none`). Tuning **không** sửa được điều này, nên E6 vẫn bắt buộc.

## 5. AUC theo fold

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM default | 0,78128 | 0,77782 | 0,77878 | 0,77842 | 0,77970 | 0,77920 | 0,00120 |
| LightGBM tuned | 0,78624 | 0,78234 | 0,78410 | 0,78387 | 0,78466 | 0,78424 | 0,00126 |
| XGBoost default | 0,77799 | 0,77482 | 0,77308 | 0,77474 | 0,77575 | 0,77528 | 0,00161 |
| XGBoost tuned | 0,78931 | 0,78528 | 0,78629 | 0,78731 | 0,78765 | 0,78717 | 0,00136 |
| CatBoost default | 0,78583 | 0,77939 | 0,78237 | 0,78303 | 0,78233 | 0,78259 | 0,00205 |
| CatBoost tuned | 0,78761 | 0,78400 | 0,78515 | 0,78762 | 0,78597 | 0,78607 | 0,00141 |

Chênh lệch từng fold (tuned − default):

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Số fold dương |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | +0,00496 | +0,00452 | +0,00532 | +0,00544 | +0,00496 | 5/5 |
| XGBoost | +0,01132 | +0,01046 | +0,01320 | +0,01257 | +0,01190 | 5/5 |
| CatBoost | +0,00179 | +0,00461 | +0,00278 | +0,00459 | +0,00364 | 5/5 |

Cả 15/15 fold đều dương, nên gain của tuning **không** phải nhiễu fold đơn lẻ như trường hợp `class_weight` ở E3. Nó vẫn là một seed, nhưng nhất quán về dấu.

## 6. Đường cong AUC theo số trial

`reports/figures/E4_optuna_history.png` vẽ best-so-far theo trial. Gain đến rất sớm:

| Model | Final best | Đạt trong 0,001 của final từ trial | Đạt đúng final tại trial | Top 3 trial |
|---|---:|---:|---:|---|
| LightGBM | 0,78424 | 4 | 31 | 31 (0,78424), 35 (0,78411), 38 (0,78407) |
| CatBoost | 0,78607 | 12 | 14 | 14 (0,78607), 32 (0,78598), 37 (0,78594) |
| XGBoost | 0,78717 | 26 | 33 | 33 (0,78717), 34 (0,78709), 27 (0,78673) |

Ngân sách 20 trial thay vì 40 sẽ cho: CatBoost đúng bằng final, LightGBM −0,00048, XGBoost −0,00113. **Kết luận cho RQ3:** 40 trial là dư; đường cong bão hoà sớm và 20 trial đủ để báo cáo, nếu chạy lại thì nên giảm ngân sách chứ không tăng.

## 7. Số vòng lặp: cap 2000 có phải under-fit?

Đây là chỗ bản ghi E4 ban đầu **sai** và đã được sửa.

Sự kiện đo được: cả ba winner học chậm (`learning_rate` 0,0105 / 0,0194 / 0,0291) và chạm trần 2000 vòng.

| Model | best iteration từng fold | Vòng lặp giữ lại |
|---|---|---|
| LightGBM | 2000, 2000, 2000, 2000, 2000 | 2.000 |
| XGBoost | 1998, 1999, 1997, 1995, 1994 | 1.998 |
| CatBoost | 1999, 1999, 1999, 1999, 1999 | 2.000 |

Early stopping gần như không kích hoạt, nên (a) tuned AUC gần như không có lạc quan do early stopping, đúng. Nhưng suy luận tiếp theo — "trần bị chặn nên model còn under-fit, nâng trần sẽ tốt hơn" — là **suy đoán, không phải đo**, và E5 đã đo:

| Vòng lặp (LightGBM tuned, fold 0) | AUC |
|---|---:|
| 2.000 | **0,78624** |
| 4.000 | 0,78470 |
| 8.000 | 0,78144 |

Trần 2000 **nằm đúng đỉnh**. Nhiều cây hơn thì overfit. Vậy:

- "Trần bị chặn" (đúng, cơ học) **không** suy ra "còn dư địa" (sai).
- E4 **không** under-fit. Nâng trần làm AUC giảm ~0,0016 ở 4.000 vòng.
- XGBoost và CatBoost chưa được probe riêng; kết luận trên là bằng chứng cho LightGBM.

Lỗi thật của E4 nằm ở chỗ khác, và nhỏ hơn: **chiều `subsample` của LightGBM tune vô ích**. `subsample_freq` mặc định 0 nên bagging không chạy, giá trị 0,70 không bao giờ tới booster. Với `subsample_freq: 1`, fold 0 được 0,78756 (+0,0013). Tức E4 hơi **under-tuned** (một trong bảy chiều chết), không phải under-fitted. Sửa nó thuộc về một pass model/feature sau, không phải chạy lại E4.

## 8. Siêu tham số tốt nhất

`reports/runs/E4_26-11-18_exp01/E4_best_params.json` lưu `best_params` (7 chiều), `best_iterations` từng fold, `fixed_iterations` và `final_params` (sẵn dùng cho E5: `best_params` + số vòng cố định, không early stopping, không validation pair).

| Model | Tham số |
|---|---|
| LightGBM | `num_leaves` 30, `learning_rate` 0,01055, `min_child_samples` 55, `subsample` 0,7003, `colsample_bytree` 0,6316, `reg_alpha` 1,3098, `reg_lambda` 1,8276, `n_estimators` 2000 |
| XGBoost | `max_depth` 4, `learning_rate` 0,01936, `min_child_weight` 6,4167, `subsample` 0,7313, `colsample_bytree` 0,9835, `reg_alpha` 7,2931, `reg_lambda` 1,3e-4, `n_estimators` 1998 |
| CatBoost | `depth` 5, `learning_rate` 0,02908, `l2_leaf_reg` 25,229, `bagging_temperature` 0,1234, `border_count` 168, `iterations` 2000 |

**Chuẩn hoá số vòng lặp theo từng library:** LightGBM `best_iteration_` đã đếm số cây giữ lại (1-based), còn XGBoost `best_iteration` và CatBoost `get_best_iteration()` là 0-based. `_ITERATION_OFFSET` trong `src/tuning/optuna_search.py` mã hoá điều này; model lạ sẽ raise thay vì đoán. Bản ghi đầu tiên của LightGBM là 2001 (cộng +1 hai lần) đã được tính lại từ `E4_optuna.db` và sửa thành 2000 trong `E4_best_params.json` và `E4_tuning_comparison.csv`.

## 9. Kiểm tra tính hợp lệ

- LightGBM default tái lập E3 `class_weight`: AUC mean 0,77920, cùng 5 giá trị fold.
- 120/120 trial chạy xong: 113 complete, 7 pruned, 0 failed. Pruned: LightGBM trial 17 (fold-0 AUC 0,7601) và 20 (0,7726); XGBoost 15 (0,7766) và 35 (0,7854); CatBoost 20 (0,7748), 22 (0,7807), 28 (0,7798).
- `best_so_far` đơn điệu không giảm ở cả ba model; trial id liên tục 0–39.
- Hold-out không bị đọc: `n_holdout_rows = 61.503` trong cả 3 run default và 0 dòng hold-out trong bất kỳ output nào.
- Không dùng fold info làm feature; categorical encode (XGBoost) fit trong từng fold.
- `git_commit = 4771c091` cho cả 3 run default; `data_version = datasets_v1`.

## 10. Chi phí

| Model | Tổng tuning (s) | Tổng (h) | Trung bình/trial (s) | Trial dài nhất (s) | Số trial |
|---|---:|---:|---:|---:|---:|
| LightGBM | 17.134,8 | 4,76 | 428,4 | 938,8 | 40 |
| XGBoost | 62.951,0 | 17,49 | 1.573,8 | 2.300,9 | 40 |
| CatBoost | 92.165,4 | 25,60 | 2.304,1 | 4.558,8 | 40 |
| **Tổng** | **172.251,2** | **47,85** | — | — | **120** |

Peak RSS của cả tiến trình tuning là **5,5 GB** (ghi ở `peak_memory_mb`); ba study chạy tuần tự trong một tiến trình nên đây là đỉnh chung, không tách được theo model. Vẫn dưới trần 16 GB. CatBoost chiếm 54% thời gian cho gain nhỏ nhất (+0,00348); nếu chạy lại, CatBoost là ứng viên cắt ngân sách trial đầu tiên.

## 11. Hạn chế và lỗi đã biết

- **Một seed.** Thứ hạng ba model ở E4 không dùng để chọn model; E5 mới quyết định.
- **Tuned AUC hơi lạc quan** so với default vì early stopping nhìn validation fold (mức nhỏ, xem mục 7).
- **Nhánh tuned không có OOF/Brier/ECE** — chỉ AUC từng fold. Muốn metric đầy đủ cho cấu hình tuned phải refit, và đó chính là việc E5 làm bằng `final_params`.
- **`subsample` của LightGBM là chiều chết** do `subsample_freq: 0` (mục 7). Không sửa trong E4.
- **Cap 2000 chỉ được probe trên LightGBM**; XGBoost/CatBoost chưa có capacity probe riêng.
- Split random, không theo thời gian; không có feature cửa sổ thời gian (`WIN_`) — hạn chế chung của E1–E4.

## 12. Artifacts

| Loại | Đường dẫn |
|---|---|
| Bảng so sánh | `reports/tables/E4_tuning_comparison.csv` |
| Lịch sử trial | `reports/tables/E4_tuning_history.csv` |
| Best params cho E5 | `reports/runs/E4_26-11-18_exp01/E4_best_params.json` |
| Optuna study (gitignored) | `reports/runs/E4_26-11-18_exp01/E4_optuna.db` |
| Hình AUC theo trial | `reports/figures/E4_optuna_history.png` |
| Default fold metrics | `reports/tables/E4_26-11-18_exp01/E4_default_<model>_cv_folds_seed42.csv` |
| Default summary | `reports/tables/E4_26-11-18_exp01/E4_default_<model>_cv_summary_seed42.csv` |
| Default importance | `reports/tables/E4_26-11-18_exp01/E4_default_<model>_feature_importance_seed42.csv` |
| Default run JSON | `reports/runs/E4_26-11-18_exp01/E4_default_<model>_seed42.json` |
| MLflow | experiment `E4_tuning` |

## 13. Kết luận và bước tiếp

E4 trả lời phần tuning của RQ3 trên `plus_installments`, `class_weight`, seed 42:

- **Tuning có ích và nhất quán về dấu:** +0,00504 (LightGBM), +0,01189 (XGBoost), +0,00348 (CatBoost), dương ở 15/15 fold. Đây là gain thật của search space, không phải nhiễu fold như `class_weight` ở E3.
- **40 trial là dư:** đường cong bão hoà từ trial 4–33; 20 trial đủ trong 0,0011. RQ3 nên trình bày hình dạng đường cong, không nên nói cần 40 trial.
- **Không under-fit.** Trần 2000 vòng nằm đúng đỉnh (0,78624 @2000 vs 0,78470 @4000 vs 0,78144 @8000 trên LightGBM fold 0). Nâng trần là bước lùi, không phải bước tiến. Bản ghi E4 ban đầu khẳng định ngược lại và đã được sửa trong `DECISIONS.md`.
- **Lỗi thật là under-tuned:** `subsample` của LightGBM không có tác dụng vì `subsample_freq: 0`. Sửa ở pass sau.
- **Calibration vẫn hỏng** (Brier 0,154–0,178, ECE 0,255–0,302): E6 bắt buộc trước khi báo cáo bất kỳ xác suất nào.

Bước tiếp:

- **E5:** refit ba model bằng `final_params` (2.000 / 1.998 / 2.000 vòng, không early stopping, không validation pair), 3 seed 42/43/44, cùng fold; so sánh bằng DeLong, thêm weighted average và stacking trên OOF.
- **E6:** calibrate trên OOF; kiểm tra stacking như một ứng viên predictor cuối, vì meta-learner logistic có thể tự sửa thang xác suất.
