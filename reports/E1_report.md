# E1 — Baseline LightGBM

Báo cáo kết quả chạy E1 (điểm tham chiếu cho toàn bộ các phase sau).

## 1. Mục tiêu

Dựng mốc so sánh (reference point) cho RQ1 và RQ2: LightGBM với **tham số mặc định**, chỉ dùng 120 cột gốc của bảng `application`, 5-fold CV trên development set. Không tune, không cân bằng lớp, không calibrate, không chạm hold-out.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E1_baseline.yaml` |
| Experiment / variant | `E1_baseline` / `main_only_default` |
| Model | LightGBM, `params: {}` (mặc định), `n_jobs=-1`, `deterministic=true`, `force_col_wise=true` |
| Feature set | `main` = 120 cột `application_original` |
| Categorical | 16 cột (contract khai báo), truyền native cho LightGBM |
| Dataset | `data/processed/datasets/train.parquet`, contract `datasets_v1` |
| Split | 5 fold, seed 42; hold-out 61.503 dòng **không đọc** |
| Model seed | 42 |
| Ngưỡng báo cáo | 0,5 + sweep 0,1/0,2/0,3/0,4, ECE 15 bins |

Lệnh chạy:

```bash
uv run python -m src.cv.runner --config configs/experiments/E1_baseline.yaml
```

## 3. Dữ liệu vào

- Development: 246.008 dòng × 183 cột (ID, TARGET, fold + 180 predictors), dùng 120 predictors.
- Hold-out: 61.503 dòng, bị khóa; runner chỉ đọc số dòng từ footer Parquet để đối chiếu metadata.
- Tỷ lệ positive trong development: 0,08073.
- Bộ nhớ frame sau khi chọn cột: 262 MB.

## 4. Kết quả CV theo fold

| Fold | n_train | n_valid | Fit (s) | AUC | Gini | KS | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 196.806 | 49.202 | 2,29 | 0,7589 | 0,5178 | 0,3850 | 0,2396 | 0,0679 | 0,0047 |
| 1 | 196.806 | 49.202 | 2,12 | 0,7501 | 0,5002 | 0,3658 | 0,2327 | 0,0684 | 0,0042 |
| 2 | 196.806 | 49.202 | 2,10 | 0,7508 | 0,5015 | 0,3752 | 0,2311 | 0,0685 | 0,0057 |
| 3 | 196.807 | 49.201 | 2,14 | 0,7546 | 0,5092 | 0,3810 | 0,2436 | 0,0679 | 0,0043 |
| 4 | 196.807 | 49.201 | 2,18 | 0,7567 | 0,5134 | 0,3935 | 0,2439 | 0,0678 | 0,0041 |

## 5. Tổng hợp (mean ± std trên 5 fold)

| Metric | Giá trị | Baseline |
|---|---|---|
| ROC-AUC | **0,75421 ± 0,00337** | 0,5 |
| Gini | 0,50843 ± 0,00674 | — |
| KS | 0,38012 ± 0,00932 | — |
| PR-AUC | 0,23816 ± 0,00537 | 0,08073 |
| Brier | 0,06810 ± 0,00026 | 0,07421 |
| ECE (15 bins) | 0,00462 ± 0,00060 | — |

AUC nằm trong khoảng kỳ vọng 0,74–0,76 cho bảng `application` gốc. Chênh lệch fold lớn nhất − nhỏ nhất = 0,0088, tức mô hình ổn định trên 5 fold.

## 6. Precision / recall theo ngưỡng

| Ngưỡng | Precision | Recall | F1 |
|---|---|---|---|
| 0,1 | 0,1885 | 0,5815 | 0,2847 |
| 0,2 | 0,2843 | 0,2868 | 0,2856 |
| 0,3 | 0,3723 | 0,1327 | 0,1956 |
| 0,4 | 0,4689 | 0,0563 | 0,1005 |
| 0,5 | 0,5510 | 0,0201 | 0,0387 |

Với base rate 8%, ngưỡng 0,5 chỉ bắt được ~2% positive nên không dùng để đánh giá. Vùng 0,1–0,2 cho F1 cao nhất (~0,285); báo cáo các phase sau nên nêu operating point theo ngưỡng chứ không chỉ 0,5.

## 7. Calibration

- Brier 0,06810 thấp hơn baseline 0,07421.
- ECE 0,00462 trên 15 bins.
- Xác suất dự đoán trung bình 0,08058 so với tỷ lệ positive thực 0,08073 → gần như trùng khớp.

Kết luận: LightGBM mặc định đã calibrated tốt trên tập này; E6 vẫn cần kiểm tra lại sau khi tune, vì tuning có thể làm lệch calibration.

## 8. Feature importance (gain, trung bình 5 fold)

| # | Feature | Importance |
|---|---|---|
| 1 | `EXT_SOURCE_3` | 26.825,6 |
| 2 | `EXT_SOURCE_2` | 22.082,4 |
| 3 | `ORGANIZATION_TYPE` | 8.726,5 |
| 4 | `EXT_SOURCE_1` | 8.362,3 |
| 5 | `DAYS_EMPLOYED` | 3.888,0 |
| 6 | `AMT_CREDIT` | 3.111,1 |
| 7 | `DAYS_BIRTH` | 2.824,6 |
| 8 | `AMT_GOODS_PRICE` | 2.670,0 |
| 9 | `AMT_ANNUITY` | 2.579,8 |
| 10 | `OCCUPATION_TYPE` | 1.866,3 |
| 11 | `DAYS_ID_PUBLISH` | 1.682,3 |
| 12 | `OWN_CAR_AGE` | 1.580,4 |
| 13 | `NAME_EDUCATION_TYPE` | 1.520,0 |
| 14 | `CODE_GENDER` | 1.331,6 |
| 15 | `DAYS_LAST_PHONE_CHANGE` | 1.218,3 |
| 16 | `DAYS_REGISTRATION` | 1.180,4 |
| 17 | `FLAG_DOCUMENT_3` | 820,3 |
| 18 | `AMT_INCOME_TOTAL` | 796,5 |
| 19 | `REGION_POPULATION_RELATIVE` | 618,1 |
| 20 | `DEF_30_CNT_SOCIAL_CIRCLE` | 612,5 |

Thứ hạng đúng kỳ vọng của bộ dữ liệu Home Credit: ba `EXT_SOURCE` dẫn đầu, sau đó là nhóm thu nhập/khoản vay và nhân khẩu học. Không có `SK_ID_CURR`, không có cột nào suy ra từ `TARGET` → không có dấu hiệu leakage.

11/120 feature có importance bằng 0: 10 cột `FLAG_DOCUMENT_*` hiếm gặp và `FLAG_MOBIL` (246.007/246.008 dòng bằng 1, coi như hằng số). Ứng viên để loại ở E2 hoặc E3, không loại ngay trong E1.

## 9. Kiểm tra tính hợp lệ

- OOF: 246.008 dòng, `SK_ID_CURR` unique, đúng 1 dự đoán/dòng.
- Phân bố fold: 49.202 / 49.202 / 49.202 / 49.201 / 49.201.
- Không có ID hold-out trong OOF; `val.parquet` không được load.
- Tỷ lệ positive OOF 0,080729 khớp development set.
- Xác suất trong [0,0051; 0,8648], không bão hòa ở 0/1.
- `git_commit` = `bb56f78`, `config_digest` = `d9962f00…`, `data_version` = `datasets_v1`; chạy lại cho cùng kết quả (deterministic).

## 10. Chi phí chạy

| Hạng mục | Giá trị |
|---|---|
| Tổng thời gian | 12,0 s |
| Fit trung bình / fold | 2,16 s |
| Predict trung bình / fold | 0,15 s |
| Peak RSS | 2.030 MB |

## 11. Artifacts

| Loại | Đường dẫn |
|---|---|
| Fold metrics | `reports/tables/E1_26-11-16_exp01/E1_baseline_cv_folds_seed42.csv` |
| Summary | `reports/tables/E1_26-11-16_exp01/E1_baseline_cv_summary_seed42.csv` |
| Importance | `reports/tables/E1_26-11-16_exp01/E1_baseline_feature_importance_seed42.csv` |
| OOF | `reports/predictions/E1_26-11-16_exp01/E1_baseline_lgbm_seed42_oof.parquet` |
| Run JSON | `reports/runs/E1_26-11-16_exp01/E1_baseline_lgbm_seed42.json` |
| MLflow | experiment `E1_baseline`, run `0c681976a6db4ec3806edb2f861e3925` |

## 12. Kết luận và bước tiếp

E1 đạt **AUC 0,75421 ± 0,00337**, đúng vùng kỳ vọng, ổn định giữa các fold, calibration sẵn tốt, chi phí rất thấp (12 s). Đây là mốc để đo:

- **E2 (RQ1):** 7 mức cộng dồn 120 → 130 → 147 → 157 → 163 → 171 → 180 feature; mỗi mức so với mốc này.
- **E3–E7:** mọi cải thiện phải vượt 0,7542 với cùng fold, cùng metric.

Hạn chế cần nêu trong báo cáo cuối: chỉ 1 seed, chỉ bảng `application`, chưa tune, chưa calibrate, hold-out chưa dùng.
