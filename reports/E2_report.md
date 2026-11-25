# E2 — Ablation nhóm feature cộng dồn

Báo cáo kết quả E2 (RQ1): đóng góp của từng nhóm feature phụ so với bảng `application` gốc.

## 1. Mục tiêu

Trả lời **RQ1**: các bảng dữ liệu phụ (`bureau`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments`) và các tỷ lệ kỹ thuật thêm bao nhiêu năng lực phân biệt so với riêng bảng `application`.

Cách đo: 7 mức cộng dồn 120 → 130 → 147 → 157 → 163 → 171 → 180 feature. Mỗi mức dùng LightGBM **tham số mặc định**, cùng 5 fold, cùng model seed 42, cùng metric. Không tune, không cân bằng lớp, không calibrate, không chạm hold-out.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E2_ablation.yaml` |
| Experiment | `E2_ablation` (mỗi feature set một run, variant = tên set) |
| Model | LightGBM, `params: {}` (mặc định), `n_jobs=-1`, `deterministic=true`, `force_col_wise=true` |
| Feature sets | `main` 120, `main_plus_ratios` 130, `plus_bureau` 147, `plus_previous` 157, `plus_pos` 163, `plus_credit_card` 171, `plus_installments` 180 |
| Categorical | 16 cột theo contract, truyền native cho LightGBM |
| Dataset | `data/processed/datasets/train.parquet`, contract `datasets_v1` |
| Split | 5 fold, split seed 42; hold-out 61.503 dòng **không đọc** |
| Model seed | 42 |
| Metric | ROC-AUC chính; kèm Gini, KS, PR-AUC, Brier, ECE 15 bins |

Lệnh chạy:

```bash
uv run python -m src.evaluation.ablation --config configs/experiments/E2_ablation.yaml
```

## 3. Dữ liệu vào

- Development: 246.008 dòng, tỷ lệ positive 0,08073, giống E1.
- Hold-out: 61.503 dòng, bị khóa; runner chỉ đọc số dòng từ footer Parquet.
- Feature set nhỏ nhất 120 cột, lớn nhất 180 cột; 16 cột categorical.
- Mỗi feature set đều đủ 5 fold (49.202 / 49.202 / 49.202 / 49.201 / 49.201 dòng).

## 4. Kết quả cộng dồn (AUC)

| Feature set | Nhóm thêm | n_features | AUC (mean ± std) | Δ so với set trước | Gain / nhóm |
|---|---|---:|---|---:|---:|
| `main` | `application_original` | 120 | **0,75421 ± 0,00337** | — | — |
| `main_plus_ratios` | `application` | 130 | 0,76056 ± 0,00273 | +0,00635 | +0,00635 |
| `plus_bureau` | `bureau` | 147 | 0,76513 ± 0,00216 | +0,00457 | +0,00457 |
| `plus_previous` | `previous` | 157 | 0,76820 ± 0,00137 | +0,00308 | +0,00308 |
| `plus_pos` | `pos` | 163 | 0,77200 ± 0,00123 | +0,00380 | +0,00380 |
| `plus_credit_card` | `credit_card` | 171 | 0,77593 ± 0,00181 | +0,00393 | +0,00393 |
| `plus_installments` | `installments` | 180 | **0,77836 ± 0,00178** | +0,00243 | +0,00243 |

Tổng gain từ 120 → 180 feature: **+0,02414 AUC** (+3,2% tương đối). Mọi bước đều dương, không nhóm nào làm giảm AUC.

`gain_per_group_auc` là phần AUC cộng dồn quy cho nhóm bước vào ở mức đó, không phải gain biên loại-một-nhóm (leave-one-group-out). Vì mỗi mức chỉ thêm đúng một nhóm nên gain bằng Δ so với mức trước.

## 5. AUC theo fold

| Feature set | n_features | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---|---|---|---|---|
| `main` | 120 | 0,7589 | 0,7501 | 0,7508 | 0,7546 | 0,7567 |
| `main_plus_ratios` | 130 | 0,7635 | 0,7570 | 0,7580 | 0,7636 | 0,7607 |
| `plus_bureau` | 147 | 0,7677 | 0,7627 | 0,7648 | 0,7675 | 0,7629 |
| `plus_previous` | 157 | 0,7693 | 0,7655 | 0,7687 | 0,7685 | 0,7690 |
| `plus_pos` | 163 | 0,7731 | 0,7698 | 0,7714 | 0,7729 | 0,7727 |
| `plus_credit_card` | 171 | 0,7769 | 0,7727 | 0,7753 | 0,7777 | 0,7770 |
| `plus_installments` | 180 | 0,7811 | 0,7755 | 0,7782 | 0,7788 | 0,7781 |

Cả **30/30** phép so sánh theo fold (6 bước × 5 fold) đều tăng AUC. Độ lệch giữa các fold giảm từ 0,00337 xuống 0,00178 khi thêm feature: mô hình ổn định hơn, không chỉ tốt hơn.

## 6. Các metric khác

| Feature set | Gini | KS | PR-AUC | Brier | ECE | Fit (s/fold) |
|---|---:|---:|---|---:|---:|---:|
| `main` | 0,50842 | 0,38012 | 0,23816 ± 0,00537 | 0,06810 ± 0,00026 | 0,00462 | 2,32 |
| `main_plus_ratios` | 0,52112 | 0,38922 | 0,24342 ± 0,00475 | 0,06778 ± 0,00020 | 0,00411 | 2,92 |
| `plus_bureau` | 0,53025 | 0,39607 | 0,25228 ± 0,00553 | 0,06739 ± 0,00019 | 0,00348 | 2,73 |
| `plus_previous` | 0,53640 | 0,40163 | 0,25473 ± 0,00508 | 0,06721 ± 0,00020 | 0,00417 | 2,81 |
| `plus_pos` | 0,54399 | 0,40608 | 0,25844 ± 0,00646 | 0,06701 ± 0,00028 | 0,00442 | 2,96 |
| `plus_credit_card` | 0,55186 | 0,41438 | 0,26351 ± 0,00759 | 0,06674 ± 0,00032 | 0,00438 | 3,04 |
| `plus_installments` | 0,55671 | 0,41866 | 0,26805 ± 0,00644 | 0,06654 ± 0,00027 | 0,00434 | 3,14 |

KS tăng 0,38012 → 0,41866, PR-AUC 0,23816 → 0,26805 (baseline 0,08073), Brier giảm 0,06810 → 0,06654. ECE gần như không đổi (~0,0035–0,0046), tức việc thêm feature không làm xấu calibration của LightGBM mặc định.

## 7. Trả lời RQ1

- **Có, feature phụ đóng góp rõ rệt.** Thêm 60 feature (10 tỷ lệ + 50 feature từ 5 bảng phụ) đưa AUC từ 0,75421 lên 0,77836, tức **+0,02414** (+3,2% tương đối), và giảm độ lệch fold.
- **Nhóm tỷ lệ ứng dụng là bước lớn nhất** (+0,00635, ~26% tổng gain) dù chỉ 10 cột. Đây là feature suy ra từ chính bảng `application`, nên gain này nói về *cách biểu diễn* dữ liệu sẵn có, không phải dữ liệu mới.
- **Bureau là nhóm dữ liệu ngoài mạnh nhất** (+0,00457, ~19%), tiếp theo là `credit_card` (+0,00393), `pos` (+0,00380), `previous` (+0,00308), `installments` (+0,00243).
- **Lợi ích giảm dần nhưng không âm:** 4 nhóm cuối mỗi nhóm dưới 0,005 AUC, nhưng cộng lại vẫn +0,01214. Không có bằng chứng nhóm nào dư thừa hoàn toàn trong cấu hình này.
- **Phạm vi trả lời:** RQ1 ở đây được trả lời trên đúng 180 predictor đã giao. Không có feature cửa sổ thời gian (`WIN_`) hay feature chọn trong fold, nên kết luận không bao gồm các biến thể đó. Báo cáo cuối phải nêu hạn chế này.

## 8. Feature importance của set cuối

Top 15 theo gain trung bình 5 fold ở set `plus_installments`:

| # | Feature | Importance |
|---|---|---:|
| 1 | `APP_EXT_SOURCE_MEAN` | 47.096,1 |
| 2 | `ORGANIZATION_TYPE` | 8.336,9 |
| 3 | `APP_CREDIT_ANNUITY_RATIO` | 3.713,7 |
| 4 | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | 3.290,5 |
| 5 | `APP_CREDIT_GOODS_RATIO` | 3.154,7 |
| 6 | `PREV_CREDIT_APPLICATION_RATIO_MEAN` | 3.076,9 |
| 7 | `EXT_SOURCE_3` | 2.762,1 |
| 8 | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` | 2.687,3 |
| 9 | `CC_LATEST_BALANCE_SUM` | 2.455,5 |
| 10 | `INS_OBS_PAYMENT_SUM` | 2.388,3 |
| 11 | `POS_LATEST_FUTURE_INSTALLMENTS_SUM` | 2.132,6 |
| 12 | `PREV_REFUSAL_RATE` | 2.011,0 |
| 13 | `EXT_SOURCE_1` | 1.915,7 |
| 14 | `AMT_ANNUITY` | 1.729,7 |
| 15 | `BUREAU_DAYS_CREDIT_MAX` | 1.599,7 |

`APP_EXT_SOURCE_MEAN` (trung bình ba `EXT_SOURCE`) chiếm ưu thế tuyệt đối, đúng kỳ vọng vì E1 đã cho thấy ba biến `EXT_SOURCE` dẫn đầu. Mỗi bảng phụ đều có đại diện trong top 15, khớp với gain dương ở mục 4.

25/180 feature có importance bằng 0 (E1: 11/120). Đây là các cột thưa hoặc gần hằng số; ứng viên cho bước chọn feature sau này, không loại ngay trong E2.

## 9. Kiểm tra tính hợp lệ

- `main` tái lập **chính xác** E1: cùng 5 giá trị AUC theo fold, mean 0,75421, std 0,00337.
- B1 khớp với bản tính lại từ 7 file fold CSV (sai khác < 1e-12).
- Mỗi feature set: 5 fold, 246.008 dòng OOF, `SK_ID_CURR` unique, một dự đoán/dòng.
- Không có ID hold-out trong OOF; `val.parquet` không được load; run JSON ghi `n_holdout_rows = 61.503` cho cả 7 run.
- Model seed 42, `data_version = datasets_v1`, `git_commit = 8d4b0e48` cho cả 7 run; mỗi run có `config_digest` riêng vì `features.use_set` khác nhau.
- Output smoke `*_sample500*` đã xoá; B1 full nằm ở `reports/tables/B1_ablation.csv`.

## 10. Chi phí chạy

| Feature set | Thời gian (s) | Peak RSS (MB) |
|---|---:|---:|
| `main` | 12,8 | 1.681 |
| `main_plus_ratios` | 15,9 | 1.968 |
| `plus_bureau` | 14,9 | 2.165 |
| `plus_previous` | 15,4 | 2.357 |
| `plus_pos` | 16,0 | 2.357 |
| `plus_credit_card` | 16,5 | 2.420 |
| `plus_installments` | 17,1 | 2.500 |
| **Tổng** | **108,6** | **2.500** |

Chi phí tăng chậm theo số feature, vẫn rất xa ngưỡng 16 GB.

## 11. Artifacts

| Loại | Đường dẫn |
|---|---|
| B1 ablation | `reports/tables/B1_ablation.csv` |
| Fold metrics | `reports/tables/E2_26-11-17_exp01/E2_<set>_cv_folds_seed42.csv` |
| Summary | `reports/tables/E2_26-11-17_exp01/E2_<set>_cv_summary_seed42.csv` |
| Importance | `reports/tables/E2_26-11-17_exp01/E2_<set>_feature_importance_seed42.csv` |
| OOF | `reports/predictions/E2_26-11-17_exp01/E2_<set>_lgbm_seed42_oof.parquet` |
| Run JSON | `reports/runs/E2_26-11-17_exp01/E2_<set>_lgbm_seed42.json` |
| MLflow | experiment `E2_ablation`, 7 run (một run mỗi feature set) |

`<set>` nhận một trong 7 tên ở mục 2.

## 12. Kết luận và bước tiếp

E2 trả lời RQ1: trên 180 predictor đã giao, feature phụ mang lại **+0,02414 AUC** so với bảng `application` gốc, mọi nhóm đều dương, nhóm tỷ lệ ứng dụng và `bureau` đóng góp nhiều nhất, và mô hình ổn định hơn giữa các fold. Tập 180 feature là ứng viên mặc định cho E3–E7.

Bước tiếp: **E3 (RQ3)** — cân bằng lớp (`scale_pos_weight`, undersampling, SMOTENC) fit trong từng fold, so với baseline `none` trên cùng feature set `plus_installments`, cùng fold, cùng seed.

Hạn chế cần nêu trong báo cáo cuối: chỉ 1 model seed, chưa có kiểm định ý nghĩa (DeLong ở Phase 7), split random không theo thời gian, gain là quy theo thứ tự cộng dồn chứ không phải gain biên, và không có feature `WIN_`.
