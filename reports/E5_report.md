# E5 — So sánh model và ensemble

Báo cáo kết quả E5 (RQ2): ba GBDT đã tune ở E4 khác nhau thế nào về AUC, KS và calibration, và ensemble có thêm được gì trên cùng feature set.

## 1. Mục tiêu

Trả lời **RQ2**: sau khi tune, LightGBM, XGBoost và CatBoost xếp hạng và hiệu chỉnh xác suất khác nhau ra sao trên `plus_installments`, và **weighted average** hoặc **stacking** có vượt được model đơn tốt nhất hay không.

Cách đo: ba model refit với `final_params` của E4 (đã cố định số vòng lặp, không early stopping), giữ nguyên `class_weight` của E3, cùng 5 fold, ba model seed 42/43/44. Hai arm ensemble được **cross-fit**: trọng số và meta-learner fit trên OOF của các fold khác, áp lên fold đang giữ. Không tune lại, không calibrate, không chạm hold-out.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E5_models.yaml` |
| Driver | `src/evaluation/model_comparison.py` |
| Experiment | `E5_models`, run `E5_26-11-24_exp01` |
| Model | LightGBM, XGBoost, CatBoost với `final_params` từ `reports/runs/E4_26-11-18_exp01/E4_best_params.json` |
| Số vòng lặp | Cố định: LightGBM 2000, XGBoost 1998, CatBoost 2000; không early stopping, không validation pair |
| Feature set | `plus_installments` = 180 predictor, 16 cột categorical |
| Imbalance | `class_weight`, `scale_pos_weight = n_negative / n_positive` (quyết định E3) |
| Split | 5 fold, split seed 42; hold-out 61.503 dòng **không đọc** |
| Model seed | 42, 43, 44 |
| Ensemble | `weighted_average` (simplex grid bước 0,05, minimize log loss) và `stacking` (logistic regression trên 3 xác suất) |
| Metric | ROC-AUC chính; kèm Gini, KS, PR-AUC, Brier, ECE 15 bins, precision/recall/F1 |
| Ngưỡng báo cáo | 0,5 + sweep 0,1/0,2/0,3/0,4 |
| Kiểm định | DeLong pairwise trên OOF gộp của từng seed |

Lệnh chạy:

```bash
uv run python -m src.evaluation.model_comparison --config configs/experiments/E5_models.yaml
```

Dựng lại bảng và hình từ OOF đã có, không refit:

```bash
uv run python -m src.evaluation.model_comparison --config configs/experiments/E5_models.yaml --table-only
```

## 3. Dữ liệu vào

- Development: 246.008 dòng, 180 predictor, tỷ lệ positive 0,08073, giống E1-E4.
- Hold-out: 61.503 dòng, bị khóa; cả 9 run JSON ghi `n_holdout_rows = 61503`, `val.parquet` không được load.
- Missing: 109/180 cột có NaN (21,9% ma trận), không điền; cả ba model xử lý missing native.
- Mỗi fold: 196.806 / 196.807 dòng fit, 49.202 / 49.201 dòng validation.
- XGBoost target-encode 16 cột categorical trong `fit` bằng nested 5-fold (chỉ dòng training của fold), CatBoost map NaN thành level `__MISSING__`, LightGBM dùng categorical native.

## 4. Kết quả CV theo fold (ROC-AUC)

| Model | Seed | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
|---|---|---:|---:|---:|---:|---:|---|
| LightGBM | 42 | 0,78624 | 0,78234 | 0,78410 | 0,78387 | 0,78466 | 0,78424 ± 0,00126 |
| LightGBM | 43 | 0,78644 | 0,78203 | 0,78344 | 0,78357 | 0,78422 | 0,78394 ± 0,00144 |
| LightGBM | 44 | 0,78661 | 0,78264 | 0,78315 | 0,78389 | 0,78479 | 0,78421 ± 0,00140 |
| XGBoost | 42 | 0,78929 | 0,78527 | 0,78629 | 0,78731 | 0,78765 | 0,78716 ± 0,00135 |
| XGBoost | 43 | 0,78898 | 0,78496 | 0,78648 | 0,78763 | 0,78785 | 0,78718 ± 0,00137 |
| XGBoost | 44 | 0,78953 | 0,78472 | 0,78710 | 0,78757 | 0,78701 | 0,78719 ± 0,00153 |
| CatBoost | 42 | 0,78761 | 0,78400 | 0,78515 | 0,78762 | 0,78597 | 0,78607 ± 0,00141 |
| CatBoost | 43 | 0,78825 | 0,78363 | 0,78561 | 0,78729 | 0,78636 | 0,78623 ± 0,00157 |
| CatBoost | 44 | 0,78793 | 0,78422 | 0,78531 | 0,78769 | 0,78567 | 0,78616 ± 0,00143 |

Hai điểm:

- LightGBM seed 42 cho **đúng** 0,78424 của E4, tức refit với `final_params` và không early stopping tái lập được kết quả tune. Harness E5 chạy đúng protocol E4.
- Thứ hạng giữa ba model ổn định qua cả ba seed: XGBoost > CatBoost > LightGBM. Fold 1 là fold khó nhất cho cả ba (thấp hơn fold 0 khoảng 0,004-0,005).

## 5. Tổng hợp B2 (mean ± std qua 3 seed)

Mỗi metric lấy trung bình 5 fold trong một seed, rồi mean ± std qua 3 seed (ddof=0). Toàn bộ metric được tính lại từ OOF, không copy từ CSV theo fold.

| Arm | ROC-AUC | Gini | KS | PR-AUC | Brier | ECE (15 bins) | Precision @0,5 | Recall @0,5 |
|---|---|---|---|---|---|---|---:|---:|
| LightGBM | 0,78413 ± 0,00014 | 0,56826 | 0,42918 | 0,27549 | 0,16601 | 0,28082 | 0,1958 | 0,6590 |
| XGBoost | **0,78718 ± 0,00001** | 0,57435 | 0,43519 | 0,27827 | 0,17252 | 0,29170 | 0,1928 | 0,6768 |
| CatBoost | 0,78615 ± 0,00006 | 0,57231 | 0,43311 | 0,27673 | 0,17529 | 0,29726 | 0,1905 | 0,6815 |
| Weighted average | 0,78413 ± 0,00014 | 0,56826 | 0,42918 | 0,27549 | 0,16601 | 0,28082 | 0,1958 | 0,6590 |
| Stacking | **0,78783 ± 0,00008** | 0,57565 | 0,43537 | 0,27937 | **0,06607** | **0,00561** | 0,6556 | 0,0003 |

Std qua seed rất nhỏ (≤ 0,00014) so với std qua fold (~0,0013-0,0015), tức chênh lệch giữa ba model **không** do may mắn của seed; nó là khác biệt có hệ thống giữa ba implementation.

## 6. Kiểm định DeLong

Pairwise trên OOF gộp của từng seed (246.008 dòng), `src/evaluation/delong.py` (Sun & Xu midrank). ΔAUC = AUC(B) − AUC(A).

| Cặp | ΔAUC seed 42 | p | ΔAUC seed 43 | p | ΔAUC seed 44 | p | Kết luận |
|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost − LightGBM | +0,00292 | 3,4e-14 | +0,00326 | 2,6e-17 | +0,00297 | 1,2e-14 | có ý nghĩa, cả 3 seed |
| CatBoost − LightGBM | +0,00183 | 4,5e-05 | +0,00230 | 3,9e-07 | +0,00196 | 1,3e-05 | có ý nghĩa, cả 3 seed |
| XGBoost − CatBoost | +0,00109 | 0,0030 | +0,00095 | 0,0101 | +0,00101 | 0,0055 | có ý nghĩa, cả 3 seed |
| Stacking − XGBoost | +0,00057 | 0,0020 | +0,00075 | 1,4e-05 | +0,00064 | 0,00017 | có ý nghĩa, cả 3 seed |
| Stacking − CatBoost | +0,00166 | 7,8e-13 | +0,00170 | 5,1e-17 | +0,00165 | 2,5e-14 | có ý nghĩa, cả 3 seed |
| Stacking − LightGBM | +0,00349 | 3,6e-33 | +0,00401 | 2,8e-30 | +0,00361 | 3,2e-28 | có ý nghĩa, cả 3 seed |
| LightGBM − Weighted average | 0,00000 | 1,0 | 0,00000 | 1,0 | 0,00000 | 1,0 | trùng khớp tuyệt đối |

DeLong ở đây kiểm định trên dòng OOF của development set, không phải hold-out; kiểm định trên hold-out thuộc Phase 9 và ghi vào `B4_delong_pairs.csv`.

## 7. Ensemble

### 7.1 Weighted average suy biến thành LightGBM

`weighted_average` chọn **`{lightgbm: 1.0, xgboost: 0.0, catboost: 0.0}` ở cả 5 fold của cả 3 seed**, nên hàng của nó trong B2 trùng khít hàng LightGBM (DeLong p = 1,0).

Cơ chế, theo thứ tự:

1. Trọng số được chọn bằng cách minimize **log loss** trên simplex bước 0,05 (231 ứng viên/fold).
2. Log loss là hàm lồi theo trọng số, nên cực tiểu nằm ở đỉnh simplex hoặc ở trong; không có cực tiểu "gần đều" nào bị bỏ sót.
3. Ba model xếp hạng gần như nhau (Spearman 0,968-0,979, xem mục 9), nên phần **thứ tự** của loss không cải thiện được bằng cách trộn.
4. Phần còn lại là **thang xác suất**, và cả ba đều bị `class_weight` đẩy lên: Brier 0,16601 (LightGBM) / 0,17252 (XGBoost) / 0,17529 (CatBoost), p trung bình OOF ~0,28-0,30 so với base rate 0,08073.
5. Trung bình lồi không thể ra khỏi khoảng giữa các thành viên, nên điểm tốt nhất là thành viên **ít hỏng thang nhất**, tức LightGBM.

Kết luận: code chạy đúng như thiết kế; **objective là câu hỏi sai cho bộ ba này**. Arm được giữ và báo cáo như bằng chứng, không dựng lại. Hai phương án bị loại (ghi ở `DECISIONS.md`):

- **Fit trọng số theo AUC:** mỗi fold chọn 1 trong 231 ứng viên bằng metric xếp hạng, tức thêm nhiễu chọn lọc; trọng số sẽ dao động qua fold và seed, còn Brier/ECE của arm mất ý nghĩa vì xác suất trộn mang thang lẫn lộn. Không đổi lại được gì đo được.
- **Cố định trọng số 1/3:** arm tồn tại để **fit** trọng số hiệu dụng, nên cố định a priori làm câu hỏi đó vô nghĩa; và ba model chỉ nằm trong khoảng 0,78413-0,78718, hẹp hơn nhiều so với std fold ~0,0013, nên 1/3 chỉ là một điểm tùy ý trong khoảng đó.

### 7.2 Stacking là arm ensemble duy nhất có giá trị

| Seed | AUC OOF gộp | Δ so với XGBoost |
|---|---:|---:|
| 42 | 0,78772 | +0,00057 |
| 43 | 0,78793 | +0,00075 |
| 44 | 0,78781 | +0,00064 |

Stacking vượt **cả ba** model đơn trên **cả ba** seed, với p ≤ 0,002. Mức tăng +0,00065 so với XGBoost là nhỏ nhưng có hệ thống, không phải nhiễu: std qua seed của XGBoost chỉ 0,00001.

Meta-learner là logistic regression trên 3 cột xác suất, fit trên OOF của 4 fold còn lại, áp lên fold đang giữ, nên không dòng nào đóng góp vào dự đoán của chính nó.

## 8. Calibration

| Arm | Brier | Brier baseline | ECE 15 bins | p trung bình OOF |
|---|---:|---:|---:|---:|
| LightGBM | 0,16601 | 0,07421 | 0,28082 | ~0,30 |
| XGBoost | 0,17252 | 0,07421 | 0,29170 | ~0,29 |
| CatBoost | 0,17529 | 0,07421 | 0,29726 | ~0,30 |
| Weighted average | 0,16601 | 0,07421 | 0,28082 | ~0,30 |
| **Stacking** | **0,06607** | 0,07421 | **0,00561** | ~0,08 |

Ba điểm:

- **Stacking sửa gần hết thiệt hại của `class_weight` mà không cần E6.** Logistic meta-learner học lại thang, nên Brier 0,06607 và ECE 0,00561 tốt hơn cả mốc E1 `none` (0,06810 / 0,00462 trên 120 cột). Hệ quả cho Phase 8: E6 phải coi **stacking là ứng viên final predictor**, không chỉ áp Platt/isotonic lên một model đơn.
- Ba model đơn vẫn hỏng thang như E3/E4 dự báo (ECE ~0,28-0,30). Trước khi E6 xong, chỉ được so sánh chúng bằng metric xếp hạng.
- Precision/recall của stacking tại ngưỡng 0,5 (0,6556 / 0,0003) **không dùng được**: meta-learner cho đầu ra thứ tự đúng nhưng không đặt quanh 0,5. Với stacking chỉ báo cáo metric xếp hạng và Brier/ECE; ngưỡng vận hành phải chọn trên PR curve sau E6.

## 9. Vì sao mức tăng nhỏ

Chẩn đoán độ đa dạng, chạy ad-hoc trên OOF đã lưu, **không giữ artifact** nên chỉ nêu như quan sát:

- Spearman giữa các cặp dự đoán: 0,968-0,979. Top-5% Jaccard: 0,698-0,752.
- Model đơn tốt nhất mỗi seed: 0,78715-0,78718. Biên của mọi cách trộn đã thử (simplex log loss, trọng số đều, rank average, stacking, cặp tốt nhất mỗi seed): 0,7877-0,7879.
- Nghĩa là **trần của ensemble trên feature set này là ~0,7878**, và stacking đã chạm trần đó. Ba GBDT trên cùng 180 cột của bảng `application` gần như là ba bản sao.
- Hệ quả: không đi sâu hơn vào stacking nhiều tầng. Khoảng cách tới top công khai (~0,798-0,804) là vấn đề **feature**, không phải ensemble.

## 10. Feature importance (seed 42)

Ba model không cùng loại importance (LightGBM gain, XGBoost gain chuẩn hoá, CatBoost PredictionValuesChange), nên chỉ so **thứ hạng**, không so giá trị.

| # | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| 1 | `APP_EXT_SOURCE_MEAN` | `APP_EXT_SOURCE_MEAN` | `APP_EXT_SOURCE_MEAN` |
| 2 | `ORGANIZATION_TYPE` | `CC_LATEST_BALANCE_SUM` | `APP_CREDIT_ANNUITY_RATIO` |
| 3 | `EXT_SOURCE_3` | `APP_CREDIT_GOODS_RATIO` | `EXT_SOURCE_3` |
| 4 | `EXT_SOURCE_2` | `PREV_REFUSAL_RATE` | `PREV_CREDIT_APPLICATION_RATIO_MEAN` |
| 5 | `APP_CREDIT_ANNUITY_RATIO` | `NAME_EDUCATION_TYPE` | `POS_LATEST_FUTURE_INSTALLMENTS_SUM` |
| 6 | `PREV_CREDIT_APPLICATION_RATIO_MEAN` | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | `EXT_SOURCE_1` |
| 7 | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` | `AMT_ANNUITY` |
| 8 | `EXT_SOURCE_1` | `APP_EMPLOYMENT_YEARS` | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` |
| 9 | `APP_CREDIT_GOODS_RATIO` | `POS_N_LOANS` | `APP_CREDIT_GOODS_RATIO` |
| 10 | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` | `CODE_GENDER` | `DAYS_BIRTH` |

`APP_EXT_SOURCE_MEAN` dẫn đầu ở cả ba, và nó dẫn rất xa: gain LightGBM 1,67 triệu, gấp 2,4 lần feature thứ hai. Cấu trúc chung giống E2-E4 (EXT_SOURCE, tỷ lệ khoản vay, rồi mỗi bảng phụ có đại diện), nhưng XGBoost và CatBoost dùng categorical và cột đếm nhiều hơn LightGBM. Đây là đầu vào cho RQ4 (top-k Jaccard, Spearman) ở Phase 10.

## 11. Kiểm tra tính hợp lệ

- LightGBM seed 42 tái lập đúng AUC 0,78424 của E4 (tuned), nên E5 refit đúng protocol.
- OOF: 246.008 dòng/seed, `SK_ID_CURR` unique, 1 dự đoán/dòng, positive rate 0,080729 khớp development set.
- `n_holdout_rows = 61503` trong cả 9 run JSON; `val.parquet` không được load; không có ID hold-out trong OOF.
- Ensemble cross-fit: trọng số/meta-learner của fold *k* chỉ fit trên OOF của 4 fold còn lại; test ở `tests/test_ensemble.py`.
- Xác suất không bão hoà; không có NaN trong OOF (hàm `align_oof` raise nếu có).
- `data_version = datasets_v1`, `git_commit = cb685ec` cho cả 9 run; `config_digest` khác nhau theo từng model (`f0485272` LightGBM, `8809f9cf` XGBoost, `ed05d1a4` CatBoost).
- Output smoke `*_sample500*` đã xoá sau lần chạy thử.

## 12. Chi phí chạy

| Model | Seed | Tổng (s) | Fit (s/fold) | Predict (s/fold) | Peak RSS (MB) |
|---|---|---:|---:|---:|---:|
| LightGBM | 42 / 43 / 44 | 153,2 / 159,7 / 160,8 | 29,5 / 30,8 / 31,0 | 1,01 / 1,05 / 1,04 | 2.854 |
| XGBoost | 42 / 43 / 44 | 204,4 / 199,6 / 190,2 | 40,4 / 39,4 / 37,5 | 0,39 / 0,40 / 0,39 | 2.973 |
| CatBoost | 42 / 43 / 44 | 671,7 / 652,6 / 650,7 | 134,0 / 130,1 / 129,8 | 0,29 / 0,28 / 0,29 | 3.801 |
| **Tổng 9 run** | — | **3.043 (50,7 phút)** | — | — | **3.801** |

CatBoost chiếm 65% thời gian với AUC thấp hơn XGBoost. Hai arm ensemble gần như miễn phí (grid 231 ứng viên/fold + một logistic regression), nằm trong thời gian ghi bảng. Peak 3,80 GB, dưới ngưỡng 16 GB.

## 13. Artifacts

| Loại | Đường dẫn |
|---|---|
| Bảng so sánh (B2) | `reports/tables/B2_model_comparison.csv` |
| DeLong pairs | `reports/tables/E5_delong_pairs.csv` |
| Fold metrics | `reports/tables/E5_26-11-24_exp01/E5_<model>_cv_folds_seed<k>.csv` |
| Summary | `reports/tables/E5_26-11-24_exp01/E5_<model>_cv_summary_seed<k>.csv` |
| Importance | `reports/tables/E5_26-11-24_exp01/E5_<model>_feature_importance_seed<k>.csv` |
| OOF | `reports/predictions/E5_26-11-24_exp01/E5_<model>_seed<k>_oof.parquet` |
| Run JSON | `reports/runs/E5_26-11-24_exp01/E5_<model>_seed<k>.json` |
| Trọng số ensemble | `reports/runs/E5_26-11-24_exp01/E5_ensembles.json` |
| Hình ROC/PR | `reports/figures/E5_roc_pr.png` |

`<model>` nhận `lightgbm`, `xgboost`, `catboost`; `<k>` nhận `42`, `43`, `44`.

## 14. Kết luận và bước tiếp

E5 trả lời RQ2 trên `plus_installments` (180 predictor), `class_weight`, ba seed:

- **XGBoost là model đơn tốt nhất:** 0,78718 ± 0,00001, hơn CatBoost +0,00102 và LightGBM +0,00304, cả hai đều có ý nghĩa DeLong trên cả ba seed. XGBoost cũng ổn định nhất qua seed (std 0,00001).
- **Stacking là arm ensemble duy nhất có giá trị:** 0,78783 ± 0,00008, hơn XGBoost **+0,00065** với p ≤ 0,002 trên cả ba seed, và **sửa gần hết calibration** (Brier 0,06607, ECE 0,00561) mà không cần E6.
- **Weighted average suy biến**, trùng khít LightGBM: minimize log loss trên ba model xếp hạng gần nhau chỉ còn phân biệt bằng thang xác suất, và LightGBM là thành viên ít hỏng thang nhất. Báo cáo như bằng chứng, không dựng lại.
- **Mức tăng nhỏ vì thiếu đa dạng, không vì thiếu thuật toán.** Ba model tương quan 0,968-0,979 và trần trộn ~0,7878; muốn vượt thì phải thêm feature, không phải thêm tầng ensemble.
- **E6 bắt buộc với model đơn, tùy chọn với stacking.** Chỉ stacking có xác suất dùng được ngay; nếu chọn model đơn làm final thì phải calibrate.

Bước tiếp:

- **E6 (Phase 8):** Platt và isotonic trên OOF, đánh giá cross-fit (fit trên OOF các fold khác, áp lên fold đang giữ). Đưa **stacking** vào danh sách ứng viên cạnh XGBoost. Tiêu chí chọn: AUC giữ nguyên (calibration đơn điệu không đổi thứ hạng), Brier/ECE về mức `none` hoặc thấp hơn.
- **Phase 9:** chỉ chạy `holdout.py --confirm-final` **một lần**, sau khi E6 chốt model và calibrator.
- **Ngoài phạm vi Phase 7:** LightGBM `subsample` 0,70 của E4 chưa bao giờ có hiệu lực vì `subsample_freq` mặc định 0; bật `subsample_freq: 1` cho fold 0 kết quả 0,78756 (+0,0013, chỉ một fold, không giữ artifact). Đây là một trong bảy chiều search của E4 bị lãng phí, sửa ở lần chạy model sau, không phải rerun E4.

Hạn chế cần nêu trong báo cáo cuối: split random không theo thời gian; không có feature cửa sổ thời gian (`WIN_`) nên RQ1 chỉ trả lời cho 180 predictor đã giao; trần ensemble ~0,7878 thấp hơn top công khai ~0,798-0,804 chủ yếu vì feature; DeLong ở E5 là trên development OOF, kiểm định hold-out thuộc Phase 9; các chẩn đoán độ đa dạng ở mục 9 là ad-hoc, không có artifact.
