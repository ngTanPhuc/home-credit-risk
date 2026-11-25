# E3 — Cân bằng lớp (imbalance handling)

Báo cáo kết quả E3 (RQ3): kỹ thuật xử lý mất cân bằng lớp nào tốt nhất trên cùng feature set, cùng fold, cùng seed.

## 1. Mục tiêu

Trả lời **RQ3** (phần cân bằng lớp): với LightGBM mặc định trên `plus_installments`, xử lý mất cân bằng lớp bằng **trọng số lớp**, **undersampling** hay **SMOTENC** có cải thiện năng lực phân biệt và calibration so với huấn luyện nguyên dữ liệu (`none`) hay không.

Cách đo: 4 variant, mỗi variant là một fit/resample object **dựng lại trong từng fold** và chỉ áp lên phần training của fold đó. Validation không bị resample, không bị điền missing. Không tune, không calibrate, không chạm hold-out. Mốc so sánh là `none`, tức đúng cấu hình E2 `plus_installments`.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E3_imbalance.yaml` |
| Experiment | `E3_imbalance` (mỗi variant một run, `variant` = tên variant) |
| Model | LightGBM, `params: {}` (mặc định), `n_jobs=-1`, `deterministic=true`, `force_col_wise=true` |
| Feature set | `plus_installments` = 180 predictor, 16 cột categorical truyền native |
| Variant `none` | Không đổi gì; train trên 100% dòng của fold-training |
| Variant `class_weight` | Sample weight từng dòng, dương = `n_negative / n_positive` (tương đương `scale_pos_weight`), âm = 1,0 |
| Variant `undersample` | `RandomUnderSampler(sampling_strategy=0.5, random_state=42)` |
| Variant `smotenc` | `SMOTENC(categorical_features=16 cột contract, sampling_strategy=0.5, k_neighbors=5, random_state=42)`; điền median (numeric) / mode (categorical) trên fold-training trước khi resample |
| Dataset | `data/processed/datasets/train.parquet`, contract `datasets_v1` |
| Split | 5 fold, seed 42; hold-out 61.503 dòng **không đọc** |
| Model seed | 42 |
| Metric | ROC-AUC chính; kèm Gini, KS, PR-AUC, Brier, ECE 15 bins, precision/recall/F1 |
| Ngưỡng báo cáo | 0,5 + sweep 0,1/0,2/0,3/0,4 |

Lệnh chạy:

```bash
uv run python -m src.evaluation.imbalance_comparison --config configs/experiments/E3_imbalance.yaml
```

`sampling_strategy = 0.5` nghĩa là sau resample, số dòng minority = 0,5 × số dòng majority. Variant tự động thành no-op khi fold đã đạt tỷ lệ này (imblearn raise thay vì bỏ qua).

## 3. Dữ liệu vào

- Development: 246.008 dòng, 180 predictor, tỷ lệ positive 0,08073, giống E1/E2.
- Hold-out: 61.503 dòng, bị khóa; runner chỉ đọc số dòng từ footer Parquet.
- Missing: 109/180 cột có NaN, tổng 9.694.754 ô = **21,9%** ma trận. Notebook 02 chuyển sentinel (365243, `XNA`, chuỗi rỗng) thành missing, **không điền**; LightGBM xử lý NaN native nên chỉ variant `smotenc` phải điền, và điền trong từng fold-training.
- Số dòng thực sự dùng để fit mỗi fold: `none` và `class_weight` 196.806 / 196.807, `undersample` 47.664, `smotenc` 271.377 / 271.378 (thêm 74.571 dòng tổng hợp, toàn bộ là minority).

## 4. Kết quả CV theo fold

| Variant | Fold | n_train | n_train_fit | n_valid | Fit (s) | AUC | Gini | KS | PR-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| `none` | 0 | 196.806 | 196.806 | 49.202 | 3,61 | 0,78111 | 0,56221 | 0,41888 | 0,26951 | 0,06638 | 0,00396 |
| `none` | 1 | 196.806 | 196.806 | 49.202 | 3,36 | 0,77551 | 0,55103 | 0,41894 | 0,25671 | 0,06705 | 0,00483 |
| `none` | 2 | 196.806 | 196.806 | 49.202 | 3,16 | 0,77822 | 0,55645 | 0,41549 | 0,26865 | 0,06652 | 0,00449 |
| `none` | 3 | 196.807 | 196.807 | 49.201 | 3,29 | 0,77881 | 0,55763 | 0,41843 | 0,27676 | 0,06630 | 0,00482 |
| `none` | 4 | 196.807 | 196.807 | 49.201 | 3,16 | 0,77813 | 0,55626 | 0,42155 | 0,26862 | 0,06645 | 0,00361 |
| `class_weight` | 0 | 196.806 | 196.806 | 49.202 | 3,59 | 0,78128 | 0,56255 | 0,41762 | 0,27247 | 0,17810 | 0,30087 |
| `class_weight` | 1 | 196.806 | 196.806 | 49.202 | 3,34 | 0,77782 | 0,55564 | 0,41905 | 0,25590 | 0,17827 | 0,30015 |
| `class_weight` | 2 | 196.806 | 196.806 | 49.202 | 3,46 | 0,77878 | 0,55756 | 0,41675 | 0,27116 | 0,17645 | 0,29753 |
| `class_weight` | 3 | 196.807 | 196.807 | 49.201 | 3,40 | 0,77842 | 0,55685 | 0,41914 | 0,27696 | 0,17774 | 0,30032 |
| `class_weight` | 4 | 196.807 | 196.807 | 49.201 | 3,45 | 0,77970 | 0,55941 | 0,42758 | 0,26819 | 0,17622 | 0,29845 |
| `undersample` | 0 | 196.806 | 47.664 | 49.202 | 1,37 | 0,78119 | 0,56237 | 0,42209 | 0,26652 | 0,11874 | 0,19877 |
| `undersample` | 1 | 196.806 | 47.664 | 49.202 | 1,51 | 0,77561 | 0,55121 | 0,41505 | 0,25222 | 0,11954 | 0,19846 |
| `undersample` | 2 | 196.806 | 47.664 | 49.202 | 1,61 | 0,77574 | 0,55149 | 0,41256 | 0,26623 | 0,11787 | 0,19588 |
| `undersample` | 3 | 196.807 | 47.664 | 49.201 | 1,48 | 0,77687 | 0,55373 | 0,42200 | 0,27336 | 0,11879 | 0,19853 |
| `undersample` | 4 | 196.807 | 47.664 | 49.201 | 1,47 | 0,77824 | 0,55648 | 0,42656 | 0,26864 | 0,11713 | 0,19602 |
| `smotenc` | 0 | 196.806 | 271.377 | 49.202 | 49,90 | 0,68371 | 0,36742 | 0,27285 | 0,14650 | 0,10804 | 0,13151 |
| `smotenc` | 1 | 196.806 | 271.377 | 49.202 | 46,15 | 0,68886 | 0,37772 | 0,27569 | 0,15724 | 0,09960 | 0,12155 |
| `smotenc` | 2 | 196.806 | 271.377 | 49.202 | 44,65 | 0,68074 | 0,36149 | 0,27009 | 0,14447 | 0,10889 | 0,13559 |
| `smotenc` | 3 | 196.807 | 271.378 | 49.201 | 43,30 | 0,70782 | 0,41564 | 0,30990 | 0,16820 | 0,09258 | 0,10071 |
| `smotenc` | 4 | 196.807 | 271.378 | 49.201 | 44,21 | 0,69485 | 0,38969 | 0,28940 | 0,16334 | 0,10830 | 0,13936 |

## 5. Tổng hợp và so với baseline

| Variant | ROC-AUC | ΔAUC so với `none` | Gini | KS | PR-AUC | Brier | ECE (15 bins) | Precision @0,5 | Recall @0,5 | F1 @0,5 |
|---|---|---:|---|---|---|---|---|---:|---:|---:|
| `none` (baseline) | **0,77836 ± 0,00178** | — | 0,55671 | 0,41866 | 0,26805 ± 0,00644 | **0,06654 ± 0,00027** | **0,00434** | 0,5433 | 0,0380 | 0,0710 |
| `class_weight` | **0,77920 ± 0,00120** | +0,00084 | 0,55840 | 0,42003 | 0,26893 ± 0,00710 | 0,17736 ± 0,00086 | 0,29946 | 0,1853 | 0,6802 | 0,2912 |
| `undersample` | 0,77753 ± 0,00206 | −0,00083 | 0,55506 | 0,41965 | 0,26539 ± 0,00707 | 0,11842 ± 0,00083 | 0,19753 | 0,2491 | 0,4707 | 0,3258 |
| `smotenc` | 0,69120 ± 0,00960 | −0,08716 | 0,38239 | 0,28358 | 0,15595 ± 0,00924 | 0,10348 ± 0,00644 | 0,12574 | 0,1939 | 0,2039 | 0,1969 |

`none` tái lập **chính xác** E2 `plus_installments`: cùng AUC mean 0,77836 và cùng 5 giá trị AUC theo fold.

Chênh lệch AUC theo từng fold so với `none`:

| Variant | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Số fold dương |
|---|---:|---:|---:|---:|---:|---:|
| `class_weight` | +0,00017 | +0,00231 | +0,00056 | −0,00039 | +0,00158 | 4/5 |
| `undersample` | +0,00008 | +0,00010 | −0,00248 | −0,00194 | +0,00011 | 3/5 |
| `smotenc` | −0,09740 | −0,08665 | −0,09748 | −0,07099 | −0,08328 | 0/5 |

Hai điểm cần đọc đúng:

- **`class_weight` hơn `none` +0,00084 AUC nhưng nằm trong nhiễu fold.** Std của `none` là 0,00178, tức chênh lệch chỉ bằng ~0,5 độ lệch chuẩn, và fold 3 còn âm. PR-AUC gần như không đổi (0,26805 → 0,26893). Đây là **hòa về năng lực xếp hạng**, không phải cải thiện có ý nghĩa.
- **`smotenc` hỏng có hệ thống**, không phải nhiễu: cả 5 fold giảm 0,071–0,098, và độ lệch giữa fold tăng từ 0,00178 lên 0,00960.

## 6. Xếp hạng so với thang xác suất

Ba variant đầu giữ nguyên thứ tự dự đoán, chỉ đổi **thang xác suất**:

| Variant | PR-AUC | KS | p trung bình OOF | Trung vị p | Khoảng p |
|---|---|---:|---:|---:|---|
| `none` | 0,26805 | 0,41866 | 0,0804 | 0,0479 | 0,0039 – 0,9158 |
| `class_weight` | 0,26893 | 0,42003 | 0,3802 | 0,3402 | 0,0121 – 0,9531 |
| `undersample` | 0,26539 | 0,41965 | 0,2783 | 0,2206 | 0,0157 – 0,9710 |
| `smotenc` | 0,15595 | 0,28358 | 0,2065 | 0,1466 | 0,0071 – 0,9685 |

Tỷ lệ positive thực là 0,08073. Chỉ `none` cho xác suất trung bình khớp (0,0804); `class_weight` đẩy lên 0,3802, `undersample` 0,2783, `smotenc` 0,2065. Brier và ECE tăng mạnh ở ba variant sau vì **thang** lệch, không phải vì xếp hạng kém hơn.

## 7. Precision / recall theo ngưỡng

| Variant | Ngưỡng | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| `none` | 0,1 | 0,2028 | 0,6160 | 0,3052 |
| `none` | 0,2 | 0,3055 | 0,3340 | 0,3191 |
| `none` | 0,3 | 0,3964 | 0,1779 | 0,2456 |
| `none` | 0,4 | 0,4695 | 0,0873 | 0,1472 |
| `none` | 0,5 | 0,5433 | 0,0380 | 0,0710 |
| `class_weight` | 0,1 | 0,0859 | 0,9938 | 0,1582 |
| `class_weight` | 0,2 | 0,1035 | 0,9528 | 0,1868 |
| `class_weight` | 0,3 | 0,1263 | 0,8800 | 0,2209 |
| `class_weight` | 0,4 | 0,1534 | 0,7905 | 0,2569 |
| `class_weight` | 0,5 | 0,1853 | 0,6802 | 0,2912 |
| `undersample` | 0,1 | 0,0946 | 0,9751 | 0,1724 |
| `undersample` | 0,2 | 0,1290 | 0,8693 | 0,2246 |
| `undersample` | 0,3 | 0,1655 | 0,7450 | 0,2709 |
| `undersample` | 0,4 | 0,2034 | 0,6083 | 0,3048 |
| `undersample` | 0,5 | 0,2491 | 0,4707 | 0,3258 |
| `smotenc` | 0,1 | 0,1083 | 0,8584 | 0,1922 |
| `smotenc` | 0,2 | 0,1347 | 0,6367 | 0,2220 |
| `smotenc` | 0,3 | 0,1565 | 0,4580 | 0,2326 |
| `smotenc` | 0,4 | 0,1753 | 0,3156 | 0,2241 |
| `smotenc` | 0,5 | 0,1939 | 0,2039 | 0,1969 |

Với base rate 8%, ngưỡng 0,5 không dùng để so sánh giữa các variant vì mỗi variant có thang xác suất khác nhau. F1 tốt nhất của từng variant: `undersample` 0,3258 (@0,5), `none` 0,3191 (@0,2), `class_weight` 0,2912 (@0,5), `smotenc` 0,2326 (@0,3). Recall cao của `class_weight` tại 0,5 (0,6802) chỉ là hệ quả của việc dời ngưỡng, không phải thông tin mới: `none` tại 0,1 đạt recall 0,6160 với precision 0,2028, cao hơn precision 0,1853 của `class_weight` tại 0,5.

## 8. Calibration

| Variant | Brier | Baseline Brier (0,07421) | ECE 15 bins | p trung bình |
|---|---:|---:|---:|---:|
| `none` | **0,06654** | tốt hơn | **0,00434** | 0,0804 |
| `class_weight` | 0,17736 | xấu hơn | 0,29946 | 0,3802 |
| `undersample` | 0,11842 | xấu hơn | 0,19753 | 0,2783 |
| `smotenc` | 0,10348 | xấu hơn | 0,12574 | 0,2065 |

- Chỉ `none` calibrated sẵn: ECE 0,00434, p trung bình 0,0804 so với 0,08073.
- `class_weight` và `undersample` làm lệch thang nghiêm trọng (ECE 0,2995 và 0,1975): xác suất không còn là tần suất mặc định, không được báo cáo trực tiếp.
- Hệ quả bắt buộc: **E6 phải calibrate lại** (Platt hoặc isotonic, fit trên OOF) trước khi báo cáo bất kỳ xác suất, ngưỡng vận hành hay expected loss nào. Nếu E6 không kéo Brier/ECE về mức của `none`, phương án dự phòng là quay lại `none`.

## 9. Feature importance (gain, trung bình 5 fold)

Top 15 của `class_weight`:

| # | Feature | Importance |
|---|---|---:|
| 1 | `APP_EXT_SOURCE_MEAN` | 277.273,9 |
| 2 | `ORGANIZATION_TYPE` | 79.170,0 |
| 3 | `APP_CREDIT_ANNUITY_RATIO` | 24.555,7 |
| 4 | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | 17.712,5 |
| 5 | `PREV_CREDIT_APPLICATION_RATIO_MEAN` | 17.542,8 |
| 6 | `APP_CREDIT_GOODS_RATIO` | 15.787,3 |
| 7 | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` | 15.786,6 |
| 8 | `INS_OBS_PAYMENT_SUM` | 12.200,5 |
| 9 | `POS_LATEST_FUTURE_INSTALLMENTS_SUM` | 11.554,5 |
| 10 | `EXT_SOURCE_3` | 10.841,2 |
| 11 | `EXT_SOURCE_1` | 10.654,7 |
| 12 | `OCCUPATION_TYPE` | 10.537,2 |
| 13 | `CC_LATEST_BALANCE_SUM` | 10.079,2 |
| 14 | `PREV_REFUSAL_RATE` | 9.911,6 |
| 15 | `AMT_ANNUITY` | 9.243,0 |

Thứ hạng giữ nguyên cấu trúc của E2: `APP_EXT_SOURCE_MEAN` dẫn đầu, sau đó là nhóm thu nhập/khoản vay, rồi mỗi bảng phụ đều có đại diện. Giá trị gain tuyệt đối lớn hơn E2 (277.274 so với 47.096) vì trọng số lớp làm tăng độ lớn gradient, nên **chỉ so sánh importance trong cùng variant**, không so giá trị gain giữa các variant.

Số feature có importance bằng 0: `none` 25/180, `class_weight` 25/180, `undersample` 24/180, `smotenc` 21/180 (E2: 25/180). Ứng viên cho bước chọn feature sau này, chưa loại trong E3.

Top 5 của `smotenc` cho thấy bức tranh khác hẳn:

| # | Feature | Importance |
|---|---|---:|
| 1 | `APP_EXT_SOURCE_MEAN` | 195.999,7 |
| 2 | `INS_N_MULTI_RECORD_PERIODS` | 131.829,0 |
| 3 | `OBS_30_CNT_SOCIAL_CIRCLE` | 120.155,8 |
| 4 | `PREV_N_RECENT_365` | 90.245,3 |
| 5 | `POS_N_LOANS` | 72.126,7 |

Importance chuyển một phần sang các cột đếm/cờ (`INS_N_MULTI_RECORD_PERIODS`, `OBS_30_CNT_SOCIAL_CIRCLE`, `PREV_N_RECENT_365`, `POS_N_LOANS`, `FLAG_EMP_PHONE`, `NONLIVINGAREA_MODE`): chúng chen vào top 2–5 với gain rất lớn, trong khi top 5 của `none` chỉ toàn `EXT_SOURCE`, tỷ lệ và feature bureau.

## 10. Vì sao SMOTENC thất bại

Kết quả: −0,08716 AUC (0,77836 → 0,69120), KS 0,41866 → 0,28358, PR-AUC 0,26805 → 0,15595, và cả 5 fold đều giảm. Chi phí 45,64 s fit/fold so với 3,31 s, peak RSS 5.821 MB so với 2.559 MB, train trên 271.377 dòng.

Hai nguyên nhân, và thiết kế hiện tại **gộp cả hai vào một variant** nên không tách được:

1. **Phải điền missing trước khi resample.** 21,9% ô của ma trận training (9.694.754 ô, 109/180 cột) không có giá trị; SMOTENC raise `Input contains NaN`, nên variant điền median/mode **trong từng fold-training**. Điểm tổng hợp vì thế được nội suy trong một không gian đã bị san phẳng một phần năm số ô, tạo ra những dòng không tồn tại trong quần thể. Đây là lý do chính người chạy nêu, và dữ liệu ủng hộ.
2. **kNN trong 180 chiều.** SMOTENC chọn láng giềng theo khoảng cách trộn numeric và one-hot categorical; với 180 chiều, khoảng cách giữa các điểm gần nhau về mặt thống kê, nên điểm nội suy dễ rơi vào vùng không có thật. Càng nhiều chiều, láng giềng càng kém tin cậy.

Bằng chứng phụ: importance chuyển một phần sang cột đếm/cờ, tức mô hình tận dụng được cấu trúc "dòng tổng hợp" chứ không chỉ cấu trúc rủi ro. Một điểm mô hình có thể khai thác là các dòng tổng hợp trùng lặp/lân cận nhau, tạo vùng mà cây tách được dễ dàng.

Đối chứng duy nhất tách được *riêng* việc đổi cân bằng lớp: `undersample` cũng đổi tỷ lệ lớp nhưng không tạo dòng giả và không cần điền missing, kết quả −0,00083 AUC. Vậy **bản thân việc cân bằng lớp là vô hại**; thiệt hại đến từ dữ liệu tổng hợp + missing phải điền. Muốn tách riêng hai nguyên nhân cần thêm một variant điền missing cho **mọi** variant (đã cân nhắc và loại ở Phase 5 vì nó đổi định nghĩa baseline E1/E2), hoặc chạy SMOTENC trên feature set nhỏ hơn (`main`, 120 cột) để đo phần do số chiều. Báo cáo cuối phải ghi đây là **confound**, không được quy kết một nguyên nhân duy nhất.

## 11. Kiểm tra tính hợp lệ

- `none` tái lập E2 `plus_installments` chính xác: AUC mean 0,77836, cùng 5 giá trị AUC theo fold.
- OOF mỗi variant: 246.008 dòng, `SK_ID_CURR` unique, đúng 1 dự đoán/dòng, positive rate 0,080729 khớp development set.
- Không có ID hold-out trong OOF; `val.parquet` không được load; `n_holdout_rows = 61.503` trong cả 4 run JSON.
- Validation không bị resample: `n_valid` luôn 49.202 / 49.201, chỉ `n_train_fit` thay đổi (47.664 ở `undersample`, 271.377 ở `smotenc`). Có test riêng: `tests/test_cv_leakage.py::test_resampling_never_touches_validation_rows`.
- Xác suất không bão hòa: `none` [0,0039; 0,9158], `class_weight` [0,0121; 0,9531], `undersample` [0,0157; 0,9710], `smotenc` [0,0071; 0,9685].
- Model seed 42, `data_version = datasets_v1`, `git_commit = 8ad2716` cho cả 4 run; mỗi variant có `config_digest` riêng (`149e3d4b`, `8e50f183`, `9e3dbffc`, `3de5d091`).
- Output smoke `*_sample500*` đã xoá; bảng E3 full nằm ở `reports/tables/E3_imbalance_comparison.csv`.

## 12. Chi phí chạy

| Variant | Tổng thời gian (s) | Fit (s/fold) | Predict (s/fold) | Peak RSS (MB) | Dòng fit/fold |
|---|---:|---:|---:|---:|---:|
| `none` | 17,9 | 3,31 | 0,17 | 2.559 | 196.806 |
| `class_weight` | 18,6 | 3,45 | 0,18 | 2.753 | 196.806 |
| `undersample` | 8,8 | 1,49 | 0,17 | 2.753 | 47.664 |
| `smotenc` | 229,6 | 45,64 | 0,17 | 5.821 | 271.377 |
| **Tổng** | **274,9** | — | — | **5.821** | — |

`smotenc` chiếm 83% tổng thời gian và gấp 2,3 lần bộ nhớ của `none`. Vẫn dưới ngưỡng 16 GB, nhưng đây là variant đắt nhất mà lại cho kết quả tệ nhất.

## 13. Artifacts

| Loại | Đường dẫn |
|---|---|
| Bảng so sánh E3 | `reports/tables/E3_imbalance_comparison.csv` |
| Fold metrics | `reports/tables/E3_26-11-17_exp01/E3_<variant>_cv_folds_seed42.csv` |
| Summary | `reports/tables/E3_26-11-17_exp01/E3_<variant>_cv_summary_seed42.csv` |
| Importance | `reports/tables/E3_26-11-17_exp01/E3_<variant>_feature_importance_seed42.csv` |
| OOF | `reports/predictions/E3_26-11-17_exp01/E3_<variant>_lgbm_seed42_oof.parquet` |
| Run JSON | `reports/runs/E3_26-11-17_exp01/E3_<variant>_lgbm_seed42.json` |
| MLflow | experiment `E3_imbalance` |

`<variant>` nhận một trong `none`, `class_weight`, `undersample`, `smotenc`.

Run MLflow của lần chạy full (development 246.008 dòng):

| Variant | Run ID |
|---|---|
| `none` | `bdd1e90aa9b346aa9573ec15b255c20c` |
| `class_weight` | `2124a25677c6478cb7f2c534a1eff0ef` |
| `undersample` | `99644de79e6543f2a4d16b879a35afd6` |
| `smotenc` | `d442503c3ec94b449b650d7cb63af44c` |

## 14. Kết luận và bước tiếp

E3 trả lời RQ3 trên `plus_installments` (180 predictor), LightGBM mặc định, seed 42:

- **`class_weight` là variant được chọn** cho E4–E7: AUC 0,77920, hơn `none` **+0,00084**. Nhưng phải đọc kèm điều kiện: mức chênh chỉ bằng ~0,5 std fold, 4/5 fold dương và 1 fold âm, PR-AUC gần như không đổi — đây là hòa về xếp hạng, không phải cải thiện có ý nghĩa thống kê. Cái giá là calibration: Brier 0,06654 → 0,17736, ECE 0,00434 → 0,29946.
- **E6 trở thành bắt buộc**, không còn là bước tùy chọn: xác suất thô của `class_weight` không dùng được (p trung bình 0,3802 so với base rate 0,08073). Trước khi E6 xong, chỉ được so sánh bằng metric xếp hạng (AUC/Gini/KS/PR-AUC).
- **`undersample` bị loại:** cùng hòa về AUC (−0,00083) nhưng chỉ giữ 24% dòng training (47.664/196.806). Điểm F1 cao nhất bảng (0,3258 @0,5) đến từ variant này, nhưng đó là hệ quả của thang xác suất, không phải năng lực mô hình.
- **`smotenc` bị loại:** −0,08716 AUC trên cả 5 fold, 13 lần thời gian fit, 2,3 lần bộ nhớ. Nguyên nhân là kết hợp của missing phải điền (21,9% ma trận) và kNN trong 180 chiều; thiết kế hiện tại không tách được hai phần này.
- **Bài học cho operating point:** không nên bán `class_weight` như "tăng recall". Recall 0,6802 tại ngưỡng 0,5 của nó thấp hơn về mặt F1 so với `none` tại 0,2 (0,3191 so với 0,2912) và `undersample` tại 0,5 (0,3258). Ngưỡng vận hành phải chọn trên PR curve sau khi calibrate, không cố định 0,5.

Bước tiếp:

- **E4 (RQ3, phần tuning):** Optuna TPE trên `plus_installments`, giữ `imbalance.strategy: class_weight`, cùng fold và seed; tuning và cân bằng lớp đều fit trong từng fold. Kỳ vọng AUC vượt 0,7784 (mốc tốt nhất hiện tại của E2/E3) mới đáng giữ.
- **E5/E6:** so sánh 3 GBDT trên tập tuned, sau đó calibrate lại trên OOF; nếu Brier/ECE sau calibrate không về mức `none` (0,06654 / 0,00434), ghi nhận và cân nhắc quay lại `none`.

Hạn chế cần nêu trong báo cáo cuối: chỉ 1 model seed; chưa có kiểm định ý nghĩa (DeLong ở Phase 7); kết luận `class_weight` dựa trên chênh lệch trong nhiễu fold; variant `smotenc` bị confound bởi bước điền missing; split random không theo thời gian; và không có feature cửa sổ thời gian (`WIN_`).
