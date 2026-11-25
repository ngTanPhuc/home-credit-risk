# E6 — Calibration và hold-out cuối cùng

Báo cáo Phase 8 + Phase 9: hiệu chỉnh xác suất cho các arm của E5 (RQ2), chốt
model cuối, và lần đọc hold-out duy nhất của dự án.

> Trạng thái E7 (Phase 10): phần `shap_analysis.py` **chưa chạy**; mới có bảng
> đồng thuận feature giữa 3 model (`reports/tables/E7_feature_agreement.csv`).
> Báo cáo này không bao gồm SHAP.

## 1. Mục tiêu

1. Xác suất của các model E5 có dùng được không. E3 đã chốt `class_weight`, và biến
   thể đó phá calibration: Brier 0,06654 → 0,17736, ECE 0,00434 → 0,29946 trên
   LightGBM defaults. E6 phải trả lại thang xác suất trước khi báo cáo bất kỳ con số
   nào phụ thuộc ngưỡng, expected loss hay pricing.
2. Chốt cặp model + calibrator, rồi đo nó **một lần** trên hold-out 61.503 dòng.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E6_calibration.yaml`, `configs/experiments/holdout_final.yaml` |
| Driver | `src/evaluation/calibration_eval.py`, `src/evaluation/holdout.py` |
| Run | `E6_26-11-24_exp01`, `holdout_26-11-24` |
| Dữ liệu vào E6 | OOF của E5 (`reports/predictions/E5_26-11-24_exp01/`), **không refit model** |
| Arm | LightGBM, XGBoost, CatBoost, `weighted_average`, `stacking` |
| Calibrator | `none`, `platt` (logistic trên logit, C=1e6), `isotonic` (`y_min=0`, `y_max=1`, clip) |
| Cách đo | Cross-fit: calibrator của fold *k* fit trên OOF của 4 fold còn lại (196.806 dòng), áp lên fold *k* |
| Metric | Brier, ECE 15 bins, log loss; AUC/KS/PR-AUC chỉ để kiểm tra bất biến |
| Cấu hình hold-out | 246.008 dòng development fit, 61.503 dòng hold-out (4.965 positive, 8,07%) |
| Model seed | 42 (cặp cuối), 42/43/44 (E6) |
| Chốt cuối | `stacking` + `isotonic` |

## 3. E6 — Brier/ECE trước và sau (mean qua 3 seed)

| Arm | `none` Brier | `none` ECE | `platt` Brier | `platt` ECE | `isotonic` Brier | `isotonic` ECE |
|---|---|---|---|---|---|---|
| LightGBM | 0,166012 | 0,280822 | **0,066154** | **0,003212** | 0,066160 | 0,003313 |
| XGBoost | 0,172520 | 0,291703 | **0,065967** | 0,003588 | 0,065992 | 0,003467 |
| CatBoost | 0,175286 | 0,297264 | **0,066053** | **0,003200** | 0,066084 | 0,003272 |
| `weighted_average` | 0,166012 | 0,280822 | 0,066154 | 0,003212 | 0,066160 | 0,003313 |
| `stacking` | 0,066070 | 0,005605 | 0,066071 | 0,005652 | **0,065950** | **0,003315** |

Log loss của cặp chốt: 0,236659 (hold-out), 0,237094 (OOF cross-fit).

### 3.1 Calibration xoá sạch thiệt hại của `class_weight`

Ba model nền từ Brier 0,166-0,175 / ECE 0,28-0,30 xuống 0,0660-0,0662 /
0,0032-0,0036, tức là ngang hoặc tốt hơn mốc E1 `none` (0,06810 / 0,00462) trong
khi AUC không đổi (0,784132 / 0,787176 / 0,786154). Kết luận: `class_weight` chỉ
làm hỏng *thang*, không làm hỏng *thứ hạng*, và một map đơn điệu sửa được toàn bộ.

### 3.2 Platt và isotonic tương đương trên model nền

Chênh lệch Brier ≤ 0,00003 và ECE ≤ 0,00015 trên cả ba model. Không có lý do chọn
cái này thay cái kia ở nhóm này.

### 3.3 Platt là no-op trên `stacking`

Stacking `none` 0,066070 / 0,005605 → `platt` 0,066071 / 0,005652: chênh +0,000001
Brier và +0,000047 ECE. Nguyên nhân: meta-learner đã là một logistic regression,
nên nó tự học lại thang xác suất; Platt chỉ fit lại gần đúng hàm identity. Đây là
mặt còn lại của phát hiện E5 (stacking đã sửa được phần lớn thiệt hại calibration
mà không cần E6). Vì vậy với `stacking`, lựa chọn thật chỉ là `none` hay `isotonic`.

### 3.4 `isotonic` thắng `none` trên stack, nhưng đổi độ mịn của score

`isotonic` cải thiện **đồng thời** trên cả ba seed: Brier −0,000126 / −0,000114 /
−0,000120 và ECE −0,00241 / −0,00236 / −0,00210 (seed 42/43/44). Đây không phải
nhiễu, nhưng chỉ là 0,18 % tương đối của Brier.

Cái giá: isotonic là hàm bậc thang, chỉ có ~100-130 giá trị phân biệt mỗi fold
(97 nếu fit một lần trên toàn bộ development), nên nó tạo ra tie. AUC từng fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `none` | 0,789780 | 0,785788 | 0,787035 | 0,788069 | 0,787964 |
| `isotonic` | 0,789703 | 0,785390 | 0,787125 | 0,787855 | 0,787813 |
| `isotonic` + tie-break bằng score thô | 0,789780 | 0,785788 | 0,787035 | 0,788069 | 0,787964 |

Cộng `1e-9 × score thô` làm tie-break thì AUC **khôi phục chính xác** AUC chưa
calibrate ở cả 5 fold. Nói cách khác: mất mát là độ mịn của score, không phải
thông tin xếp hạng. Trung bình đọc trực tiếp trên vector đã calibrate: AUC 0,787638
so với 0,787826, KS 0,4339 so với 0,4352.

Lưu ý đo lường: vector cross-fit trộn 5 map khác nhau nên AUC gộp trên nó không
phải score của một model triển khai. Fit một map isotonic duy nhất trên toàn bộ
OOF rồi áp lên chính các dòng đó (đúng hành vi của Phase 9) cho 97 mức, Brier
0,065857 và AUC tie-broken 0,787717 — bằng đúng AUC thô.

## 4. Chốt model cuối

**`stacking` + `isotonic`** (seed 42), ghi trong `configs/experiments/E6_calibration.yaml`
ở block `final`, và được `holdout_final.yaml` kế thừa.

Lý do:

1. Stacking là arm tốt nhất của 5 arm trên Brier, log loss và ECE sau calibration.
2. Nó hơn XGBoost + Platt về AUC +0,00046 (DeLong p ≤ 0,002 ở E5) trong khi chênh
   Brier chỉ 0,000017 và log loss 0,000014 — hoà trên metric xác suất, thắng trên
   metric xếp hạng.
3. Calibration là mục đích của phase này; báo cáo một xác suất chưa calibrate trong
   khi có map monotone tốt hơn là bỏ dở yêu cầu của E3.

Đánh đổi đã chấp nhận và phải ghi trong báo cáo: score giao cho người dùng có ~100
mức, nên AUC/KS đọc trực tiếp thấp hơn stack thô ~0,0002 / ~0,0012;
`B4_delong_pairs.csv` vì thế kiểm định trên xác suất thô của từng arm.

## 5. Hold-out — lần đọc duy nhất (`holdout_26-11-24`)

61.503 dòng, 4.965 positive (8,07 %). Ba model fit trên toàn bộ 246.008 dòng
development với `final_params` của E4; meta-learner và calibrator fit trên OOF
development, không dùng nhãn hold-out.

### 5.1 B3 — metric trên hold-out

| Arm | Calibrator | ROC-AUC | Gini | KS | PR-AUC | Brier | ECE | Log loss |
|---|---|---|---|---|---|---|---|---|
| LightGBM | none | 0,784903 | 0,569805 | 0,432110 | 0,280291 | 0,169861 | 0,287150 | 0,511183 |
| LightGBM | isotonic | 0,784894 | 0,569787 | 0,430464 | 0,274427 | 0,065987 | 0,003312 | 0,237593 |
| XGBoost | none | 0,788096 | 0,576193 | 0,435458 | 0,286777 | 0,175700 | 0,297146 | 0,526366 |
| XGBoost | isotonic | 0,787942 | 0,575885 | 0,434026 | 0,281087 | 0,065703 | 0,003288 | 0,236679 |
| CatBoost | none | 0,786926 | 0,573852 | 0,431994 | 0,283525 | 0,178407 | 0,302286 | 0,533315 |
| CatBoost | isotonic | 0,786794 | 0,573588 | 0,431520 | 0,278297 | 0,065786 | 0,003354 | 0,237004 |
| `stacking` | none | 0,788220 | 0,576440 | 0,435916 | 0,285968 | 0,065895 | 0,005675 | 0,236953 |
| **`stacking`** | **isotonic (final)** | **0,788079** | **0,576157** | **0,433446** | **0,280041** | **0,065718** | **0,003827** | **0,236659** |
| `weighted_average` | none / isotonic | 0,784903 | 0,569805 | 0,432110 | 0,280291 | 0,169861 → 0,065987 | 0,287150 → 0,003312 | 0,511183 → 0,237593 |

`weighted_average` lại trùng LightGBM (hệ số 1,0), đúng như E5.

### 5.2 B4 — DeLong trên hold-out

| A | B | ΔAUC | p | Ý nghĩa 0,05 |
|---|---|---|---|---|
| `stacking` | XGBoost | +0,000124 | 0,7307 | không |
| `stacking` | CatBoost | +0,001294 | 0,0040 | có |
| `stacking` | LightGBM | +0,003317 | 9,96e-13 | có |
| XGBoost | CatBoost | +0,001170 | 0,0790 | không |
| XGBoost | LightGBM | +0,003194 | 6,06e-06 | có |
| CatBoost | LightGBM | +0,002023 | 0,0149 | có |

**Điểm quan trọng nhất của hold-out:** khoảng cách stacking − XGBoost trên CV là
+0,00065 (p ≤ 0,002 ở E5) **không tái lập**; trên hold-out còn +0,000124 với
p = 0,73. Ensemble không thực sự hơn model đơn tốt nhất, và E5 đã cảnh báo đúng
hướng (ba model gần trùng nhau, trần của feature set này thấp). Stacking vẫn hơn
LightGBM và CatBoost một cách rõ ràng.

### 5.3 Điểm vận hành (cặp chốt)

| Ngưỡng | Precision | Recall | F1 |
|---|---|---|---|
| 0,1 | 0,2008 | 0,6574 | 0,3076 |
| 0,2 | 0,2991 | 0,3932 | 0,3397 |
| 0,3 | 0,3888 | 0,2159 | 0,2776 |
| 0,4 | 0,4955 | 0,1001 | 0,1666 |
| 0,5 | 0,5750 | 0,0556 | 0,1014 |

Ngưỡng 0,5 gần như không dùng được ở base rate 8 %; điểm có F1 cao nhất là 0,2
(bắt 39,3 % default với precision 29,9 %). Báo cáo nên nêu 0,2 chứ không phải 0,5.

### 5.4 CV so với hold-out

| | CV (mean 3 seed) | Hold-out | Chênh |
|---|---|---|---|
| `stacking` AUC | 0,787826 | 0,788220 | +0,00039 |
| `stacking` Brier | 0,066070 | 0,065895 | −0,00018 |
| `stacking` ECE | 0,005605 | 0,005675 | +0,00007 |
| cặp chốt AUC | 0,787638 | 0,788079 | +0,00044 |
| cặp chốt Brier | 0,065950 | 0,065718 | −0,00023 |
| cặp chốt ECE | 0,003315 | 0,003827 | +0,00051 |

Không có dấu hiệu overfit vào CV: hold-out tốt hơn một chút về AUC/Brier, kém hơn
0,0005 về ECE. CV là ước lượng hơi bi quan, không phải lạc quan.

## 6. Chi phí

| Bước | Thời gian | Ghi chú |
|---|---|---|
| E6 (5 arm × 3 calibrator × 3 seed) | 86 s | Chỉ đọc OOF, không refit |
| Hold-out (3 model × ~2000 vòng + stack + calibrate + DeLong) | không ghi trong record | Ước tính 5-10 phút, < 6 GB; record hiện chưa lưu `elapsed_seconds` |

## 7. Artifacts

- `reports/tables/E6_calibration.csv`, `E6_calibration_folds.csv`
- `reports/figures/E6_reliability.png` (reliability 15 bins, `none` vs calibrator)
- `reports/runs/E6_26-11-24_exp01/E6_calibration.json`
- `reports/tables/B3_calibration.csv`, `reports/tables/B4_delong_pairs.csv`
- `reports/figures/holdout_reliability.png`
- `reports/runs/holdout_26-11-24/holdout_final.json` (choice, row counts, git commit, final metrics)

## 8. Kết luận

1. Calibration sửa được toàn bộ thiệt hại của `class_weight` mà không đổi AUC; Platt
   và isotonic tương đương trên model nền, và Platt vô dụng trên stack.
2. Cặp cuối là `stacking` + `isotonic`: 0,788079 AUC / 0,576157 Gini / 0,433446 KS /
   0,280041 PR-AUC / 0,065718 Brier / 0,003827 ECE / 0,236659 log loss trên hold-out.
3. Trên hold-out, stacking **không** hơn XGBoost một cách có ý nghĩa (p = 0,73);
   claim đúng là "stacking hơn LightGBM và CatBoost, và hoà với XGBoost". Không được
   viết rằng ensemble vượt model đơn tốt nhất.
4. Xác suất đã dùng được: ECE 0,0038, Brier thấp hơn baseline 0,07421. Ngưỡng nên
   chọn theo chi phí, không mặc định 0,5; 0,2 là điểm F1 tốt nhất trên hold-out.
5. RQ2 trả lời được: khác biệt giữa ba GBDT là nhỏ (0,0020 AUC giữa LightGBM và
   XGBoost, p = 6e-06 nhưng thực dụng là không đáng kể), ensemble chỉ hơn nhóm dưới,
   và giới hạn còn lại nằm ở feature set 180 cột, không ở model.
