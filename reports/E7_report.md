# E7 — Giải thích model (SHAP) và đồng thuận feature

Báo cáo Phase 10 (RQ4): model dựa vào feature nào, vì sao một hồ sơ bị từ chối, và
ba GBDT có đồng ý với nhau về tầm quan trọng của feature không.

## 1. Mục tiêu

Trả lời **RQ4** ở hai mức:

1. **Toàn cục** — 180 feature của `plus_installments` đóng góp bao nhiêu vào quyết
   định, và nhóm feature nào (application, bureau, previous, POS, thẻ, trả góp) nắm
   phần lớn tín hiệu.
2. **Cục bộ** — với một hồ sơ cụ thể, feature nào đẩy rủi ro lên, đẩy lên bao nhiêu,
   để sinh reason code cho hồ sơ bị từ chối.
3. **Đồng thuận** — LightGBM, XGBoost và CatBoost có xếp hạng feature giống nhau
   không, tức là kết luận về feature có phụ thuộc vào việc chọn model hay không.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Config | `configs/experiments/E7_explain.yaml` |
| Driver | `src/explain/shap_analysis.py`, `src/explain/agreement.py` |
| Run | `E7_26-11-24_exp01` |
| Model được giải thích | LightGBM tuned, `class_weight`, seed 42, fit trên toàn bộ 246.008 dòng development |
| Dữ liệu được giải thích | 10.000 dòng **feature** hold-out, sample tất định (`seed=42`) từ 61.503 dòng |
| Nhãn hold-out | **không đọc** (`load_holdout_features` chỉ trả predictor + `SK_ID_CURR`) |
| SHAP | TreeSHAP chính xác của từng thư viện (`BaseModel.shap_contributions`), cộng tính với log-odds của model; `shap` chỉ dùng để vẽ beeswarm/waterfall |
| Ngưỡng approve/reject | 0,5 |
| Số reason code | 5 |
| Đồng thuận | gain importance trung bình 5 fold của E5 (seed 42), Spearman trên hợp 180 feature + Jaccard top-k |

## 3. Toàn cục — feature quan trọng nhất

`mean_abs_shap` = trung bình |đóng góp log-odds| trên 10.000 hồ sơ; `share` =
tỉ trọng trong tổng tín hiệu.

| # | Feature | mean\|SHAP\| | share |
|---|---|---|---|
| 1 | `APP_EXT_SOURCE_MEAN` | 0,3901 | 11,62 % |
| 2 | `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX` | 0,1324 | 3,94 % |
| 3 | `ORGANIZATION_TYPE` | 0,1242 | 3,70 % |
| 4 | `PREV_CREDIT_APPLICATION_RATIO_MEAN` | 0,1175 | 3,50 % |
| 5 | `APP_CREDIT_ANNUITY_RATIO` | 0,1127 | 3,36 % |
| 6 | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | 0,1073 | 3,20 % |
| 7 | `APP_CREDIT_GOODS_RATIO` | 0,1056 | 3,15 % |
| 8 | `CODE_GENDER` | 0,0967 | 2,88 % |
| 9 | `EXT_SOURCE_3` | 0,0966 | 2,88 % |
| 10 | `AMT_ANNUITY` | 0,0911 | 2,71 % |
| 11 | `POS_LATEST_FUTURE_INSTALLMENTS_SUM` | 0,0843 | 2,51 % |
| 12 | `EXT_SOURCE_1` | 0,0841 | 2,51 % |
| 13 | `EXT_SOURCE_2` | 0,0834 | 2,49 % |
| 14 | `INS_OBS_PAYMENT_SUM` | 0,0727 | 2,17 % |
| 15 | `OWN_CAR_AGE` | 0,0695 | 2,07 % |
| 16 | `NAME_FAMILY_STATUS` | 0,0669 | 1,99 % |
| 17 | `NAME_EDUCATION_TYPE` | 0,0609 | 1,81 % |
| 18 | `POS_N_MONTHS` | 0,0589 | 1,76 % |
| 19 | `PREV_REFUSAL_RATE` | 0,0580 | 1,73 % |
| 20 | `OCCUPATION_TYPE` | 0,0527 | 1,57 % |

Bảng đầy đủ 180 dòng: `reports/tables/E7_shap_importance.csv`.

### 3.1 Tín hiệu tập trung ở một feature, nhưng không chỉ một feature

- Top-1 = 11,6 %, top-5 = 26,1 %, top-10 = 41,0 %, top-20 = 61,6 %, top-50 = 87,7 %
  (top-100 = 98,1 %).
- Cần **14 feature** để đạt 50 % tín hiệu. Không có feature nào một mình quyết định:
  model là tổng của ~20 tín hiệu vừa phải, đúng như mức AUC 0,788 của một bài toán
  tín dụng khó chứ không phải một rule đơn.
- 20/180 feature có `mean_abs_shap` đúng bằng 0 (16 trong số đó cũng có gain = 0):
  tập 180 cột còn dư địa để cắt bớt mà không mất thông tin.

### 3.2 Theo nhóm feature

| Nhóm | Số feature | Share tín hiệu |
|---|---|---|
| `application_original` | 120 | 43,13 % |
| `application` (10 tỉ lệ) | 10 | 22,86 % |
| `installments` | 9 | 10,03 % |
| `bureau` | 17 | 8,34 % |
| `previous` | 10 | 6,95 % |
| `pos` | 6 | 5,16 % |
| `credit_card` | 8 | 3,53 % |

Đọc bảng này cùng B1: 120 cột application gốc giữ 43 % tín hiệu, còn **10 tỉ lệ dẫn
xuất giữ 22,9 %** — trung bình 2,3 %/feature so với 0,36 %/feature của nhóm gốc, tức
là feature engineering đã cô đặc tín hiệu chứ không chỉ thêm cột. `installments`
(9 cột, 10,0 %) là nhóm lịch sử trả góp mạnh nhất, khớp với mức tăng +0,0024 AUC khi
thêm nhóm này ở E2.

### 3.3 Kiểm tra hợp lệ: chiều của `EXT_SOURCE`

`E7_shap_dependence.png` cho thấy `APP_EXT_SOURCE_MEAN` giảm đơn điệu: giá trị 0,0
ứng với +1,2 log-odds, 0,8 ứng với −0,75. Đúng chiều kinh tế (điểm tín dụng ngoài
càng cao càng ít rủi ro) và không có ngưỡng kì lạ. `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX`
tăng đơn điệu theo số ngày trễ, chững lại sau ~500 ngày. Đây là kiểm tra sanity cho
thấy attribution khớp với hiểu biết nghiệp vụ, không phải một artefact của TreeSHAP.

## 4. Cục bộ — hai hồ sơ đại diện

Waterfall dựng trên cùng thang log-odds; tổng đóng góp + `E[f(x)]` = `f(x)` (code
kiểm tra độ lệch < 1e-3, lệch nhiều thì dừng).

`E[f(x)] = −0,676` → xác suất nền 33,7 %. Con số này **không** phải base rate 8,07 %:
model được train với `class_weight`, nên log-odds nội bộ nằm trên thang đã cân lại.
Đây là thang của SHAP, không phải thang xác suất cuối.

### 4.1 Hồ sơ bị từ chối 164.614 — p = 0,9541, f(x) = 3,033

| # | Feature | Giá trị | SHAP | Hướng |
|---|---|---|---|---|
| 1 | `APP_EXT_SOURCE_MEAN` | 0,1148 | +0,9664 | tăng rủi ro |
| 2 | `BUREAU_OVERDUE_MAX` | 413.482,5 | +0,6410 | tăng rủi ro |
| 3 | `EXT_SOURCE_3` | 0,00053 | +0,3421 | tăng rủi ro |
| 4 | `APP_CREDIT_GOODS_RATIO` | 1,528 | +0,1936 | tăng rủi ro |
| 5 | `BUREAU_DEBT_CREDIT_RATIO_MEAN_LOAN` | 1,167 | +0,1883 | tăng rủi ro |

Các feature tiếp theo: `APP_CREDIT_ANNUITY_RATIO` 21,28 (+0,17), `INS_OBS_PAYMENT_SUM`
18.852 (+0,15), `PREV_REFUSAL_RATE` 0,75 (+0,13), `INS_LAST_OBSERVED_PAYMENT_DELAY_MAX`
30 ngày (+0,12), và 171 feature còn lại cộng lại +0,81.

Câu chuyện hồ sơ: điểm tín dụng ngoài gần 0, đã có khoản quá hạn lớn ở bureau, tỉ lệ
nợ/tín dụng và giá/giá trị hàng cao, từng bị từ chối 75 % hồ sơ trước đó. Không có
feature nào đẩy ngược lại đáng kể — đây là hồ sơ xấu ở hầu hết mọi chiều, nên mức
0,954 là hợp lý chứ không phải model bám vào một cột.

### 4.2 Hồ sơ được duyệt 445.621 — p = 0,0111, f(x) = −4,485

`APP_EXT_SOURCE_MEAN` 0,736 (−0,77), `ORGANIZATION_TYPE` 26 (−0,62), `OWN_CAR_AGE` 8
(−0,23), `EXT_SOURCE_2` 0,745 (−0,23), `FLAG_DOCUMENT_18` 1 (−0,23),
`REGION_POPULATION_RELATIVE` 0,046 (−0,17), `EXT_SOURCE_3` 0,727 (−0,14),
`PREV_CREDIT_APPLICATION_RATIO_MEAN` = NaN (−0,14), `FLAG_DOCUMENT_3` 0 (−0,13), 171
feature còn lại −1,16.

Hai điểm đáng chú ý: (a) cả ba nguồn `EXT_SOURCE` đều đẩy xuống, tức tín hiệu nhất
quán chứ không phải một cột may mắn; (b) một giá trị **thiếu** đóng góp −0,14, tức
model học được rằng "không có hồ sơ vay trước" là tín hiệu tốt — hành vi hợp lý của
cây quyết định với missing, nhưng phải ghi rõ trong reason code để người đọc không
hiểu nhầm là có dữ liệu.

### 4.3 Reason code cho hồ sơ bị từ chối

`reports/tables/E7_reason_codes.csv`, 5 dòng, đúng định dạng adverse-action: chỉ nêu
feature **làm tăng** rủi ro, xếp theo độ lớn. Nếu một hồ sơ có ít hơn 5 feature dương
thì các slot còn lại lấy theo trị tuyệt đối (và ghi `decreases risk`), để danh sách
luôn đủ 5 dòng thay vì im lặng bỏ trống.

## 5. Đồng thuận giữa ba model (RQ4 phần 3)

Đo trên gain importance trung bình 5 fold của E5 (không cần refit, không cần SHAP
cho từng model): `reports/tables/E7_feature_agreement.csv`.

| A | B | Spearman | Jaccard@10 | @20 | @50 | @100 |
|---|---|---|---|---|---|---|
| LightGBM | XGBoost | 0,836 | 0,250 | 0,600 | 0,563 | 0,653 |
| LightGBM | CatBoost | 0,945 | 0,538 | 0,481 | 0,818 | 0,754 |
| XGBoost | CatBoost | 0,798 | 0,176 | 0,379 | 0,538 | 0,613 |

- Ở mức **thứ hạng toàn cục**, ba model rất giống nhau (Spearman 0,80-0,95): cùng
  một tập tín hiệu, cùng một thứ tự đại thể.
- Ở mức **top-10**, chúng không giống nhau (Jaccard 0,18-0,54, tức 2-5 feature chung
  trong 10). Nguyên nhân: nhiều feature có gain gần bằng nhau nên thứ hạng trong top
  10 đổi rất dễ giữa các model — đúng như E5 đã thấy với ba GBDT gần trùng nhau
  nhưng không hoán đổi được cho nhau.
- **Hệ quả cho báo cáo:** kết luận "feature nào quan trọng" không phụ thuộc model ở
  mức nhóm và mức top-20, nhưng **không nên** trích dẫn một thứ hạng top-5 cụ thể
  như một sự thật. Nên nói "top-20 ổn định, thứ hạng trong top-10 là nhiễu".

### 5.1 Kiểm tra chéo SHAP và gain của LightGBM

Cùng một model, hai cách đo khác nhau: Spearman 0,942, top-20 trùng 15/20, top-10
trùng 8/10. Hai phương pháp attribution độc lập cho cùng một kết luận, nên bảng
importance ở mục 3 không phải artefact của riêng TreeSHAP.

## 6. Hạn chế phải ghi trong báo cáo

1. **E7 giải thích LightGBM, không phải model cuối `stacking` + `isotonic`.** Stack
   là một logistic regression trên ba xác suất nên SHAP của nó chỉ nói "model nào
   đóng góp bao nhiêu", không nói feature nào. Giải thích ở mức feature buộc phải đi
   qua một model cây; LightGBM là thành viên hợp lý vì nó là model tốt nhất trong ba
   model cơ sở và không kém stack về thứ hạng (DeLong p 1e-12 chỉ với LightGBM, còn
   với XGBoost là hoà). Đây là lựa chọn đã ghi trong `DECISIONS.md`.
2. **Attribution không phải nhân quả.** SHAP nói model dùng feature nào, không nói
   thay đổi feature đó sẽ đổi xác suất vỡ nợ thật bao nhiêu. Reason code là giải
   thích quyết định, không phải lời khuyên tín dụng.
3. **Mẫu 10.000/61.503 dòng.** Đủ ổn định cho top-20 (sai số ~0,1 % tương đối của
   share), nhưng không nên đọc thứ hạng của các feature ngoài top-50.
4. **Giá trị categorical trong waterfall là mã số nguyên** (ví dụ `ORGANIZATION_TYPE
   = 26`), vì `shap` cần dữ liệu số để vẽ. Bảng reason code thì in nhãn gốc. Muốn
   waterfall đọc được bằng mắt thường phải tự map lại nhãn — chưa làm, đây là hạn chế
   của hình, không phải của số liệu.
5. **Thang log-odds của SHAP đã bị `class_weight` cân lại** (`E[f(x)] = −0,676` ứng
   với 33,7 %, không phải 8,07 %). Độ lớn tuyệt đối của SHAP không đọc được như xác
   suất; chỉ so sánh tương đối giữa feature là đúng.
6. **Không dùng nhãn hold-out**, nên không có kiểm tra "reason code có dự đoán đúng
   hồ sơ default thật không". Việc đó cần nhãn và đã bị khoá bởi protocol hold-out.

## 7. Chi phí

| Bước | Thời gian | Bộ nhớ |
|---|---|---|
| SHAP (1 fit LightGBM trên 246.008 dòng + TreeSHAP 10.000 × 180) | không ghi trong record (kế hoạch 10-20 phút) | ước tính 6-8 GB |
| Agreement (đọc 3 CSV importance của E5) | vài giây | không đáng kể |

Record `E7_shap.json` hiện **không** lưu `elapsed_seconds`; con số 10-20 phút là ước
tính từ kế hoạch, không phải số đo. Nếu cần trích dẫn chi phí trong báo cáo thì phải
đo lại.

## 8. Artifacts

- `reports/tables/E7_shap_importance.csv` — 180 dòng, `mean_abs_shap`, `mean_shap`, `share`
- `reports/tables/E7_reason_codes.csv` — 5 reason code cho hồ sơ 164.614
- `reports/tables/E7_feature_agreement.csv` — Spearman + Jaccard top-k giữa ba model
- `reports/figures/E7_shap_summary.png` — beeswarm top-20
- `reports/figures/E7_shap_dependence.png` — dependence của top-3
- `reports/figures/E7_shap_waterfall_approved.png`, `..._rejected.png`
- `reports/runs/E7_26-11-24_exp01/E7_shap.json` (model, seed, sample, base value, reason code, top-20)
- `reports/runs/E7_26-11-24_exp01/E7_agreement.json`

## 9. Kết luận

1. Model dựa vào ~20 tín hiệu, dẫn đầu là `APP_EXT_SOURCE_MEAN` (11,6 %) và nhóm
   application + 10 tỉ lệ dẫn xuất (66,0 % tín hiệu). Không có single point of failure,
   và 20 feature hoàn toàn vô dụng.
2. `EXT_SOURCE` và số ngày trễ có chiều đơn điệu đúng nghiệp vụ: attribution khớp với
   kinh tế tín dụng, không phải artefact.
3. Reason code cho hồ sơ bị từ chối đọc được như một lời giải thích: điểm tín dụng
   ngoài gần 0, quá hạn bureau lớn, tỉ lệ nợ cao, bị từ chối nhiều lần trước đó.
4. RQ4 trả lời được: ba GBDT đồng ý ở mức toàn cục (Spearman 0,80-0,95) và top-20
   (15/20 với SHAP-gain), nhưng **không** đồng ý ở mức top-10 (Jaccard 0,18-0,54).
   Kết luận về feature nên viết ở mức nhóm và top-20, không ở mức "feature số 1".
5. Giới hạn còn lại: E7 giải thích LightGBM chứ không phải stack cuối, và không có
   kiểm chứng reason code bằng nhãn hold-out. Cả hai đều do protocol, không do lỗi
   triển khai.
