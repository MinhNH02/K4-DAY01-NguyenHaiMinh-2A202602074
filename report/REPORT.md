# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
`class_id` 468, `class_name` cab, `rank` 1, `score` 0.510915, `taxonomy_name` ImageNet-1K.
- Record này mô tả toàn ảnh như thế nào? 
 Record này mô tả toàn ảnh: một nhãn duy nhất cho cả ảnh, không có vị trí hay số lượng. Model đoán "cab" với 0.51 trong khi ảnh có nhiều xe buýt, ô tô và người đi bộ..
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
 taxonomy ImageNet-1K của bộ dữ liệu huấn luyện checkpoint; model chỉ trả về 1 trong 1000 lớp đó, không có lớp "bus".
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
ID dùng cho máy, tên cho người đọc, taxonomy cho biết ID thuộc danh sách nào. Cùng số 5 là "bus" trong COCO-80 nhưng là lớp khác trong ImageNet-1K.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
 guideline phải quy định cách chọn chủ thể chính (diện tích lớn nhất, ở giữa, hay chủ đề), có cho nhiều nhãn không, và khi nào escalation.
- Vì sao model score không phải ground truth?
 score là độ tin cậy của model, có thể tự tin mà sai. Ví dụ `kitchen` có hạng 1 là "gong" 0.42. Ground truth là quyết định của người theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): 
`class_name` person, `score` 0.912558, `bbox_xyxy` [385.33, 69.24, 498.92, 348.92], `bbox_width` 113.58, `bbox_height` 279.68, đơn vị pixel, ảnh 640 × 427.
- Diễn giải vị trí box bằng lời: bên phải ảnh, từ 60% đến 78% chiều rộng và 16% đến 82% chiều cao; box cao gấp 2,5 lần rộng, là người đầu bếp đứng quay lưng.
- So sánh số prediction ở hai threshold: 0.35 cho 11 vật thể (person 2, bowl 5, oven 2, cup 2); 0.60 còn 6 (mất 3 bowl và 2 cup dưới 0.60); 0.20 [điền từ dòng in trong Colab].
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  hạ ngưỡng bắt thêm vật nhỏ nhưng reviewer xem nhiều box sai/trùng hơn; nâng ngưỡng ít việc hơn nhưng bỏ sót. Ngưỡng chỉ lọc prediction, ground truth vẫn phải gán đủ mọi vật thể.
- Đề xuất một quy tắc box chặt: box chạm pixel ngoài cùng của vật ở cả 4 cạnh, sai số ±2 px, không bao bóng hay vật kề bên.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
 Trong ảnh có 3 box chạm mép trái với x_min nhỏ hơn 1 px: 3 box chạm mép trái (person 0.61 chỉ thấy cánh tay, oven 0.69, bowl 0.38). Guideline cần quy định tỉ lệ nhìn thấy tối thiểu, vẽ theo phần thấy hay ước lượng, cờ truncated/occluded. Escalation khi không xác định được lớp, ví dụ "oven 0.63" bên phải có thể là bồn rửa.


## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
`instance_id` kitchen-001, `class_name` person, `score` 0.8993, 348 điểm, `polygon_xy` bắt đầu [446, 70], [445, 71], [444, 71]
- Polygon bổ sung chi tiết gì so với box?
đường viền thật của vật (tóc, vai, dây tạp dề, chân), loại bỏ nền trong box, cho biết pixel nào thuộc vật nào khi chồng lấn.
- `instance_id` dùng để làm gì và không phải loại ID nào?
định danh một vật thể trong một ảnh của output này để reviewer chỉ đích danh. Không phải `class_id` (4 bowl cùng lớp nhưng khác id), không phải id ảnh, không phải tracking id qua khung hình.
- Đề xuất một quy tắc biên mask: đi theo cạnh nhìn thấy, lệch tối đa 2 px, hai instance kề nhau không chồng pixel, mật độ điểm đủ để không lệch cạnh thật quá 2 px.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
guideline quyết định pixel chung của cụm bowl ở góc dưới trái thuộc vật nào, mask dining table có gồm phần dưới bát và bột không. Escalation khi hai model cùng ngưỡng cho tập vật thể khác nhau (seg thấy potted plant, dining table, 2 spoon; det thấy 2 cup, oven phải).

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn/ảnh theo taxonomy cố định  |`traffic` đoán "cab" dù nhiều buýt; `kitchen` đoán "gong"  |Chọn một nhãn theo quy tắc chủ thể chính, không dựa vào score | Nhãn đúng quy tắc và taxonomy chưa, nhất quán giữa annotator chưa |
| Phát hiện vật thể | Mỗi vật một box xyxy pixel + lớp  |3 box cắt mép trái; cup/bowl chồng nhau; nồi chảo treo không có box |Vẽ box chặt cho mọi vật kể cả vật model bỏ sót, đánh cờ che khuất  | Box chặt chưa, thiếu vật không, lớp đúng không, box trùng không |
| Instance segmentation |Mỗi instance một `instance_id` + lớp + polygon pixel  |Seg và det cho tập vật khác nhau; mask bàn phủ cả bát; bowl kitchen-010 chỉ 12 điểm  | Vẽ polygon theo biên nhìn thấy, mỗi vật một id, không chồng pixel | Biên lệch quá 2 px không, instance gộp/tách sai không, id duy nhất không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: không ghi họ tên, MSSV, email vào báo cáo hay output; chỉ dùng 3 ảnh COCO trong phạm vi bài và giữ attribution.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:  giảng viên hoặc Lab Coach

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
