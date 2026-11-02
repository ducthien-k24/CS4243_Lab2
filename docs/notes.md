# Lab 2

# Kế hoạch chia việc — CS4243 Lab 2 (2 người)

**Hạn nộp:** Chủ nhật 8/11/2026, 23:59 — **một file ZIP cho cả nhóm**

**Đề tài:** Perspective-Aware Virtual Advertisement Overlay

---

## 1. Tổng quan bài lab

| Stage | Nội dung | Điểm | File code |
| --- | --- | --- | --- |
| 1 – Localise | Harris, SIFT descriptor, matching, normalised DLT, RANSAC, warp | **35%** | `keypoints.py`, `geometry.py`, `localisation.py` |
| 2 – Track | Lucas-Kanade flow, 2 biến thể tracking, failure monitor | **20%** | `temporal.py` |
| 3 – Composite | Phát hiện vật che (compensated difference + hysteresis), alpha compositing | **20%** | `compositing.py` |
| 4 – Your clip | Tự quay clip, annotate, Scorecard, so sánh với model video sinh ảnh | **25%** | Notebook 4 |

Tổng cộng 23 hàm `YOUR CODE HERE`, kèm Notebook 1–4, `REPORT.md` (trả lời Q1–Q5) và các manifest.
**Báo cáo là phần chấm điểm lớn** — notebook chạy được thôi là chưa đủ.

### Phụ thuộc giữa các phần

- Stage 2 và 3 cần `localise_board` (Stage 1) để chạy trên dữ liệu thật → **Stage 1 phải xong sớm nhất**.
- Làm song song được ngay từ đầu: `bilinear_sample`, `hysteresis` (Lab 1), `lk_flow`, và quay/annotate clip cá nhân.
- Stage 3 có thể viết và debug trước trên `synthetic.py` (homography đã biết).
- Thứ tự chốt threshold: **Stage 1 → 2 → 3** (cùng ghi vào `outputs/thresholds.json`). Stage đã chốt thì không sửa nữa.

---

## 2. Phân công

### Người A — Localise + Track (~49%)

- **`keypoints.py`**: `harris_response`, `detect_corners`, `describe_keypoints`, `match_descriptors`
- **`geometry.py`**: `normalise_points`, `dlt_homography`, `symmetric_transfer_error`, `ransac_homography`, `transfer_points`, `warp_artwork`
- **`localisation.py`**: `localise_board`, `artwork_to_frame`, `overlay_frame`, `evaluate_audit_frames`
- **`temporal.py`**: `track_board`, `failure_monitor`
- Notebook 1; phần Stage 1/2 của Notebook 2; chốt threshold Stage 1 và Stage 2
- Báo cáo: **Stage 1 (Q1, Q2)** và **Stage 2 (Q3)**
- Lúc quay clip: **làm vật che** (tay/sách/bóng, **không lộ mặt**)
- Cuối kỳ: thống nhất `REPORT.md`, điền bảng evidence, **đóng gói ZIP**

### Người B — Flow + Composite + Your clip (~51%)

- **`geometry.py`**: `bilinear_sample` — **làm ngày đầu, push sớm** (A cần cho `warp_artwork`)
- **`temporal.py`**: `lk_flow` (chỉ cần Sobel + `bilinear_sample`, làm độc lập được)
- **`compositing.py`**: `hysteresis`, `compensated_difference`, `occlusion_mask`, `select_difference_thresholds`, `composite`
- Notebook 3; chốt threshold Stage 3
- **Stage 4:**
    - Quay clip (cầm máy), làm `reference.png` và artwork riêng
    - Chuyển định dạng bằng ffmpeg, rồi annotate 10 frame góc + 5 mask
    - `capture_manifest.csv`
    - Chọn và cài model sinh video, chạy **1 lần**, ghi `genai_log.csv`
    - Notebook 4, Overlay Scorecard
- Báo cáo: **Stage 3 (Q4)** và **Stage 4 (Q5)**
- Cuối kỳ: bảng AI-use disclosure, appendix provenance, kiểm tra checklist

### Cả hai cùng làm

- Chạy `run_pipeline --all`, `report`, `evaluate` (trên **một máy**)
- Đọc chéo báo cáo của nhau
- Mỗi người tự điền dòng AI-use cho phần mình

---

## 3. Lịch làm việc

| Ngày | A | B |
| --- | --- | --- |
| **30–31/10** | Setup, chạy test (ban đầu: 37 pass / 45 `NotImplementedError`). Viết Harris, NMS (cả `window` và `radius`), descriptor, matching, `normalise_points`, DLT | `bilinear_sample` (push sớm), `hysteresis`, `lk_flow`. **Kiểm tra GPU, email TA nếu không có.** Cùng A quay clip → ffmpeg → annotate |
| **1–2/11** | RANSAC, `transfer_points`, `warp_artwork`, `localise_board`, `artwork_to_frame`, `overlay_frame`, `evaluate_audit_frames`. Notebook 1. **Chốt Stage 1** | `compensated_difference`, `occlusion_mask`, `composite`, `select_difference_thresholds`, debug trên `synthetic.py`. Cài model sinh video, viết script inference |
| **3–4/11** | `track_board`, `failure_monitor`, Notebook 2. **Chốt Stage 2.** Bắt đầu viết báo cáo Stage 1–2 | Notebook 3 (dùng `localise_board` của A). **Chốt Stage 3.** Bắt đầu viết báo cáo Stage 3 |
| **5/11** | Notebook 1–3 từ kernel mới → `run_pipeline --all` full-res → `report` → `evaluate` 4 video còn lại | Chạy pipeline trên clip cá nhân (2 variant). **Chạy model sinh video 1 lần**, ghi log |
| **6/11** | Hoàn thiện Q1–Q3, hình, phân tích 2 trường hợp tệ nhất | Notebook 4, Scorecard, Q4–Q5, bảng so sánh classical vs generative |
| **7/11** | Đọc chéo, thống nhất `REPORT.md`, bảng evidence | Đọc chéo, bảng AI-use, appendix provenance |
| **8/11** | Chạy lại toàn bộ test + Notebook 1–4 từ kernel mới. A đóng gói, B kiểm tra checklist → **nộp trước tối** |  |

---

## 4. Stage 4 — chi tiết quay clip

**Một clip cho cả nhóm.** B cầm máy, A làm vật che.

**Yêu cầu clip:**

- 5–15 giây, **landscape**, frame rate cố định (khuyến nghị 1280×720, ≥24 fps). Khoá exposure/focus nếu được.
- Board phẳng, có texture (poster dán tường, bảng thông báo…), chiếm **≥15% frame đầu**, thấy được gần hết clip.
- Có đủ: **pan**, **đổi khoảng cách**, **đổi góc nhìn**, **một vật che đi ngang qua board**.
- **Không**: mặt người, giấy tờ riêng tư, biển số xe, vị trí cụ thể. Ghi rõ có sự đồng ý của người có tay xuất hiện. Xoá metadata.

**Thứ tự bắt buộc:**

1. Quay.
2. Chuyển định dạng (tạo thư mục trước):
    
    ```
    ffmpeg -i <recording.mp4> -map_metadata -1 -map_chapters -1 -an -vf fps=30 -c:v libx264 -crf 18 -pix_fmt yuv420p ../data/personal/<clip_id>/clip.mp4
    ```
    
3. Làm `data/personal/<clip_id>/reference.png` (board đã rectify, rộng **xấp xỉ bằng trên màn hình**, vì descriptor không bất biến tỉ lệ).
4. Làm artwork `data/artwork/<tên>.png`: đúng tỉ lệ board, không logo/thương hiệu/mặt người, nhiều chi tiết tương phản cao, có chữ hoặc motif dễ nhận ra.
5. **Chọn trước** 10 frame góc (gồm frame **clean**, **occluded**, **strongest-perspective**), 5 frame mask nằm trong 10 frame đó (≥3 frame có vật che thật, gồm frame occluded), và **một cặp frame liên tiếp**.
6. Annotate **trên laptop** (công cụ cần màn hình đồ hoạ, không chạy trên Colab/server):
Thứ tự góc: trên-trái → trên-phải → dưới-phải → dưới-trái. Vẽ mask **bằng tay** (không dùng segmenter cho audit frame).
    
    ```
    python -m overlaylab.annotate corners <clip_id> --frames <10 frame> --annotator <tên B>python -m overlaylab.annotate masks <clip_id> --frames <5 frame> --occluder "<vật che>" --annotator <tên B>python -m overlaylab.annotate check <clip_id>
    ```
    
7. Điền `manifests/capture_manifest.csv`.

⚠️ Sau khi annotate thì **không được cắt, resize, retime** clip nữa. Annotation dùng để chấm chỉ đứng tên B; nếu A annotate thêm để đối chiếu thì để trong `student_files/`.

### Model sinh video

- **Một** model open-weight, **một** lần chạy, settings và seed cố định. Nếu phải chia đoạn thì dùng cùng model/settings/seed, log từng đoạn, ghép theo thứ tự gốc. Cặp frame liên tiếp phải nằm trong cùng một đoạn.
- Gợi ý: MiniMax-H3, JoyAI-Video-Edit, LTX-2.3, Wan2.1-VACE-14B; nhẹ nhất là **Wan2.1-VACE-1.3B**. (Bản demo của staff: MiniMax-H3 mất ~36 phút trên GPU 24 GB.)
- Nếu model cần board mask thì phải chờ pipeline classical chạy xong trên clip (Notebook 4 export mask).
- Retry kỹ thuật được phép nhưng **giữ nguyên settings** và ghi vào log.
- Trích frame theo **timestamp gốc**. Nếu model retime thì báo cáo là lỗi, **không chọn frame đẹp hơn**.
- Lưu `outputs/genai/output.mp4`, `frame_XXXXXX.png`, `inputs/`. Gắn `[AI-GENERATION]`.
- Đánh giá cả hai phương pháp bằng `-frames` trên **cùng các frame**: clean, occluded, strongest-perspective, cặp liên tiếp.

---

## 5. Quy tắc dễ mất điểm

- **Chỉ tune trên frame validation của common clip** (13 frame chẵn + 3 mask). Không tune bằng test rows, stress clip hay audit frame của clip cá nhân.
- Không dùng `cv2.SIFT`, `cv2.findHomography`, `cv2.warpPerspective`, `cv2.calcOpticalFlowPyrLK`, `cv2.goodFeaturesToTrack` ngoài các cell đã dán nhãn. `cv2.getPerspectiveTransform` chỉ dùng cho GT-fallback trong Notebook 3 mục 6, có nhãn `GT-fallback`.
- Ground truth chỉ để chọn threshold trên validation và để đánh giá, **không dùng để khởi tạo hay sửa pipeline**.
- Video nộp phải **full resolution**. `-width 960` chỉ dùng khi phát triển.
- **Dán nguyên** `stress_tables.md` vào báo cáo, không sửa tay (grader sẽ build lại).
- Metric NaN/n/a: **để nguyên, không thay bằng 0**. Luôn ghi kèm số frame scored / hidden / missing.
- Failure analysis: lấy 2 trường hợp tệ nhất. Nếu ít hơn 2 frame fail thì vẫn đưa ra 2 frame tệ nhất và ghi rõ là chúng pass. Không cố tình phá phương pháp để tạo lỗi.
- Không đổi signature hàm public; không hard-code path, kích thước hay toạ độ góc. Ghi lại mọi seed (kể cả RANSAC).
- Mọi số liệu và hình phải ghi rõ **clip, subset (validation/test/audit), variant**.
- AI-use: gắn marker (`[AI-CODE]`, `[AI-LANGUAGE]`, `[HUMAN-CHECK]`…) ngay sau đoạn liên quan và điền bảng disclosure. Chỉ gắn `[HUMAN-CHECK]` khi đã kiểm bằng test, metric hay annotation.

---

## 6. Báo cáo (`code/REPORT.md`, dựa trên `REPORT_TEMPLATE.md`)

Giữ **tất cả section và bảng** của template; phần bổ sung đặt sau các section bắt buộc.

| Phần | Người viết | Cần có |
| --- | --- | --- |
| Stage 1 + Q1, Q2 | A | Implementation; stress table; plot inlier count/ratio theo frame; RANSAC vs least squares; 2 ca tệ nhất (matches, histogram residual, quadrilateral); SIFT cell; affine vs homography trên viewpoint clip; parallax |
| Stage 2 + Q3 | A | Stress table; jitter vs drift, số lần reinit; một đoạn flow ổn định và một đoạn flow fail; eigenvalue trên blur clip (aperture vs brightness constancy); timeline HIDE/REINIT trên exit clip; statistic nào kích hoạt trước trên combined clip |
| Stage 3 + Q4 | B | Stress table; mask IoU/F1/leakage; baseline không có occlusion; ảnh hưởng của strong threshold và min area; hysteresis với vật mỏng vs opening; false occlusion trên viewpoint clip |
| Stage 4 + Q5 | B | Điều kiện quay, privacy; Scorecard (cả 2 variant, 1 frame đại diện + 1 frame fail); bảng so sánh classical vs generative trên cùng frame; placement, artwork identity, occlusion order, background PSNR; controllability, cost; timing so với ví dụ MiniMax-H3 của staff |
| Bảng evidence | A | Hoàn thành đủ 5 dòng, mỗi hàm `YOUR CODE HERE` được nhắc ít nhất một lần trong phần viết |
| Bảng AI-use + appendix | B (A bổ sung phần mình) | Tool/model, artefact bị ảnh hưởng, marker, đóng góp, cách kiểm chứng; prompt, raw output, retry |

---

## 7. Checklist nộp bài

### Có trong ZIP

- [ ]  `code/` đầy đủ (overlaylab, tests, notebooks, manifests, README, `REPORT.md`, `REPORT_TEMPLATE.md`, requirements)
- [ ]  23 hàm đã viết xong; `python -m unittest discover -s tests -v` pass
- [ ]  Notebook 1–4 chạy được từ kernel mới
- [ ]  8 video MP4 của các clip được cấp (full-res, threshold đã chốt, `ad.png`):
`common_clip_independent`, `common_clip_flow_assisted`, `common_clip_flow_assisted_no_occlusion`, `coke_perspective_flow_assisted`, `coke_scale_flow_assisted`, `coke_blur_flow_assisted`, `coke_out_of_view_flow_assisted`, `coke_unconstrained_flow_assisted`
- [ ]  2 video MP4 của clip cá nhân (`independent`, `flow_assisted`)
- [ ]  Mỗi video có `_stats.json` và thư mục `_masks/`
- [ ]  `outputs/evaluation/` (report của mọi video, `genai_frames_report.json`, `stage4_genai_reports.json`)
- [ ]  `outputs/genai/` (inputs, `output.mp4`, `frame_XXXXXX.png`)
- [ ]  `outputs/scorecards/<clip_id>_scorecard.md`
- [ ]  `stress_tables.md`, `stress_tables.json`, `thresholds.json`, `stage1_settings.json`
- [ ]  Manifest: `capture_manifest.csv`, `audit_corners.csv`, `audit_masks.csv`, `genai_log.csv`, `genai_conditions.json` (đã điền)
- [ ]  `data/personal/<clip_id>/` (clip.mp4, reference.png, frames/, masks/)
- [ ]  `data/artwork/<tên>.png`
- [ ]  File phụ (appendix AI, script) chỉ nằm trong `code/student_files/`

### Không có trong ZIP

- [ ]  `data/common_clip/`, `data/diagnostic/`, `data/genai_oracle/`, `data/artwork/ad.png`
- [ ]  File `.webm`
- [ ]  Model weights, venv, `__pycache__`, `.ipynb_checkpoints`, cache (`subset_640/`…)
- [ ]  `outputs/annotation_check/` (trừ khi chọn lọc đưa vào `student_files/`)

---

## 8. Lệnh hay dùng (chạy trong `code/`)

```
pip install -r requirements.txt
python -m overlaylab.extract_frames ../data/common_clip
python -m unittest discover -s tests -v

python -m overlaylab.run_pipeline --all --thresholds outputs/thresholds.json
python -m overlaylab.report

python -m overlaylab.evaluate --video outputs/videos/common_clip_independent.mp4
python -m overlaylab.evaluate --video outputs/videos/coke_scale_flow_assisted.mp4 --clip coke_scale
python -m overlaylab.evaluate --video outputs/videos/coke_blur_flow_assisted.mp4 --clip coke_blur
python -m overlaylab.evaluate --video outputs/videos/coke_out_of_view_flow_assisted.mp4 --clip coke_out_of_view

python -m overlaylab.run_pipeline --clip <clip_id> --variant independent   --artwork <artwork> --thresholds outputs/thresholds.json
python -m overlaylab.run_pipeline --clip <clip_id> --variant flow_assisted --artwork <artwork> --thresholds outputs/thresholds.json
python -m overlaylab.evaluate --frames outputs/genai/frame_*.png --clip <clip_id> --artwork <artwork>
```

**Git:** chia branch theo file để tránh conflict. A sửa `keypoints` / `geometry` (trừ `bilinear_sample`) / `localisation` / `temporal` (trừ `lk_flow`); B sửa `bilinear_sample`, `lk_flow`, `compositing` và phần Stage 4.