# VACE-14B cho Stage 4 trên NUS cluster

**Active A100 40-GB profile:** use [A100_40GB.md](A100_40GB.md) and
`segmented_infer.py`: 3 x 5 s, 832x480, 81 raw frames per segment,
sequential CPU offload, final 240 frames at 16 fps. This supersedes the
original 720p/241-frame single-call plan. No GPU benchmark has been run.

Bộ script dùng **`Wan-AI/Wan2.1-VACE-14B-diffusers`**, nhận video gốc,
artwork, board masks, seed và output path. Model sinh toàn bộ frame đầu ra;
script lưu nguyên kết quả, không ghép lại background bằng classical compositing.
[AI-CODE]

**Trạng thái:** đã viết và review tĩnh theo API chính thức. Chưa cài trên
cluster, chưa tải weights, chưa import/load model hoặc chạy inference để
kiểm chứng. VRAM, RAM và thời gian thực tế cần xác nhận trên GPU được cấp.

## 1. Môi trường và tài nguyên

- Linux, Python **3.11**, NVIDIA GPU hỗ trợ BF16. Recipe dùng PyTorch 2.7.1
  CUDA **12.6**, hướng tới A100/H100 với driver tương thích. Nếu cluster cấp
  Blackwell/GPU khác, kiểm tra wheel tương thích trước, không tự đổi stack
  giữa các lần chạy chính thức.
- Cần **ffmpeg và ffprobe trên PATH**; dùng module của cluster nếu có.
- **Một process / một GPU**, không dùng torchrun/FSDP. Xin thêm GPU không tự
  cộng VRAM cho script. Ưu tiên GPU 80 GB và host RAM rộng cho CPU offload;
  đây là gợi ý cấp phát, không phải yêu cầu VRAM đã benchmark.
- Mẫu Slurm xin 128 GB host RAM, 8 CPU, 8 giờ. Sửa theo quota và scheduler
  thực tế; không giả định tên account, partition hoặc module của NUS.
- Đặt environment, weights và cache trên scratch/project storage ngoài repo
  và ZIP nộp bài. Không cài đè environment classical.

Từ repo trên cluster, thay các đường dẫn mẫu:

```bash
export LAB_ROOT=/absolute/path/to/CS4243_Lab2
export VACE_ENV=/absolute/scratch/path/venvs/lab2-vace
export VACE_WEIGHTS=/absolute/scratch/path/models/Wan2.1-VACE-14B-diffusers
export HF_HOME=/absolute/scratch/path/huggingface-cache
export VACE_PYTHON=python3.11
bash "$LAB_ROOT/code/student_files/vace/setup_cluster.sh"
```

Setup chỉ cài dependencies và `pip check`, không generation. Các thư viện chính
được pin: torch 2.7.1/cu126, Diffusers 0.35.1, Transformers 4.56.2,
Accelerate 1.10.1. Toàn bộ environment được ghi vào
`$VACE_ENV/installed-packages.txt`. Không cần build flash-attn hoặc CUDA
extensions cho recipe này. Các pin đã được đối chiếu API, chưa runtime-test.
[AI-CODE]

## 2. Tải weights trên node có Internet

```bash
"$VACE_ENV/bin/python" "$LAB_ROOT/code/student_files/vace/download_weights.py" \
  --weights "$VACE_WEIGHTS"
```

Downloader tải official **Diffusers format**, resolve `main` thành commit SHA,
ghi `lab_snapshot.json`. Lần tải lại dùng SHA đã ghi, không tự chuyển revision.
Có thể truyền `--revision <commit-sha>` ở lần đầu. Nếu compute node không có
Internet, tải trước trên node được phép kết nối và dùng shared storage.
Inference chỉ đọc weights local, không gọi API ngoài.

## 3. Chuẩn bị trước attempt chính thức

1. Quay clip, chuẩn hóa CFR và xóa metadata **trước annotation**.
2. Hoàn tất 10 audit corner frames và 5 audit masks, ít nhất 3 mask có vật che.
   Dùng công cụ annotate của lab để ghi `annotated_before_inference=yes`.
3. Chạy frozen classical pipeline trên clip. Dùng Notebook 4 để export
   `outputs/genai/inputs/masks/mask_XXXXXX.png`, mang chỉ số frame nguồn gốc.
4. Điền `code/manifests/genai_conditions.json` trước generation:
   - `clip_id`, đường dẫn artwork và reference theo format sẵn có;
   - `frames.clean`, `frames.occluded`, `frames.perspective`, `frames.pair`;
   - `model.name`: `Wan-AI/Wan2.1-VACE-14B-diffusers`;
   - `model.seed`: seed đã chọn, ví dụ `2026`;
   - `model.output_fps`: **16**;
   - `model.interval.start/end`: frame đầu/cuối **inclusive** trên clip gốc,
     hoặc null/null để lấy toàn clip.

Prompt giữ nguyên câu ngắn của lab, không prompt extension hay thêm chỉ dẫn
bảo toàn geometry/background. Script kiểm tra điều kiện, không tự điền audit
hoặc giả định annotation đã hoàn tất.

**Mask trắng = vùng board cần sinh; đen = vùng conditioning.** Dùng board
quadrilateral từ frozen pipeline. Không dùng audit foreground masks hoặc thư
mục `_masks` chỉ chứa vài mask vật che. Script không trừ foreground bằng GT.
Model tự xử lý artwork và thứ tự vật che. Board mask trống khi tracker lost
phải được giữ lại để phân tích; không sửa bằng audit labels. [AI-DESIGN]

## 4. Timing và số frame

Truyền **video gốc đầy đủ** `data/personal/<clip_id>/clip.mp4`, không truyền
`inputs/interval.mp4`. Script lấy cut từ conditions để tránh trừ offset hai
lần. Mask filenames vẫn mang chỉ số source gốc.

Script kiểm tra CFR bằng timestamp ffprobe, sample input ở **16 fps**. Video
và mask luôn lấy cùng source index. RGB resize trực tiếp về width/height;
mask resize nearest neighbour. Không crop source hoặc đổi annotation. Artwork
là reference image riêng, không được warp trước lên board.

The active plan divides the source into frames 0-149, 150-299 and 300-449.
Each model call emits 81 frames (4n+1); all raw frames are retained.
The joined output keeps the first 80 frames per segment, excluding only
the predeclared last padding frame. Final length is exactly 240/16 = 15 s.
See A100_40GB.md for the timestamp proof and unchanged audit mapping.

Các source frames đã chọn phải map sang các output frames phân biệt:

```text
output_index = round((source_index - interval.start) * 16 / source_fps)
```

Một số cặp source liên tiếp ở 30 fps sẽ trùng output index ở 16 fps. Preflight
sẽ từ chối; chọn cặp phù hợp **trước generation**. Notebook 4 dùng cùng công
thức. Sampling xử lý FPS, **không sửa được model retiming**: báo cáo motion
hoặc timing mismatch là lỗi, không tìm frame thuận lợi hơn. Không dùng tail
padding làm bằng chứng temporal trên source.

## 5. Preflight và inference

Sau khi có dữ liệu thật, từ shell cluster:

```bash
cd "$LAB_ROOT/code"
export VACE_VIDEO="$LAB_ROOT/data/personal/my_clip/clip.mp4"
export VACE_ARTWORK="$LAB_ROOT/data/artwork/my_artwork.png"
export VACE_MASKS="$LAB_ROOT/code/outputs/genai/inputs/masks"
export VACE_SEED=2026

"$VACE_ENV/bin/python" student_files/vace/segmented_infer.py preflight \
  --video "$VACE_VIDEO" --artwork "$VACE_ARTWORK" --masks "$VACE_MASKS" \
  --weights "$VACE_WEIGHTS" --seed "$VACE_SEED" \
  --output outputs/genai/output.mp4 --num-frames 81
```

Thay `my_clip`, `my_artwork` bằng dữ liệu thật. Preflight chỉ đọc inputs,
metadata và declarations; **không import torch, load model hoặc generation**.

Khi đã có GPU allocation, dùng cùng arguments và đổi `preflight` thành `run`.
Active defaults: 832x480, 16 fps, 81 raw frames for each of three segments,
50 steps, guidance 5, conditioning scale 1, UniPC flow shift 3, BF16 with
FP32 VAE, VAE tiling and sequential CPU offload. Do not change settings
between segments or automatically fall back after an OOM. Seed cố định giúp tái lập, không bảo đảm bitwise giống nhau giữa
GPU/library stacks khác nhau. [AI-CODE]

Hoặc submit mẫu Slurm từ thư mục này để console log nằm trong student_files:

```bash
cd "$LAB_ROOT/code/student_files/vace"
sbatch --account=<your-account> --partition=<your-gpu-partition> \
  submit_slurm.sh --num-frames 81
```

Nếu NUS allocation của bạn dùng scheduler khác, dùng lệnh `run` trong GPU job
của scheduler đó. Không chạy inference trên login node hoặc gọi bằng torchrun.

## 6. Artefacts và log

The segmented driver writes `outputs/genai/output.mp4` and retains each
81-frame raw result under `outputs/genai/segmented_attempt/segment_00/`
(also 01 and 02), with input snapshots, invocation logs and hashes.
`manifests/genai_log.csv` contains one row per segment/invocation.
The parent attempt manifest records the exact joining rule and status.

Use `segmented_infer.py run ... --resume` for an unchanged-settings
technical restart; completed segments are verified and never regenerated.
No automatic retry, seed search, overlap blending or background paste is
performed. See A100_40GB.md for output layout and stale-lock recovery.

Sau generation, tiếp tục Notebook 4 từ cell extract audit frames. Giữ nguyên
conditions so với snapshot. Notebook 4 resize audit extracts về source size
và score hai classical variants cùng generative trên các indices đã chọn và
cùng full audit dictionary. Gắn `[AI-GENERATION]` và báo cáo denominators.

## 7. Sources và provenance

- [Official VACE](https://github.com/ali-vilab/VACE)
- [Official Diffusers checkpoint](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
- [Diffusers v0.35.1 pipeline source](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/wan/pipeline_wan_vace.py)
- [Diffusers v0.35.1 Wan docs](https://github.com/huggingface/diffusers/blob/v0.35.1/docs/source/en/api/pipelines/wan.md)
- [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/)
- Lab handout Stage 4, Notebook 4 và genai_conditions.json trong repo.

Codex viết scripts/setup và đối chiếu API, input/output với nguồn chính thức.
Chưa có GPU runtime evidence; chỉ thêm `[HUMAN-CHECK]` sau khi bạn tự kiểm chứng
trên cluster. Lưu các file này trong appendix AI-use; loại environment, cache
và weights khỏi ZIP. [AI-CODE] [AI-LANGUAGE]
