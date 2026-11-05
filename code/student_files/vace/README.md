# VACE-14B cho Stage 4 trên NUS cluster

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

Số frame model phải là **4n+1**. Mặc định **81**, tương đương 5.0625 giây
output. Script tính số frame hợp lệ nhỏ nhất đủ phủ cut và báo lỗi nếu
`--num-frames` không khớp; không âm thầm cắt clip hoặc retime thành 81 frame.
Ví dụ 150 source frames @30 fps dùng 81 model frames; 165 frames @30 fps dùng
**89**. Đuôi ngắn được pad bằng frame cuối để thỏa VAE và ghi rõ trong mapping.
Clip dài hơn có thể cần nhiều VRAM/thời gian hơn.

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

"$VACE_ENV/bin/python" student_files/vace/infer.py preflight \
  --video "$VACE_VIDEO" --artwork "$VACE_ARTWORK" --masks "$VACE_MASKS" \
  --weights "$VACE_WEIGHTS" --seed "$VACE_SEED" \
  --output outputs/genai/output.mp4 --num-frames 81
```

Thay `my_clip`, `my_artwork` bằng dữ liệu thật. Preflight chỉ đọc inputs,
metadata và declarations; **không import torch, load model hoặc generation**.

Khi đã có GPU allocation, dùng cùng arguments và đổi `preflight` thành `run`.
Mặc định: 1280x720, 16 fps, 50 steps, guidance 5, conditioning scale 1,
UniPC flow shift 5, BF16 + VAE FP32, VAE tiling và model CPU offload. Muốn bỏ
CPU offload dùng `--offload none`, chọn trước attempt. Không tự đổi settings
khi OOM. Seed cố định giúp tái lập, không bảo đảm bitwise giống nhau giữa
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

Script tạo:

- `outputs/genai/output.mp4`: bản sao byte-for-byte của raw generation;
- `outputs/genai/runs/<id>/raw_output.mp4`: giữ nguyên output model;
- `runs/<id>/invocation.json`, `pip_freeze.txt`, `error.txt` nếu lỗi;
- `outputs/genai/inputs/vace/<id>/`: input PNGs thực tế, binary masks,
  artwork, conditions snapshot và `plan.json` chứa mapping/hashes;
- `manifests/genai_log.csv`: một row mỗi invocation, trạng thái running,
  complete hoặc failed; model SHA, settings, seed, prompt, FPS, compute, time.

`processing_time_s` gồm preparation/load/inference/export; inference-only time
nằm trong notes/manifest. Script không ghi tokens/API keys vào provenance.
Output có sẵn không bị ghi đè. CSV được ghi trước model loading; không tự
retry hoặc chọn kết quả tốt nhất. Chỉ retry lỗi kỹ thuật với cùng inputs,
model/settings/seed và thêm `--retry-note "<lý do và invocation cũ>"`.
Không đổi output path để thử seed/prompt khác hoặc cherry-pick kết quả.

Job bị SIGKILL có thể để lại row `running` và `.vace.lock` cạnh file CSV.
Chỉ xóa lock sau khi xác nhận job cũ đã dừng; giữ raw/log cũ và ghi retry note.
Recipe hỗ trợ **một cut/một invocation**, chưa tự chia/ghép segments. Nếu cần
segments, thiết kế thêm mapping, log và ghép source-order trước khi generation.

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
