"""Run a separate MiniMax-H3 Ref2VA experiment on poster_03. [AI-CODE]"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "student_files" / "vace"))
from infer import digest, probe, save_json
from segment_timing import audit_mapping
from segmented_infer import join_segments

MODEL = "MiniMaxAI/MiniMax-H3"
PROMPT = "Replace the advertisement on the board in <Video 1> with the attached artwork <Picture 1>."
SEED = 2026
FPS = 24
WIDTH, HEIGHT = 864, 480
STEPS = 30
RAW_FRAMES, KEEP_FRAMES = 124, 120  # 124 = 17 * 7 + 5, H3's VAE frame grid
PARTS = ("transformer_ref", "text_encoder", "tokenizer", "processor", "vae",
         "audio_vae", "scheduler", "audio_scheduler")


def download(weights):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download

    weights.mkdir(parents=True, exist_ok=True)
    request = weights / "lab_download_request.json"
    if request.exists():
        metadata = json.loads(request.read_text())
        if metadata["model"] != MODEL:
            raise ValueError(f"Checkpoint directory already belongs to {metadata['model']}")
    else:
        metadata = {"model": MODEL, "revision": HfApi().model_info(MODEL).sha}
        save_json(request, metadata)
    snapshot_download(
        MODEL, revision=metadata["revision"], local_dir=weights, max_workers=1,
        allow_patterns=["modular_model_index.json", "model_index.json", "LICENSE",
                        *(f"{part}/*" for part in PARTS)],
    )
    save_json(weights / "lab_snapshot.json", metadata)
    print(f"Downloaded {MODEL}@{metadata['revision']} to {weights}", flush=True)


def run(weights, output_dir):
    import av
    from PIL import Image
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    from diffusers.utils.export_utils import encode_video

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name or gpu.total_memory < 75 * 1024**3:
        raise RuntimeError(f"Expected an A100 80 GB, got {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)")

    snapshot = json.loads((weights / "lab_snapshot.json").read_text())
    if snapshot["model"] != MODEL:
        raise ValueError("Unexpected checkpoint in MINIMAX_WEIGHTS")
    video = CODE.parent / "data/personal/poster_03/clip.mp4"
    artwork = CODE.parent / "data/artwork/ad4.png"
    info = probe(video)
    if (info["frames"], info["fps"]) != (450, 30):
        raise ValueError("Expected the 450-frame, 30 fps poster_03 source")
    with Image.open(artwork) as image:
        picture = MiniMaxH3ImageReference(image=image.convert("RGB"))

    segments = [dict(index=i, start=i * 150, end=i * 150 + 149,
                     output_start=i * KEEP_FRAMES, keep_frames=KEEP_FRAMES)
                for i in range(3)]
    pair = [390, 391]
    mapping = audit_mapping(segments, [0, 45, 240, *pair], 30, FPS, pair)
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {**snapshot, "status": "loading", "workflow": "ref2va", "prompt": PROMPT,
              "seed": SEED, "steps": STEPS, "fps": FPS, "width": WIDTH, "height": HEIGHT,
              "raw_frames_per_segment": RAW_FRAMES, "reference_image_short_edge": 768,
              "precision": "bfloat16", "offload": "ComponentsManager auto CPU offload",
              "gpu": gpu.name, "gpu_bytes": gpu.total_memory,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "video_sha256": digest(video),
              "artwork_sha256": digest(artwork), "code_sha256": digest(Path(__file__)),
              "versions": {pkg: importlib.metadata.version(pkg) for pkg in
                           ("torch", "diffusers", "transformers", "accelerate", "av")},
              "audit_output_frames": mapping, "segments": segments}
    record_path = output_dir / "attempt.json"
    save_json(record_path, record)
    try:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        pipe = ModularPipeline.from_pretrained(str(weights), workflow="ref2va", components_manager=manager)
        pipe.update_components(reference_image_short_edge=768)
        pipe.load_components(dtype=torch.bfloat16)
        record["status"] = "running"
        raw_outputs = []
        with av.open(str(video)) as container:
            frames = container.decode(video=0)
            for segment in segments:
                source_frames = [next(frames).to_image() for _ in range(150)]
                reference = MiniMaxH3VideoReference(frames=source_frames, fps=30.0)
                segment["status"] = "running"
                save_json(record_path, record)
                print(f"Generating source frames {segment['start']}..{segment['end']}", flush=True)
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                result = pipe(prompt=PROMPT, references=[reference, picture], num_frames=RAW_FRAMES,
                              height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                              generator=torch.Generator().manual_seed(SEED), output=["videos"])
                raw = output_dir / f"segment_{segment['index']:02d}_raw.mp4"
                encode_video(result["videos"][0], fps=FPS, output_path=str(raw))
                del result, reference, source_frames
                raw_info = probe(raw)
                if (raw_info["frames"], raw_info["fps"]) != (RAW_FRAMES, FPS):
                    raise RuntimeError(f"Unexpected raw segment metadata: {raw_info}")
                segment.update(status="complete", raw=raw.name, sha256=digest(raw),
                               wall_seconds=time.monotonic() - started,
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                raw_outputs.append({"path": raw, "keep_frames": KEEP_FRAMES})
                save_json(record_path, record)

        joined = output_dir / "output.mp4"
        temporary = output_dir / "joined.partial.mp4"
        join_segments(raw_outputs, temporary, FPS, "ffmpeg")
        joined_info = probe(temporary)
        if (joined_info["frames"], joined_info["fps"]) != (360, FPS):
            raise RuntimeError(f"Unexpected joined metadata: {joined_info}")
        temporary.rename(joined)
        record.update(status="complete", output_sha256=digest(joined), joined_info=joined_info)
        print(f"[AI-GENERATION] Complete: {joined}", flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save_json(record_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "run"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "download":
        download(args.weights)
    else:
        if args.output_dir is None:
            parser.error("run requires --output-dir")
        run(args.weights.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
