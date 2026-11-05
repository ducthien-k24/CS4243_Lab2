"""VACE-14B single-GPU Stage 4 runner; no classical output compositing. [AI-CODE]

preflight validates local inputs without importing torch or loading weights.
run preserves the actual model inputs, raw generation, timestamps and lab CSV log.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from segment_timing import validate_segments, audit_mapping

CODE = Path(__file__).resolve().parents[2]
MODEL = "Wan-AI/Wan2.1-VACE-14B-diffusers"
PROMPT = "Replace the advertisement on the board in this video with the attached artwork."
VERSIONS = {"torch": "2.7.1", "diffusers": "0.35.1", "transformers": "4.56.2",
            "accelerate": "1.10.1", "huggingface-hub": "0.34.4"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def probe(path):
    """Decode timestamps with ffprobe: reject VFR instead of guessing nominal timing."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    doc = json.loads(result.stdout)
    stream = doc["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    times = [float(f["best_effort_timestamp_time"]) for f in doc.get("frames", [])]
    if fps <= 0 or not times:
        raise ValueError(f"No valid timestamped video in {path}")
    if max(abs((t - times[0]) - i / fps) for i, t in enumerate(times)) > 0.002:
        raise ValueError("Video is not CFR within 2 ms; normalise it BEFORE audit annotation")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": fps, "frames": len(times), "first_pts": times[0]}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check_audit(conditions, chosen):
    """Check declarations and files, never use GT annotations as model masks."""
    clip_id = conditions["clip_id"]
    corners_path = CODE / "manifests" / "audit_corners.csv"
    masks_path = CODE / "manifests" / "audit_masks.csv"
    corners = [r for r in csv_rows(corners_path) if r["clip_id"] == clip_id]
    masks = [r for r in csv_rows(masks_path) if r["clip_id"] == clip_id]
    corner_ids = {int(r["frame"]) for r in corners}
    mask_ids = {int(r["frame"]) for r in masks}
    if len(corner_ids) != 10 or len(mask_ids) != 5 or not mask_ids <= corner_ids:
        raise ValueError("Require 10 audit corner frames and 5 mask frames among them")
    if not all(r.get("annotated_before_inference", "").strip().lower() in ("true", "1", "yes")
               for r in corners + masks):
        raise ValueError("Audit manifests must declare annotated_before_inference=true")
    if not set(chosen[:3]) <= corner_ids or conditions["frames"]["occluded"] not in mask_ids:
        raise ValueError("Named clean/occluded/perspective frames need corners; occluded also needs a mask")
    # Read audit labels solely to validate the capture protocol, not to form inputs.
    from PIL import Image
    import numpy as np
    nonempty = 0
    for row in masks:
        with Image.open((masks_path.parent / row["mask_path"]).resolve()) as im:
            nonempty += bool((np.asarray(im.convert("L")) > 127).any())
    if nonempty < 3:
        raise ValueError("At least three audit masks must contain actual foreground")
    return {"corners_sha256": digest(corners_path), "masks_sha256": digest(masks_path)}


def make_plan(args, *, allow_existing_output=False):
    if not shutil.which("ffprobe"):
        raise RuntimeError("Load/install ffmpeg including ffprobe on the cluster")
    if args.output.exists() and not allow_existing_output:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if (args.seed < 0 or args.steps < 1
            or not all(math.isfinite(v) and v >= 0 for v in (args.guidance, args.conditioning_scale, args.flow_shift))):
        raise ValueError("Use a nonnegative seed/scales and positive steps")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("Explicit model width and height must be positive multiples of 16")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        raise ValueError("num_frames must be 4n+1, at least 5")
    if args.fps != 16:
        raise ValueError("This recipe uses VACE's documented 16-fps setting")
    if args.output.suffix.lower() != ".mp4" or not args.masks.is_dir():
        raise ValueError("Output must be MP4 and masks must be a directory of source-indexed PNGs")
    for file in (args.video, args.artwork, args.conditions):
        if not file.is_file():
            raise FileNotFoundError(file)
    if not args.weights.is_dir():
        raise FileNotFoundError("Download weights with download_weights.py first")
    snapshot = json.loads((args.weights / "lab_snapshot.json").read_text(encoding="utf-8"))
    if snapshot["repo_id"] != MODEL:
        raise ValueError("Require the official 14B Diffusers checkpoint")
    for component in ("vae", "text_encoder", "tokenizer", "scheduler", "transformer"):
        if not (args.weights / component).is_dir():
            raise FileNotFoundError(f"Checkpoint component missing: {component}")
    model_index = json.loads((args.weights / "model_index.json").read_text(encoding="utf-8"))
    if model_index.get("_class_name") != "WanVACEPipeline":
        raise ValueError("Checkpoint is not a WanVACEPipeline")
    cond = json.loads(args.conditions.read_text(encoding="utf-8"))
    if cond["model"].get("seed") != args.seed or cond["model"].get("output_fps") != args.fps:
        raise ValueError("Set the same seed and output_fps in genai_conditions.json BEFORE generation")
    if cond["model"].get("name") != MODEL:
        raise ValueError(f"Set model.name in genai_conditions.json to {MODEL}")
    if cond["prompt"]["template"] != PROMPT:
        raise ValueError("Use the short lab prompt; this recipe does not extend it")
    declared = cond["model"].get("settings", {})
    for key, value in {"width": args.width, "height": args.height, "num_frames": args.num_frames,
                       "num_inference_steps": args.steps, "guidance_scale": args.guidance,
                       "conditioning_scale": args.conditioning_scale, "flow_shift": args.flow_shift,
                       "cpu_offload": args.offload}.items():
        if key in declared and declared[key] != value:
            raise ValueError(f"{key}={value} disagrees with predeclared conditions ({declared[key]}); "
                             "use the segmented runner or correct settings BEFORE the first attempt")
    expected_artwork = (CODE / "notebooks" / cond["artwork"]).resolve()
    expected_video = (CODE.parent / "data" / "personal" / cond["clip_id"] / "clip.mp4").resolve()
    if args.artwork.resolve() != expected_artwork or args.video.resolve() != expected_video:
        raise ValueError("Pass the original full personal clip and the artwork declared in genai_conditions.json")
    # Audit masks are evaluation labels, never the inference edit-region masks.
    audit_dir = (expected_video.parent / "masks").resolve()
    if args.masks.resolve() == audit_dir:
        raise ValueError("Use full-length frozen-pipeline BOARD masks from Notebook 4, not audit occlusion masks")
    info = probe(args.video)
    interval = cond["model"]["interval"]
    start = interval["start"] if interval["start"] is not None else 0
    end = interval["end"] if interval["end"] is not None else info["frames"] - 1
    if (type(start) is not int or type(end) is not int
            or not 0 <= start <= end < info["frames"]):
        raise ValueError("Invalid inclusive source-frame interval in conditions")
    frames = cond["frames"]
    named = [frames[k] for k in ("clean", "occluded", "perspective")]
    pair = frames["pair"]
    if (len(pair) != 2 or not all(type(f) is int for f in named + pair)
            or len(set(named)) != 3 or pair[1] != pair[0] + 1):
        raise ValueError("Preselect three distinct named frames and a consecutive source-frame pair")
    chosen = named + pair
    if not all(start <= f <= end for f in chosen):
        raise ValueError("The declared cut must contain all selected frames")
    mapped = {f: round((f - start) * args.fps / info["fps"]) for f in set(chosen)}
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("Selected source frames collide at 16 fps; choose a valid pair BEFORE generation")
    full_interval = {"start": start, "end": end}
    segment = None
    if declared.get("execution_mode") == "segmented":
        segments = validate_segments(cond["model"]["segments"], start, end, info["fps"], args.fps)
        audit_mapping(segments, set(chosen), info["fps"], args.fps, pair)
        if args.segment_index is None or not 0 <= args.segment_index < len(segments):
            raise ValueError("Use segmented_infer.py to run the complete predeclared segmented attempt")
        segment = segments[args.segment_index]
        start, end = segment["start"], segment["end"]
        mapped = {f: round((f - start) * args.fps / info["fps"])
                  for f in set(chosen) if start <= f <= end}
    elif args.segment_index is not None:
        raise ValueError("--segment-index requires predeclared segmented conditions")
    duration = (end - start + 1) / info["fps"]
    # Only minimal tail padding to satisfy VAE's 4n+1 constraint is permitted.
    minimum = math.ceil(max(0, duration * args.fps - 1) / 4) * 4 + 1
    if args.num_frames != minimum:
        raise ValueError(f"This interval needs --num-frames {minimum} at 16 fps. "
                         "Choose a cut in conditions or explicitly use that frame count; no silent truncation.")
    if mapped and max(mapped.values()) >= args.num_frames:
        raise ValueError("Selected frame is outside generated timestamps")
    mapping = [{"model_frame": k, "source_frame": min(end, start + round(k * info["fps"] / args.fps)),
                "nominal_source_time_seconds": start / info["fps"] + k / args.fps,
                "sampled_source_time_seconds": min(end, start + round(k * info["fps"] / args.fps)) / info["fps"],
                "tail_padding": k / args.fps > (end - start) / info["fps"]}
               for k in range(args.num_frames)]
    audit = check_audit(cond, chosen)
    from PIL import Image
    with Image.open(args.artwork) as artwork:
        if artwork.format != "PNG":
            raise ValueError("The personal artwork must be a PNG")
        artwork.verify()
    for frame_id in sorted({entry["source_frame"] for entry in mapping}):
        mask_path = args.masks / f"mask_{frame_id:06d}.png"
        with Image.open(mask_path) as mask:
            if mask.size != (info["width"], info["height"]):
                raise ValueError(f"Mask {mask_path} must match the original video dimensions")
    return {"model": MODEL, "revision": snapshot["revision"], "clip_id": cond["clip_id"],
            "prompt": PROMPT, "negative_prompt": "", "seed": args.seed,
            "interval": {"start": start, "end": end}, "full_interval": full_interval,
            "segment": segment, "source": info,
            "width": args.width, "height": args.height, "fps": args.fps,
            "num_frames": args.num_frames, "steps": args.steps, "guidance": args.guidance,
            "conditioning_scale": args.conditioning_scale, "flow_shift": args.flow_shift,
            "offload": args.offload, "precision": "bfloat16; VAE float32", "audit": audit,
            "audit_output_frames": mapped, "frame_mapping": mapping,
            "video_sha256": digest(args.video), "artwork_sha256": digest(args.artwork),
            "conditions_sha256": digest(args.conditions)}


def prepare_inputs(args, plan, folder):
    """Save the exact sampled PIL inputs; preserve source-index mapping in plan.json."""
    from PIL import Image
    import cv2
    import numpy as np
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "frames").mkdir()
    (folder / "masks").mkdir()
    shutil.copy2(args.artwork, folder / "artwork.png")
    shutil.copy2(args.conditions, folder / "genai_conditions.json")
    wanted = {entry["source_frame"] for entry in plan["frame_mapping"]}
    cap = cv2.VideoCapture(str(args.video))
    decoded = {}
    try:
        for index in range(max(wanted) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode source frame {index}")
            if index in wanted:
                decoded[index] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
                    (args.width, args.height), Image.Resampling.LANCZOS)
    finally:
        cap.release()
    images, masks, mask_hashes = [], [], {}
    for entry in plan["frame_mapping"]:
        k, index = entry["model_frame"], entry["source_frame"]
        image = decoded[index]
        original_mask = args.masks / f"mask_{index:06d}.png"
        with Image.open(original_mask) as im:
            mask = im.convert("L").resize((args.width, args.height), Image.Resampling.NEAREST)
            # Explicit white=generate, black=condition. Do not invert board masks.
            mask = Image.fromarray(((np.asarray(mask) > 127) * 255).astype(np.uint8)).convert("RGB")
        image.save(folder / "frames" / f"frame_{k:06d}.png")
        mask.save(folder / "masks" / f"mask_{k:06d}.png")
        mask_hashes[str(index)] = digest(original_mask)
        images.append(image)
        masks.append(mask)
    if not any(np.asarray(mask).any() for mask in masks):
        raise ValueError("All sampled board masks are empty; inspect frozen pipeline before generation")
    plan["mask_sha256"] = mask_hashes
    save_json(folder / "plan.json", plan)
    with Image.open(args.artwork) as im:
        artwork = im.convert("RGB").copy()
    return images, masks, artwork


def update_log(path, invocation_id, row):
    """Keep one row per invocation, including failures and interrupted-running records."""
    template = CODE / "manifests" / "genai_log_template.csv"
    with template.open(encoding="utf-8", newline="") as stream:
        fields = next(csv.reader(stream))
    records = csv_rows(path) if path.exists() else []
    matches = []
    for i, old in enumerate(records):
        try:
            notes = json.loads(old.get("notes") or "{}")
        except (ValueError, TypeError):
            continue  # Preserve pre-existing human-written free-text notes.
        if isinstance(notes, dict) and notes.get("invocation_id") == invocation_id:
            matches.append(i)
    if matches:
        records[matches[0]] = row
    else:
        records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _run_invocation(args, plan):
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    raw_dir = args.output.parent / "runs" / invocation_id
    input_dir = args.output.parent / "inputs" / "vace" / invocation_id
    raw_dir.mkdir(parents=True)
    save_json(raw_dir / "invocation.json", {"status": "starting", "plan": plan})
    row = {"date": datetime.now(timezone.utc).isoformat(), "model": MODEL,
           "visible_version": plan["revision"],
           "settings": json.dumps({k: v for k, v in plan.items() if k not in ("frame_mapping", "audit")}),
           "seed": args.seed, "interval": json.dumps(plan["interval"]), "output_fps": args.fps,
           "prompt": PROMPT, "inputs": json.dumps({"video": str(args.video.resolve()),
               "artwork": str(args.artwork.resolve()), "board_masks": str(args.masks.resolve()),
               "prepared": str(input_dir.resolve())}),
           "input_resolution": f"{plan['source']['width']}x{plan['source']['height']}",
           "output_resolution": f"{args.width}x{args.height}", "processing_time_s": "",
           "compute": "", "output_path": str(args.output.resolve()),
           "notes": json.dumps({"invocation_id": invocation_id, "status": "running", "marker": "[AI-GENERATION]"})}
    started = time.perf_counter()
    status, error = "failed", None
    try:
        update_log(args.log, invocation_id, row)
        images, masks, artwork = prepare_inputs(args, plan, input_dir)
        import numpy as np
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler
        from diffusers.utils import export_to_video

        installed = {name: importlib.metadata.version(name) for name in VERSIONS}
        for name, required in VERSIONS.items():
            if installed[name].split("+")[0] != required:
                raise RuntimeError(f"Use pinned environment: {name}=={required}, got {installed[name]}")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Require an NVIDIA CUDA GPU with BF16 support")
        row["compute"] = json.dumps({"gpu": torch.cuda.get_device_name(0),
                                     "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
                                     "cuda": torch.version.cuda, "libraries": installed})
        update_log(args.log, invocation_id, row)
        packages = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                  check=True, capture_output=True, text=True).stdout
        (raw_dir / "pip_freeze.txt").write_text(packages, encoding="utf-8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        vae = AutoencoderKLWan.from_pretrained(
            str(args.weights), subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
        pipe = WanVACEPipeline.from_pretrained(
            str(args.weights), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=plan["flow_shift"])
        pipe.vae.enable_tiling()
        if args.offload == "model":
            pipe.enable_model_cpu_offload()
        elif args.offload == "sequential":
            # Offload submodules instead of placing the entire 14B transformer
            # on the 40-GB GPU. Slower, but leaves more room for activations.
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            generated = pipe(video=images, mask=masks, reference_images=[artwork],
                             prompt=PROMPT, negative_prompt="", width=args.width, height=args.height,
                             num_frames=args.num_frames, num_inference_steps=args.steps,
                             guidance_scale=args.guidance, conditioning_scale=args.conditioning_scale,
                             generator=torch.Generator(device="cpu").manual_seed(args.seed)).frames[0]
        torch.cuda.synchronize()
        plan["inference_seconds"] = time.perf_counter() - inference_start
        plan["peak_vram_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        plan["peak_vram_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        if generated.shape != (args.num_frames, args.height, args.width, 3):
            np.save(raw_dir / "unexpected_output.npy", generated)
            raise RuntimeError(f"Unexpected generated video shape: {generated.shape}")
        if not np.isfinite(generated).all():
            np.save(raw_dir / "nonfinite_output.npy", generated)
            raise RuntimeError("Model returned non-finite pixels; raw array retained for technical diagnosis")
        # Save the ENTIRE generated frame. Never paste generated pixels onto the source.
        raw_output = raw_dir / "raw_output.mp4"
        export_to_video(generated, str(raw_output), fps=args.fps, quality=8, macro_block_size=1)
        output_info = probe(raw_output)
        if output_info["frames"] != args.num_frames or abs(output_info["fps"] - args.fps) > 1e-6:
            raise RuntimeError("Encoded output has unexpected timing or frame count")
        # Exclusive destination creation prevents overwriting another result.
        with raw_output.open("rb") as source, args.output.open("xb") as target:
            shutil.copyfileobj(source, target)
        plan["output_sha256"] = digest(args.output)
        status = "complete"
        print(f"[AI-GENERATION] Saved {args.output}; raw output retained at {raw_output}")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (raw_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        row["processing_time_s"] = f"{time.perf_counter() - started:.3f}"
        row["notes"] = json.dumps({"invocation_id": invocation_id, "status": status,
                                   "marker": "[AI-GENERATION]", "error": error,
                                   "inference_seconds": plan.get("inference_seconds"),
                                   "peak_vram_allocated_gib": plan.get("peak_vram_allocated_gib"),
                                   "peak_vram_reserved_gib": plan.get("peak_vram_reserved_gib"),
                                   "raw_output": str(raw_dir / "raw_output.mp4"),
                                   "technical_retry_note": args.retry_note})
        save_json(raw_dir / "invocation.json", {"status": status, "error": error, "plan": plan, "log_row": row})
        update_log(args.log, invocation_id, row)


def run(args, plan):
    # A single process owns both the CSV and output. No torchrun replicas.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch one Python process on one GPU; multi-GPU is not implemented in this recipe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.log.with_suffix(".vace.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"Active process {os.getpid()}; remove only after confirming the job is no longer running.\n")
    try:
        _run_invocation(args, plan)
    finally:
        lock.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    for name in ("video", "artwork", "masks", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--conditions", type=Path, default=CODE / "manifests" / "genai_conditions.json")
    parser.add_argument("--log", type=Path, default=CODE / "manifests" / "genai_log.csv")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--segment-index", type=int, default=None,
                        help="Internal segment index; normally use segmented_infer.py")
    parser.add_argument("--retry-note", default="", help="Describe unchanged-settings technical retry, if applicable")
    return parser


def main():
    args = build_parser().parse_args()
    plan = make_plan(args)
    print(json.dumps(plan, indent=2))
    if args.action == "preflight":
        print("Preflight complete; no weights loaded and no video generated.")
    else:
        run(args, plan)


if __name__ == "__main__":
    main()
