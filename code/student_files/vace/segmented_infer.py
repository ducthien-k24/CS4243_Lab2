"""Run one predeclared VACE attempt in memory-bounded segments. [AI-CODE]

Each segment uses an isolated process, identical settings/seed, and retains
its full raw generation. Joining discards only predeclared VAE tail padding.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import infer as core
from segment_timing import audit_mapping


def join_segments(items, destination, fps, ffmpeg):
    """Concatenate exact frame prefixes, without FPS conversion or seam blending.

    H.264 lossless re-encoding preserves decoded YUV frames of the segment
    MP4s. Every original 4n+1 raw video is retained separately.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    cmd = [str(ffmpeg), "-hide_banner", "-nostdin", "-n"]
    filters, labels = [], []
    for index, item in enumerate(items):
        cmd.extend(["-i", str(item["path"])])
        filters.append(f"[{index}:v:0]trim=start_frame=0:end_frame={item['keep_frames']},"
                       f"setpts=PTS-STARTPTS[s{index}]")
        labels.append(f"[s{index}]")
    filters.append("".join(labels) + f"concat=n={len(items)}:v=1:a=0[out]")
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                "-map_metadata", "-1", "-map_chapters", "-1",
                "-c:v", "libx264", "-preset", "fast", "-crf", "0", "-pix_fmt", "yuv420p",
                "-r", str(fps), "-fps_mode", "cfr", "-movflags", "+faststart", str(destination)])
    # Do not impose -frames:v here: an erroneous extra frame must fail validation,
    # not be silently cut to the expected total.
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def child_command(args, index, output):
    return [sys.executable, str(Path(core.__file__).resolve()), "run",
            "--video", str(args.video.resolve()), "--artwork", str(args.artwork.resolve()),
            "--masks", str(args.masks.resolve()), "--weights", str(args.weights.resolve()),
            "--output", str(output.resolve()), "--conditions", str(args.conditions.resolve()),
            "--log", str(args.log.resolve()), "--seed", str(args.seed),
            "--width", str(args.width), "--height", str(args.height), "--fps", str(args.fps),
            "--num-frames", str(args.num_frames), "--steps", str(args.steps),
            "--guidance", str(args.guidance), "--conditioning-scale", str(args.conditioning_scale),
            "--flow-shift", str(args.flow_shift), "--offload", args.offload,
            "--segment-index", str(index)]


def build_attempt(args):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Load ffmpeg and ffprobe on PATH before preflight")
    conditions = json.loads(args.conditions.read_text(encoding="utf-8"))
    model = conditions["model"]
    if model.get("settings", {}).get("execution_mode") != "segmented":
        raise ValueError("Predeclare the segmented plan in genai_conditions.json before running")
    work = args.output.parent / "segmented_attempt"
    plans, outputs = [], []
    for index in range(len(model["segments"])):
        child = deepcopy(args)
        child.segment_index = index
        child.output = work / f"segment_{index:02d}" / "raw_video.mp4"
        plan = core.make_plan(child, allow_existing_output=True)
        plans.append(plan)
        outputs.append(child.output)
    # To keep every invocation's settings fixed, durations/counts must agree.
    if len({(p["num_frames"], p["segment"]["keep_frames"]) for p in plans}) != 1:
        raise ValueError("This runner requires identical generated/kept frame counts for every segment")
    frames = conditions["frames"]
    chosen = {frames["clean"], frames["occluded"], frames["perspective"], *frames["pair"]}
    mapping = audit_mapping([p["segment"] for p in plans], chosen,
                            plans[0]["source"]["fps"], args.fps, frames["pair"])
    # Hash all actual conditioning masks, including each segment's padding input.
    sampled = sorted({r["source_frame"] for p in plans for r in p["frame_mapping"]})
    mask_hashes = {str(f): core.digest(args.masks / f"mask_{f:06d}.png") for f in sampled}
    signature = {"plans": plans, "conditioning_mask_sha256": mask_hashes}
    code_sha = {name: core.digest(Path(__file__).with_name(name))
                for name in ("infer.py", "segment_timing.py", "segmented_infer.py")}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, "model": core.MODEL, "seed": args.seed,
            "segments": plans, "audit_output_frames": mapping,
            "joined_frames": sum(p["segment"]["keep_frames"] for p in plans),
            "output_fps": args.fps, "width": args.width, "height": args.height,
            "duration_seconds": sum(p["segment"]["keep_frames"] for p in plans) / args.fps,
            "signature": signature, "code_sha256": code_sha}, outputs, work


def validate_raw(path, plan):
    info = core.probe(path)
    expected = (plan["num_frames"], plan["fps"], plan["width"], plan["height"])
    actual = (info["frames"], info["fps"], info["width"], info["height"])
    if actual != expected:
        raise ValueError(f"Unexpected raw segment metadata at {path}: {actual}, expected {expected}")
    return info


def recover_completed_child(output, expected, log_path):
    """Recover a verified child completed just before its parent was interrupted."""
    canonical_expected = json.loads(json.dumps(expected))
    for path in sorted((output.parent / "runs").glob("*/invocation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        actual = record.get("plan", {})
        if any(actual.get(key) != value for key, value in canonical_expected.items()):
            continue
        if actual.get("output_sha256") != core.digest(output):
            continue
        validate_raw(output, expected)
        row = record.get("log_row")
        if not row:
            continue  # Do not silently adopt an unlogged result.
        invocation_id = json.loads(row["notes"])["invocation_id"]
        core.update_log(log_path, invocation_id, row)
        return path
    raise FileExistsError(f"Unconfirmed raw output {output}; inspect child logs before recovery. "
                          "It will not be overwritten or regenerated.")


def execute(args, attempt, outputs, work):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run one Python process in a single-GPU allocation, not torchrun")
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "attempt.json"
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("An attempt already exists. Use --resume only for an unchanged-settings technical restart.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != attempt["fingerprint"]:
            raise ValueError("Resume rejected: source, artwork, masks, conditions, weights revision or settings changed")
        if args.output.exists() and state.get("output_sha256"):
            if core.digest(args.output) != state["output_sha256"]:
                raise ValueError("The completed joined output has changed")
            state["status"] = "complete"
            core.save_json(state_path, state)
            print("Attempt already complete; no generation repeated.")
            return
    else:
        if args.resume:
            raise FileNotFoundError("No existing attempt to resume")
        if args.output.exists():
            raise FileExistsError(args.output)
        state = {"attempt_id": uuid.uuid4().hex, "fingerprint": attempt["fingerprint"],
                 "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending",
                 "plan": attempt, "segments": [{"index": i, "status": "pending"}
                                               for i in range(len(outputs))]}
        core.save_json(state_path, state)
    lock = args.output.parent / ".vace-segments.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"Active segmented attempt; pid={os.getpid()}. Do not remove while a parent/child job runs.\n")
    try:
        state["status"] = "running"
        if args.resume:
            state.setdefault("resume_history", []).append({"time_utc": datetime.now(timezone.utc).isoformat(),
                                                           "code_sha256": attempt.get("code_sha256", {})})
        core.save_json(state_path, state)
        for index, (plan, output) in enumerate(zip(attempt["segments"], outputs)):
            segment_state = state["segments"][index]
            if segment_state["status"] == "complete":
                if not output.exists() or core.digest(output) != segment_state["sha256"]:
                    raise ValueError(f"Completed segment {index} is missing or changed; it must not be regenerated")
                validate_raw(output, plan)
                print(f"Keeping completed segment {index + 1}; no generation repeated.", flush=True)
                continue
            if output.exists():
                manifest = recover_completed_child(output, plan, args.log)
                segment_state.update(status="complete", sha256=core.digest(output),
                                     raw_video=str(output.resolve()), recovered_child_manifest=str(manifest.resolve()))
                core.save_json(state_path, state)
                print(f"Recovered completed child for segment {index + 1}; no generation repeated.", flush=True)
                continue
            retried = segment_state["status"] in ("running", "failed")
            command = child_command(args, index, output)
            if retried:
                command.extend(["--retry-note", f"Technical resume of attempt {state['attempt_id']}, segment {index}; "
                                                "same inputs, model revision, settings and seed; raw attempts retained"])
            segment_state["status"] = "running"
            core.save_json(state_path, state)
            output.parent.mkdir(parents=True, exist_ok=True)
            console = output.parent / ("console_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                                       + "_" + uuid.uuid4().hex[:6] + ".log")
            print(f"Generating segment {index + 1}/{len(outputs)}: source "
                  f"{plan['interval']['start']}..{plan['interval']['end']}; log {console}", flush=True)
            with console.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                segment_state.update(status="failed", console_log=str(console), returncode=result.returncode)
                core.save_json(state_path, state)
                raise RuntimeError(f"Segment {index + 1} failed; no auto-retry/settings change. Read {console}")
            validate_raw(output, plan)
            segment_state.update(status="complete", sha256=core.digest(output),
                                 raw_video=str(output.resolve()), console_log=str(console.resolve()))
            core.save_json(state_path, state)
        if args.output.exists():
            raise FileExistsError(args.output)
        temporary = work / "joining.mp4"
        if temporary.exists():
            # This is only the interrupted derived join, never a raw generation.
            temporary.unlink()
        state["status"] = "joining"
        core.save_json(state_path, state)
        items = [{"path": path, "keep_frames": p["segment"]["keep_frames"]}
                 for path, p in zip(outputs, attempt["segments"])]
        result = join_segments(items, temporary, args.fps, shutil.which("ffmpeg"))
        (work / "join_ffmpeg.log").write_text(result.stderr, encoding="utf-8")
        info = core.probe(temporary)
        if (info["frames"], info["fps"], info["width"], info["height"]) != (
                attempt["joined_frames"], args.fps, args.width, args.height):
            raise ValueError("Joined output timing/dimensions do not match the predeclared plan")
        created = False
        try:
            with args.output.open("xb") as destination:
                created = True
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, destination)
        except BaseException:
            if created:
                args.output.unlink(missing_ok=True)  # only our incomplete publication
            raise
        state.update(status="complete", output_sha256=core.digest(args.output), output=str(args.output.resolve()),
                     joined_info=info, completed_at_utc=datetime.now(timezone.utc).isoformat())
        core.save_json(state_path, state)
        temporary.unlink()
        print(f"[AI-GENERATION] Complete: {args.output}; "
              f"{info['frames']} frames / {info['fps']} fps = {info['frames']/info['fps']:.3f}s")
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        core.save_json(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = core.build_parser()
    parser.description = __doc__
    parser.set_defaults(width=832, height=480, num_frames=81, flow_shift=3.0, offload="sequential")
    parser.add_argument("--resume", action="store_true", help="Explicit unchanged-settings technical resume")
    args = parser.parse_args()
    if args.segment_index is not None:
        raise ValueError("The segmented runner processes the full plan, not a user-selected segment")
    attempt, outputs, work = build_attempt(args)
    print(json.dumps({k: v for k, v in attempt.items() if k not in ("signature", "segments")}, indent=2))
    for plan in attempt["segments"]:
        print("Segment:", plan["segment"])
    if args.action == "preflight":
        print("Preflight complete. No model loaded, no invocation or video generated.")
    else:
        execute(args, attempt, outputs, work)


if __name__ == "__main__":
    main()
