"""File, timestamp and segment helpers for MiniMax inference. [AI-CODE]"""


from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


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


def audit_mapping(segments, source_frames, source_fps, output_fps, pair):
    """Map using segment-local source times; assert agreement with joined time."""
    rate = Fraction(str(output_fps)) / Fraction(str(source_fps))
    start = segments[0]["start"]
    mapping, owner = {}, {}
    for frame in source_frames:
        segment = next((s for s in segments if s["start"] <= frame <= s["end"]), None)
        if segment is None:
            raise ValueError("An audit frame is not covered by a segment")
        local = round(Fraction(frame - segment["start"]) * rate)
        if not 0 <= local < segment["keep_frames"]:
            raise ValueError("An audit timestamp rounds onto a discarded tail; choose boundaries before generation")
        joined = segment["output_start"] + local
        if joined != round(Fraction(frame - start) * rate):
            raise ValueError("Segment-local rounding differs from the notebook's global timestamp mapping")
        mapping[frame], owner[frame] = joined, segment["index"]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Distinct selected source frames collide at the output FPS")
    if owner[pair[0]] != owner[pair[1]]:
        raise ValueError("The temporal pair must stay within one generated segment")
    return mapping


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
