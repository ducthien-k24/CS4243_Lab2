"""Prepare the selected personal capture before audit annotation. [AI-CODE]

The reference-photo corners describe input preparation only, not audit labels
or tracking initialisation. Defaults document the user's selected capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from overlaylab.geometry import dlt_homography, warp_artwork
from overlaylab.supplied import rect_corners


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=CODE.parent / "data/video/video.mp4")
    parser.add_argument("--photo", type=Path, default=CODE.parent / "data/video/reference.jpg")
    parser.add_argument("--clip-id", default="poster_01")
    parser.add_argument("--start", type=float, default=0.3)
    parser.add_argument("--end", type=float, default=15.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--reference-width", type=int, default=320)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    if not 5 <= args.end - args.start <= 15:
        raise ValueError("The selected capture must last 5 to 15 seconds")
    if args.ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    else:
        ffmpeg = str(args.ffmpeg)
    destination = CODE.parent / "data/personal" / args.clip_id
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing/possibly annotated clip: {destination}")
    evidence = CODE / "student_files/personal_preparation" / args.clip_id
    evidence.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True)
    (destination / "masks").mkdir()
    duration = args.end - args.start
    expected_count = int(round(duration * args.fps))
    output_video = destination / "clip.mp4"
    command = [ffmpeg, "-hide_banner", "-nostdin", "-n", "-i", str(args.video),
               "-ss", f"{args.start:.6f}", "-t", f"{duration:.6f}",
               "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
               "-map_chapters", "-1", "-vf", f"fps={args.fps}", "-fps_mode", "cfr",
               "-frames:v", str(expected_count), "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-pix_fmt", "yuv420p", "-fflags", "+bitexact",
               "-flags:v", "+bitexact", "-metadata", "creation_time=",
               "-metadata", "encoder=", "-movflags", "+faststart", str(output_video)]
    process = subprocess.run(command, capture_output=True, text=True)
    (evidence / "ffmpeg_encode.log").write_text(process.stderr, encoding="utf-8")
    process.check_returncode()

    # [AI-VISION] The blue wall separates the near-frontal poster photograph.
    # Detect only the outer poster boundary; leave printed content and glare intact.
    photo_bgr = cv2.imread(str(args.photo))
    if photo_bgr is None:
        raise FileNotFoundError(args.photo)
    b, g, r = cv2.split(photo_bgr.astype(np.float32))
    blue_wall = (b > 1.30 * r) & (b > 1.17 * g) & (b > 50)
    contours, _ = cv2.findContours((~blue_wall).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if quad.shape != (4, 2):
        raise ValueError("Reference photo did not yield a four-corner poster; inspect it manually")
    # Order TL, TR, BR, BL for this portrait, near-frontal photograph.
    top = quad[np.argsort(quad[:, 1])[:2]]
    bottom = quad[np.argsort(quad[:, 1])[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    corners = np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
    horizontal = np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    vertical = np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    ref_width = args.reference_width
    ref_height = int(round(ref_width * vertical / horizontal))
    H, _, ok = dlt_homography(corners, rect_corners(ref_width, ref_height))
    if not ok:
        raise RuntimeError("Cannot rectify the input reference photo")
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    reference, coverage = warp_artwork(photo_rgb, H, (ref_height, ref_width, 3))
    if not coverage.all():
        raise RuntimeError("Rectified reference contains unsupported pixels")
    Image.fromarray(np.clip(np.rint(reference * 255), 0, 255).astype(np.uint8)).save(
        destination / "reference.png")
    preview = photo_bgr.copy()
    cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    for index, point in enumerate(corners.astype(int)):
        cv2.putText(preview, str(index + 1), tuple(point + (8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(evidence / "reference_source_corners.png"), preview)

    frames_dir = destination / "frames"
    frames_dir.mkdir()
    cap = cv2.VideoCapture(str(output_video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not cv2.imwrite(str(frames_dir / f"frame_{count:06d}.png"), frame):
                raise RuntimeError(f"Could not save frame {count}")
            count += 1
    finally:
        cap.release()
    if count != expected_count or abs(fps - args.fps) > 1e-6:
        raise RuntimeError(f"Unexpected normalised video: {count} frames at {fps} fps")
    metadata = subprocess.run([ffmpeg, "-v", "error", "-i", str(output_video),
                               "-f", "ffmetadata", "-"], check=True, capture_output=True, text=True).stdout
    (evidence / "output_metadata.txt").write_text(metadata, encoding="utf-8")
    record = {
        "clip_id": args.clip_id, "source_video": str(args.video.resolve()),
        "source_video_sha256": sha256(args.video), "source_photo_sha256": sha256(args.photo),
        "source_interval_seconds": [args.start, args.end], "source_interval_end_exclusive": True,
        "output": {"video": str(output_video.resolve()), "sha256": sha256(output_video),
                   "width": width, "height": height, "fps": fps,
                   "frames": count, "duration_seconds": count / fps},
        "reference": {"source_photo_corners_TL_TR_BR_BL": corners.tolist(),
                      "width": ref_width, "height": ref_height,
                      "sha256": sha256(destination / "reference.png"),
                      "homography_photo_to_reference": H.tolist(),
                      "aspect_note": "Estimated from the near-frontal photo edges; physical board dimensions unknown",
                      "scale_note": f"{ref_width} px approximates the board width visible near the start of the video"},
        "ffmpeg_command": command, "audit_annotations_created": False,
        "nominal_source_timestamp": f"original video time = {args.start} + normalised frame index / {args.fps}",
        "notes": "No model inference or overlay run. No glare removal or printed-content editing. "
                 "Reference corners are input preparation, not audit labels or pipeline initialisation.",
        "markers": ["[AI-CODE]", "[AI-VISION]"],
    }
    (evidence / "preparation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record["output"], indent=2))
    print(f"Reference: {ref_width}x{ref_height}; evidence: {evidence}")


if __name__ == "__main__":
    main()
