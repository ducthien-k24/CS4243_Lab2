"""Exact source/output timing for predeclared VACE segments. [AI-CODE]"""
from fractions import Fraction


def validate_segments(segments, start, end, source_fps, output_fps):
    """Require contiguous intervals and an integral output grid at every seam.

    Each model emits 4n+1 frames. Retain only timestamps in the segment's
    half-open source interval, discarding predeclared VAE tail padding.
    """
    source_fps = Fraction(str(source_fps))
    output_fps = Fraction(str(output_fps))
    if source_fps <= 0 or output_fps <= 0 or not segments:
        raise ValueError("Positive FPS and a nonempty segmentation plan are required")
    cursor, output_cursor = start, 0
    result = []
    for index, item in enumerate(segments):
        a, b = item["start"], item["end"]
        if type(a) is not int or type(b) is not int or a != cursor or b < a or b > end:
            raise ValueError("Segments must cover the declared source interval once, in order, without gaps/overlaps")
        duration = Fraction(b - a + 1, 1) / source_fps
        kept = duration * output_fps
        if kept.denominator != 1:
            raise ValueError("Segment boundaries must align exactly to the output FPS grid; do not round seam times")
        kept = int(kept)
        # Ceil to the smallest 4n+1 count that covers all retained timestamps.
        generated = ((kept - 1 + 3) // 4) * 4 + 1
        if kept < 1 or item.get("num_frames") != generated or item.get("keep_frames") != kept:
            raise ValueError("Predeclared generated/kept frame counts do not match the interval duration")
        result.append({"index": index, "start": a, "end": b, "num_frames": generated,
                       "keep_frames": kept, "discard_tail_frames": generated - kept,
                       "output_start": output_cursor, "output_end": output_cursor + kept - 1})
        output_cursor += kept
        cursor = b + 1
    if cursor != end + 1:
        raise ValueError("Segmentation does not cover the full declared source interval")
    return result


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
