"""Stage 3 Composite: occlusion handling and compositing.

Implement the functions marked ``YOUR CODE HERE``. Motion-compensated
differencing is the required occlusion signal; ``mode="colour"`` is an
optional ablation. Permitted library calls: ``cv2.morphologyEx``,
``cv2.connectedComponentsWithStats``, ``cv2.distanceTransform`` (feathering),
and your Lab 1 ``hysteresis``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import OcclusionConfig, CompositeConfig
from .supplied import to_float
from .geometry import warp_artwork
from . import metrics


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels reachable from strong pixels (reuse your Lab 1 implementation).

    ``strong`` and ``weak`` are Boolean masks of the same shape (weak includes
    strong). Traverse with the 4- or 8-connected neighbourhood from every
    strong pixel, through chains of weak pixels, and return the Boolean mask
    of reached pixels. Validate shapes and the connectivity value.
    """
    # YOUR CODE HERE
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)

    if strong.ndim != 2 or weak.ndim != 2:
        raise ValueError("Masks must be 2-D")
    if strong.shape != weak.shape:
        raise ValueError("Masks must have the same shape")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    height, width = strong.shape
    result = strong.copy()

    # Every strong pixel is a starting point.
    ys, xs = np.nonzero(strong)
    stack = list(zip(ys, xs))

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    while stack:
        y, x = stack.pop()

        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx

            if not (0 <= ny < height and 0 <= nx < width):
                continue

            if weak[ny, nx] and not result[ny, nx]:
                # Mark before pushing to avoid duplicate visits.
                result[ny, nx] = True
                stack.append((ny, nx))

    return result

    


def compensated_difference(frame: np.ndarray, warped_reference: np.ndarray, coverage: np.ndarray,
                           config: OcclusionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per-pixel difference after a global gain/offset fit, zero outside ``coverage``.

    Fit I_frame ~ a * I_ref + b on the ``config.gain_quantile`` fraction of
    covered pixels whose raw difference is smallest (so that illumination
    change alone does not become an occluder), then return the mean absolute
    difference over the colour channels between the frame and the compensated
    reference, together with (a, b). Use colour, not grayscale alone: a
    coloured occluder over a coloured board can have almost the same luminance
    (the ball over the blue board differs by only ~0.13 in grayscale but ~0.3
    in mean channel difference), and the default thresholds assume colour.
    """
    # [AI-CODE] Fit one shared colour gain/offset on the least-changing pixels.
    observed = to_float(frame)
    reference = to_float(warped_reference)
    coverage = np.asarray(coverage, dtype=bool)
    if observed.shape != reference.shape or observed.shape[:2] != coverage.shape:
        raise ValueError("Images must match and coverage must have their spatial shape")
    if not 0 < config.gain_quantile <= 1:
        raise ValueError("gain_quantile must be in (0, 1]")

    difference = np.zeros(coverage.shape, dtype=np.float32)
    if not coverage.any():
        return difference, (1.0, 0.0)
    raw = np.abs(observed - reference)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    cutoff = np.quantile(raw[coverage], config.gain_quantile)
    agreeing = coverage & (raw <= cutoff)
    x = reference[agreeing].astype(np.float64).ravel()
    y = observed[agreeing].astype(np.float64).ravel()
    xc, yc = x - x.mean(), y - y.mean()
    variance = np.dot(xc, xc)
    # A constant reference cannot separate gain from offset. Keep unit gain
    # and absorb the mean change into the offset in that degenerate case.
    gain = float(np.dot(xc, yc) / variance) if variance > 1e-12 * x.size else 1.0
    offset = float(y.mean() - gain * x.mean())
    error = np.abs(observed - (gain * reference + offset))
    if error.ndim == 3:
        error = error.mean(axis=2)
    difference[coverage] = error[coverage]
    return difference, (gain, offset)


def _mask_from_difference(difference: np.ndarray, coverage: np.ndarray,
                          config: OcclusionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Shared mask construction for inference and validation search. [AI-CODE]"""
    if not 0 <= config.weak <= config.strong <= 1:
        raise ValueError("Require 0 <= weak <= strong <= 1")
    radius = int(config.open_radius)
    if radius < 0 or radius != config.open_radius or config.min_area < 0:
        raise ValueError("Opening radius must be a nonnegative integer; min_area >= 0")
    strong = coverage & (difference > config.strong)
    weak = coverage & (difference > config.weak)
    mask = hysteresis(strong, weak, config.connectivity)
    if radius:
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disc = (xx * xx + yy * yy <= radius * radius).astype(np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disc,
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    mask &= coverage
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=config.connectivity)
    kept = np.flatnonzero(stats[1:count, cv2.CC_STAT_AREA] >= config.min_area) + 1
    lookup = np.zeros(count, dtype=bool)
    lookup[kept] = True
    components = np.column_stack((kept, stats[kept])).astype(np.int32)
    return lookup[labels], components


def occlusion_mask(frame: np.ndarray, H: np.ndarray, reference: np.ndarray, config: OcclusionConfig
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground mask inside the board by motion-compensated differencing.

    1. Warp ``reference`` into the frame with ``H`` (``warp_artwork``) and
       compute the ``compensated_difference`` inside the coverage mask.
    2. Threshold at ``config.strong`` and ``config.weak`` and apply
       ``hysteresis`` with ``config.connectivity``.
    3. Open with a disc of radius ``config.open_radius`` (``cv2.morphologyEx``:
       erosion then dilation, which deletes anything thinner than 2r+1 pixels;
       0 disables) and keep connected components with at least
       ``config.min_area`` pixels.
    4. Return (mask bool H x W, difference map float H x W, component stats).

    ``component stats`` is an array with one row per kept component:
    [label, x, y, width, height, area]. With ``config.mode == "colour"`` an
    appearance model may be used instead (optional ablation).
    """
    # [AI-CODE] Motion compensation precedes photometric compensation.
    if config.mode != "difference":
        raise NotImplementedError("Only the required difference mode is implemented")
    warped_reference, coverage = warp_artwork(to_float(reference), H, frame.shape)
    difference, _ = compensated_difference(frame, warped_reference, coverage, config)
    mask, components = _mask_from_difference(difference, coverage, config)
    return mask, difference, components


def select_difference_thresholds(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 masks: Sequence[np.ndarray], config: OcclusionConfig
                                 ) -> Tuple[OcclusionConfig, List[Dict]]:
    """Choose strong/weak thresholds and the minimum area on validation masks only.

    ``frames`` is a sequence of (frame, H, reference) triples and ``masks`` the
    matching ground-truth occlusion masks. Grid-search the thresholds and the
    minimum area, score each setting by mean ``metrics.mask_iou`` over the
    validation frames, and return (frozen config, search log) where the log
    holds one dict per setting with its parameters and score.

    Several settings usually tie at the best score: the few validation masks
    do not separate the strong threshold. Break ties towards the middle of the
    tied range of ``strong`` values (not the first or the largest grid point),
    so that the frozen value sits inside the plateau rather than on its edge,
    and state the rule in the report.

    The grid scales the input strong by (0.6, 0.8, 1, 1.2, 1.4), weak by
    (0.5, 0.75, 1, 1.25), and min_area by (0.5, 1, 2). Undefined IoUs
    (empty prediction and empty truth) remain NaN and are excluded from the
    mean; ``n_scored`` records its denominator. No finite scores raises
    ValueError. The input config is never changed. [AI-CODE]
    """
    # [AI-CODE] Only the caller-supplied validation examples enter this search.
    frames, masks = list(frames), list(masks)
    if not frames or len(frames) != len(masks):
        raise ValueError("Provide a nonempty sequence of frames and matching masks")
    if config.mode != "difference":
        raise NotImplementedError("Threshold selection requires difference mode")
    cached = []
    for (frame, H, reference), gt in zip(frames, masks):
        gt = np.asarray(gt, dtype=bool)
        if gt.shape != frame.shape[:2]:
            raise ValueError("Each validation mask must match its frame")
        warped, coverage = warp_artwork(to_float(reference), H, frame.shape)
        difference, _ = compensated_difference(frame, warped, coverage, config)
        cached.append((difference, coverage, gt))

    # Multipliers retain the input config's processing scale (notably min_area).
    strong_values = sorted(set(float(np.clip(config.strong * s, 0, 1))
                               for s in (0.6, 0.8, 1.0, 1.2, 1.4)))
    weak_values = sorted(set(float(np.clip(config.weak * s, 0, 1))
                             for s in (0.5, 0.75, 1.0, 1.25)))
    areas = sorted(set(max(1, int(round(config.min_area * s))) for s in (0.5, 1.0, 2.0)))
    log = []
    for strong in strong_values:
        for weak in weak_values:
            if weak >= strong:
                continue
            for area in areas:
                candidate = replace(config, strong=strong, weak=weak, min_area=area)
                scores = [metrics.mask_iou(_mask_from_difference(d, c, candidate)[0], gt)
                          for d, c, gt in cached]
                finite_scores = [score for score in scores if np.isfinite(score)]
                log.append({"strong": strong, "weak": weak, "min_area": area,
                            "mean_iou": float(np.mean(finite_scores)) if finite_scores else float("nan"),
                            "n_scored": len(finite_scores)})
    if not log:
        raise ValueError("Input thresholds produce no valid weak < strong search settings")
    scored = [row for row in log if np.isfinite(row["mean_iou"])]
    if not scored:
        raise ValueError("No finite validation IoU; supply masks containing an occluder")
    best_score = max(row["mean_iou"] for row in scored)
    tied = [row for row in log if np.isclose(row["mean_iou"], best_score,
                                           rtol=0, atol=1e-12)]
    middle = (min(row["strong"] for row in tied) + max(row["strong"] for row in tied)) / 2
    # Prefer the centre of the tied strong range; remaining ties prefer values
    # closest to the input config, then smaller values for deterministic output.
    best = min(tied, key=lambda row: (abs(row["strong"] - middle),
                                     abs(row["weak"] - config.weak),
                                     abs(row["min_area"] - config.min_area),
                                     row["strong"], row["weak"], row["min_area"]))
    return replace(config, strong=best["strong"], weak=best["weak"],
                   min_area=best["min_area"]), log


def composite(frame: np.ndarray, warped: np.ndarray, coverage: np.ndarray, occlusion: np.ndarray,
              config: CompositeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the warped artwork into the frame.

    alpha = C * (1 - O) * F, where C is the coverage mask, O the occlusion
    mask and F an optional feathering term that softens the quadrilateral and
    occluder boundaries over ``config.feather`` pixels (0 disables).
    I_out = alpha * W + (1 - alpha) * I. Optional brightness matching of the
    artwork to the board is controlled by ``config.match_gain``.
    Returns (composited RGB float32 frame, alpha map float32 in [0, 1]).
    """
    # [AI-CODE] Feather inward so foreground and pixels outside coverage stay intact.
    frame, warped = to_float(frame), to_float(warped)
    coverage, occlusion = np.asarray(coverage, bool), np.asarray(occlusion, bool)
    if (frame.shape != warped.shape or coverage.shape != frame.shape[:2]
            or occlusion.shape != coverage.shape):
        raise ValueError("Images and spatial masks must have matching shapes")
    if config.feather < 0:
        raise ValueError("feather must be nonnegative")
    visible = coverage & ~occlusion
    alpha = visible.astype(np.float32)
    if config.feather > 0:
        # Padding supplies an outside boundary even when coverage fills the frame.
        padded = np.pad(visible.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        alpha *= np.clip(distance / config.feather, 0, 1)
    if config.match_gain and visible.any():
        mean_artwork = float(warped[visible].mean())
        if mean_artwork > 1e-6:
            gain = float(frame[visible].mean()) / mean_artwork
            warped = np.clip(warped * gain, 0, 1)
    weight = alpha[..., None] if frame.ndim == 3 else alpha
    out = weight * warped + (1 - weight) * frame
    return out.astype(np.float32), alpha.astype(np.float32)


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
