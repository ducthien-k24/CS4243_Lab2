"""Stage 1 Localise: board localisation from the reference image and single-frame overlay.

Implement the functions marked ``YOUR CODE HERE`` using your keypoint and
geometry branches. Optional feature families and alternative settings must be
controlled only through ``LocalisationConfig``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import LocalisationConfig
from .supplied import to_gray, rect_corners
from .keypoints import detect_corners, describe_keypoints, match_descriptors
from .geometry import ransac_homography, transfer_points, warp_artwork, rectangle_to_rectangle
from . import metrics


def localise_board(reference: np.ndarray, frame: np.ndarray, config: LocalisationConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Find the board in ``frame`` from the rectified ``reference`` image alone.

    Detect and describe keypoints in both images, match with the ratio test,
    estimate the reference-to-frame homography with RANSAC, and project the
    four reference corners (TL, TR, BR, BL) into the frame.

    Returns (H, corners 4 x 2, matches M x 2 (reference index, frame index),
    inlier_mask M, diagnostics). ``diagnostics`` must contain exactly these keys
    (the notebooks and the grading harness read them): ``n_keypoints_ref``,
    ``n_keypoints_frame``, ``n_matches``, ``n_inliers``, ``inlier_ratio``
    (inliers / ratio-test matches), ``median_residual``, ``iterations``,
    ``status``, ``residuals`` (the per-match symmetric transfer errors e_i from
    RANSAC, length M), ``keypoints_ref`` and ``keypoints_frame`` (the arrays the
    match indices refer to). ``median_residual`` is the median over inliers of
    sqrt(e_i / 2), i.e. the RMS one-way transfer error in pixels.
    When localisation fails, ``H`` is None, ``corners`` is NaN and ``status``
    is False.
    """
    # YOUR CODE HERE
    g_ref = to_gray(reference)
    g_frm = to_gray(frame)

    # 1) keypoint + descriptor cho cả hai ảnh (chỉ giữ keypoint có descriptor)
    k_ref, _ = detect_corners(g_ref, config.detector)
    d_ref, keep_ref = describe_keypoints(g_ref, k_ref, config.descriptor)
    k_ref = k_ref[keep_ref]
    k_frm, _ = detect_corners(g_frm, config.detector)
    d_frm, keep_frm = describe_keypoints(g_frm, k_frm, config.descriptor)
    k_frm = k_frm[keep_frm]

    # 2) ghép cặp bằng ratio test: pairs[:, 0] chỉ vào k_ref, pairs[:, 1] chỉ vào k_frm
    pairs, _, _ = match_descriptors(d_ref, d_frm, config.matcher)
    M = len(pairs)

    # 3) RANSAC homography reference -> frame
    if M >= 4:
        H, inliers, resid, iters, ok = ransac_homography(k_ref[pairs[:, 0]], k_frm[pairs[:, 1]], config.ransac)
    else:
        H, inliers, resid, iters, ok = None, np.zeros(M, dtype=bool), np.full(M, np.inf), 0, False
    ok = bool(ok and H is not None)

    # 4) chiếu 4 góc reference (TL, TR, BR, BL) vào frame
    if ok:
        corners = transfer_points(H, rect_corners(reference.shape[1], reference.shape[0]))
        if not np.all(np.isfinite(corners)):
            ok = False
    if not ok:
        H = None
        corners = np.full((4, 2), np.nan)

    n_in = int(np.sum(inliers))
    diag = {
        "n_keypoints_ref": int(len(k_ref)),
        "n_keypoints_frame": int(len(k_frm)),
        "n_matches": int(M),
        "n_inliers": n_in,
        "inlier_ratio": float(n_in / M) if M > 0 else float("nan"),
        "median_residual": float(np.median(np.sqrt(resid[inliers] / 2.0))) if n_in > 0 else float("nan"),
        "iterations": int(iters),
        "status": ok,
        "residuals": np.asarray(resid, dtype=np.float64),
        "keypoints_ref": k_ref,
        "keypoints_frame": k_frm,
    }
    return H, corners, pairs, np.asarray(inliers, dtype=bool), diag


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    # YOUR CODE HERE
    ah, aw = artwork.shape[:2]
    ref_shape = config.warp.reference_shape
    rh, rw = (ref_shape[0], ref_shape[1]) if ref_shape is not None else (ah, aw)
    # artwork pixel -> reference pixel (co giãn chữ nhật), rồi reference -> frame (H)
    S = rectangle_to_rectangle((aw, ah), (rw, rh))
    Ha = np.asarray(H, dtype=np.float64) @ S
    if abs(Ha[2, 2]) > 1e-12:
        Ha = Ha / Ha[2, 2]
    return Ha


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    # YOUR CODE HERE
    out = np.asarray(frame, dtype=np.float32).copy()
    if out.max() > 1.0:                                   # chấp nhận frame uint8
        out /= 255.0
    if H is None:
        return out
    art = np.asarray(artwork, dtype=np.float32)
    if art.max() > 1.0:
        art = art / 255.0
    Ha = artwork_to_frame(art, H, config)
    warped, cov = warp_artwork(art, Ha, out.shape)

    # alpha = coverage (tuỳ chọn: làm mềm mép bằng feather)
    alpha = cov.astype(np.float32)
    if config.warp.feather and config.warp.feather > 0:
        from .supplied import gaussian_smooth
        alpha = np.clip(gaussian_smooth(alpha, config.warp.feather / 2.0), 0, 1) * cov
    if config.warp.match_gain and cov.any():              # tuỳ chọn: khớp độ sáng trung bình
        gain = out[cov].mean() / max(float(warped[cov].mean()), 1e-6)
        warped = np.clip(warped * gain, 0, 1)
    if out.ndim == 3 and alpha.ndim == 2:
        alpha = alpha[..., None]
    return (alpha * warped + (1 - alpha) * out).astype(np.float32)


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    # YOUR CODE HERE
    per_frame, errors = {}, []
    for f in sorted(gt_corners):
        gt = np.asarray(gt_corners[f], dtype=np.float64)
        if f not in frames:                               # frame thiếu -> tính là thất bại
            per_frame[int(f)] = {"error": float("inf"), "n_matches": 0, "n_inliers": 0,
                                 "inlier_ratio": float("nan"), "median_residual": float("nan"),
                                 "status": False, "corners": np.full((4, 2), np.nan)}
            errors.append(float("inf"))
            continue
        H, corners, pairs, inliers, diag = localise_board(reference, frames[f], config)
        err = metrics.corner_error(corners, gt) if diag["status"] else float("inf")
        errors.append(err)
        per_frame[int(f)] = {"error": float(err), "n_matches": diag["n_matches"],
                             "n_inliers": diag["n_inliers"], "inlier_ratio": diag["inlier_ratio"],
                             "median_residual": diag["median_residual"], "status": diag["status"],
                             "corners": corners}
    return {"per_frame": per_frame, "summary": metrics.summarise_errors(errors, config.success_threshold)}


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
