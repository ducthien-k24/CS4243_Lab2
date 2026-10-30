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
    raise NotImplementedError("YOUR CODE HERE")


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
