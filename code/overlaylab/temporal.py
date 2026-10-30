"""Stage 2 Track: optical flow, temporal variants, and failure monitoring.

Implement the functions marked ``YOUR CODE HERE``. ``cv2.calcOpticalFlowPyrLK``
and ``cv2.goodFeaturesToTrack`` may appear only in the labelled comparison
cell of Notebook 2. Grayscale images are float32 in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from .supplied import to_gray, sobel_gradients, gaussian_pyramid, quad_mask, rect_corners
from .geometry import ransac_homography, transfer_points, bilinear_sample
from .localisation import localise_board

OK, REINIT, HIDE = "OK", "REINIT", "HIDE"


def lk_flow(prev_gray: np.ndarray, next_gray: np.ndarray, points: np.ndarray, config: FlowConfig
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative single-level Lucas-Kanade at sparse points.

    For each point use a (2w+1) x (2w+1) window (``config.window``), the
    supplied Sobel gradients of ``prev_gray``, bilinear sampling of the warped
    window in ``next_gray`` at every iteration, and at most
    ``config.iterations`` updates (stop early when the update is below
    ``config.convergence``). The structure tensor is normalised by the number
    of window pixels so that ``config.min_eigenvalue`` is scale-free.

    Returns (displacements N x 2, eigenvalues N x 2 with lambda_1 >= lambda_2,
    residuals N (mean absolute brightness residual), valid N bool). A point is
    invalid when lambda_2 < config.min_eigenvalue, the residual exceeds
    config.max_residual, the update failed to converge, or the point leaves
    the image. Coarse-to-fine estimation over ``config.pyramid_levels`` is an
    optional extension.
    """
    # YOUR CODE HERE
    prev_gray = np.asarray(prev_gray, dtype=np.float32)
    next_gray = np.asarray(next_gray, dtype=np.float32)
    points = np.asarray(points, dtype=np.float64)

    if prev_gray.ndim != 2 or next_gray.ndim != 2:
        raise ValueError("Input images must be grayscale")
    if prev_gray.shape != next_gray.shape:
        raise ValueError("Input images must have the same shape")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    height, width = prev_gray.shape
    n_points = len(points)

    displacements = np.zeros((n_points, 2), dtype=np.float64)
    eigenvalues = np.zeros((n_points, 2), dtype=np.float64)
    residuals = np.full(n_points, np.inf, dtype=np.float64)
    valid = np.zeros(n_points, dtype=bool)

    gx, gy, _, _ = sobel_gradients(prev_gray)

    radius = config.window
    offset_y, offset_x = np.mgrid[
        -radius:radius + 1,
        -radius:radius + 1,
    ]
    sample_count = offset_x.size

    def inside_image(xs, ys):
        return (
            np.all(np.isfinite(xs))
            and np.all(np.isfinite(ys))
            and np.all((xs >= 0) & (xs <= width - 1))
            and np.all((ys >= 0) & (ys <= height - 1))
        )
    for i, (x, y) in enumerate(points):
        # Coordinates of the template window.
        xs = x + offset_x
        ys = y + offset_y

        if not inside_image(xs, ys):
            continue

        template = bilinear_sample(prev_gray, ys, xs).ravel()

        ix = bilinear_sample(gx, ys, xs).ravel()
        iy = bilinear_sample(gy, ys, xs).ravel()

        # One row per window pixel: [Ix, Iy].
        gradients = np.column_stack((ix, iy))

        # Normalised 2 x 2 structure tensor.
        tensor = (gradients.T @ gradients) / sample_count

        # eigvalsh returns ascending order; the contract needs descending.
        eig = np.linalg.eigvalsh(tensor)[::-1]
        eigenvalues[i] = eig

        # Flat regions or an aperture-problem window cannot constrain 2-D flow.
        if eig[1] < config.min_eigenvalue or eig[1] <= 0:
            continue

        displacement = np.zeros(2, dtype=np.float64)
        converged = False

        for _ in range(config.iterations):
            warped_x = xs + displacement[0]
            warped_y = ys + displacement[1]

            if not inside_image(warped_x, warped_y):
                break

            current = bilinear_sample(
                next_gray, warped_y, warped_x
            ).ravel()

            error = template - current
            rhs = (gradients.T @ error) / sample_count

            try:
                update = np.linalg.solve(tensor, rhs)
            except np.linalg.LinAlgError:
                break

            if not np.all(np.isfinite(update)):
                break

            displacement += update

            if np.linalg.norm(update) < config.convergence:
                converged = True
                break

        displacements[i] = displacement

        # Recompute the residual at the final displacement.
        final_x = xs + displacement[0]
        final_y = ys + displacement[1]

        if not inside_image(final_x, final_y):
            continue

        final_window = bilinear_sample(
            next_gray, final_y, final_x
        ).ravel()

        residuals[i] = np.mean(np.abs(template - final_window))

        valid[i] = (
            converged
            and np.isfinite(residuals[i])
            and residuals[i] <= config.max_residual
        )

    return displacements, eigenvalues, residuals, valid


@dataclass
class TrackInit:
    """What ``track_board`` needs to (re)initialise from the reference image."""
    reference: np.ndarray
    localisation: LocalisationConfig


@dataclass
class TrackResult:
    homographies: List[Optional[np.ndarray]]
    corners: List[np.ndarray]                 # 4 x 2 per frame (NaN when lost)
    status: List[str]                          # "ok" or "lost"
    stats: List[Dict]                          # consumed by failure_monitor
    reinit_frames: List[int] = field(default_factory=list)
    flow: List[Dict] = field(default_factory=list)  # per-frame vectors, eigenvalues, validity


def _frame_stats(prev_corners, corners, gray_prev, gray, inlier_count, inlier_ratio,
                 median_residual, mean_flow, n_valid, status, n_matches=None):
    """Supplied helper: assemble the per-frame statistics dictionary (``n_matches`` on (re)initialised frames)."""
    disp = float(np.mean(np.linalg.norm(corners - prev_corners, axis=1))) if (
        prev_corners is not None and np.all(np.isfinite(corners)) and np.all(np.isfinite(prev_corners))) else float("nan")
    if gray_prev is not None and np.all(np.isfinite(corners)):
        m = quad_mask(gray.shape, corners)
        bright = float(np.mean(np.abs(gray[m] - gray_prev[m]))) if m.any() else float("nan")
    else:
        bright = float("nan")
    return {"inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
            "median_residual": float(median_residual), "corner_displacement": disp,
            "mean_flow": float(mean_flow), "brightness_change": bright,
            "n_valid_points": int(n_valid), "status": status,
            "n_matches": (int(n_matches) if n_matches is not None else None)}


def track_board(frames: Sequence[np.ndarray], init: TrackInit, config: TrackConfig) -> TrackResult:
    """Track the board through a sequence with one of two variants.

    ``config.variant == "independent"``: run ``localise_board`` on every frame
    from the reference image.

    ``config.variant == "flow_assisted"``: localise frame 0, then propagate the
    inlier correspondences (or the corners, ``config.propagate``) to the next
    frame with ``lk_flow``, drop invalid points, re-estimate the homography
    with RANSAC between the reference points and the propagated points, and
    reinitialise from the reference image when fewer than
    ``config.reinit_min_points`` valid points remain, every
    ``config.reinit_every`` frames (0 disables), or when RANSAC fails.

    Optional corner propagation has four points only: use reinit_min_points <= 4,
    RANSAC min_inliers = 4 and monitor min_inlier_count <= 4 if selecting it.

    Both variants return a ``TrackResult`` with a homography, projected
    corners, a status flag and the statistics the failure monitor consumes
    for every frame (build them with the supplied ``_frame_stats``; use
    ``inlier_ratio`` = inliers / valid propagated points for the flow variant
    and ``median_residual`` = median over inliers of sqrt(e_i / 2) px). The
    flow-assisted variant also stores, for every frame after the first, a dict
    in ``TrackResult.flow[t]`` with keys ``points`` (tracked points N x 2),
    ``vectors`` (displacements N x 2), ``eigenvalues`` (N x 2), ``residuals``
    (N) and ``valid`` (N bool); frames that were (re)initialised store {}.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


def failure_monitor(stats: Sequence[Dict], config: MonitorConfig) -> List[Tuple[str, str]]:
    """Decide OK / REINIT / HIDE for every frame from the per-frame statistics.

    ``stats`` is the list produced by ``track_board``: dictionaries with
    ``inlier_count``, ``inlier_ratio``, ``median_residual``,
    ``corner_displacement``, ``mean_flow``, ``brightness_change`` and
    ``status``. Returns a list of (flag, trigger) where ``trigger`` names the
    statistic (or "none") responsible for the decision. Because a homography
    is defined only up to scale, compare projected-corner displacement and
    reprojection error, never homography matrices.

    Required policy (the tests check it; the thresholds are yours to set):
    1. ``status != "ok"`` -> HIDE with trigger "no_model";
    2. ``inlier_count`` < min_inlier_count or ``inlier_ratio`` < min_inlier_ratio
       -> REINIT with trigger "inlier_count" / "inlier_ratio", escalating to
       HIDE once this has happened on ``hide_after_consecutive`` consecutive
       frames (frames under rule 1 count towards the run);
    3. otherwise the first of ``median_residual``, ``corner_displacement``,
       ``mean_flow``, ``brightness_change`` that exceeds its maximum -> REINIT
       with that statistic as trigger (NaN statistics never fire);
    4. otherwise OK with trigger "none"; an OK frame resets the run.
    """
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
