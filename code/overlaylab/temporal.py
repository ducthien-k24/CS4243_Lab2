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
    raise NotImplementedError("YOUR CODE HERE")


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
    if config.variant not in ("independent", "flow_assisted"):
        raise ValueError(f"unknown variant: {config.variant!r}")
    if config.propagate not in ("inliers", "corners"):
        raise ValueError(f"unknown propagate mode: {config.propagate!r}")
    ref = init.reference
    loc = init.localisation
    ref_rect = rect_corners(ref.shape[1], ref.shape[0])
    nan4 = np.full((4, 2), np.nan)
    res = TrackResult(homographies=[], corners=[], status=[], stats=[])

    def _median_px(resid, mask):
        return float(np.median(np.sqrt(resid[mask] / 2.0))) if np.any(mask) else float("nan")

    prev_gray, prev_corners = None, None
    ref_pts = cur_pts = None          # flow_assisted: điểm trên reference và vị trí hiện tại trong frame
    since_init = 0                    # số frame kể từ lần (re)init gần nhất
    need_init = True

    for t, frame in enumerate(frames):
        gray = to_gray(frame)

        def _initialise():
            """localise_board từ reference; trả về (H, corners, status, stats)."""
            nonlocal ref_pts, cur_pts, since_init, need_init
            H, corners, pairs, inl, diag = localise_board(ref, frame, loc)
            res.reinit_frames.append(t)
            ok = bool(diag["status"])
            if ok and config.variant == "flow_assisted":
                if config.propagate == "inliers":
                    ref_pts = diag["keypoints_ref"][pairs[inl, 0]]
                    cur_pts = diag["keypoints_frame"][pairs[inl, 1]]
                else:                                   # "corners": chỉ 4 góc
                    ref_pts, cur_pts = ref_rect.copy(), np.asarray(corners, np.float64)
                since_init, need_init = 0, False
            else:
                need_init = True
            st = _frame_stats(prev_corners, corners if ok else nan4, prev_gray, gray,
                              diag["n_inliers"], diag["inlier_ratio"], diag["median_residual"],
                              float("nan"), diag["n_matches"] if config.variant == "independent" else
                              (len(cur_pts) if ok else 0), "ok" if ok else "lost", n_matches=diag["n_matches"])
            return (H if ok else None), (corners if ok else nan4), ("ok" if ok else "lost"), st

        if config.variant == "independent" or need_init:
            H, corners, status, st = _initialise()
            res.flow.append({})
        else:
            # 1) lan truyền điểm sang frame mới bằng Lucas-Kanade
            d, eig, resid_f, valid = lk_flow(prev_gray, gray, cur_pts, config.flow)
            new_pts = cur_pts + d
            n_valid = int(valid.sum())
            since_init += 1
            redo = (n_valid < config.reinit_min_points or
                    (config.reinit_every > 0 and since_init >= config.reinit_every))
            if not redo:
                # 2) RANSAC giữa điểm reference và điểm đã lan truyền (chỉ điểm hợp lệ)
                H, mask, e, _, ok = ransac_homography(ref_pts[valid], new_pts[valid], loc.ransac)
                redo = not ok or H is None
            if redo:
                # 3) reinit từ reference (ít điểm / đến hạn / RANSAC fail)
                H, corners, status, st = _initialise()
                res.flow.append({})
            else:
                corners = transfer_points(H, ref_rect)
                status = "ok"
                mean_flow = float(np.mean(np.linalg.norm(d[valid], axis=1))) if n_valid else float("nan")
                st = _frame_stats(prev_corners, corners, prev_gray, gray, int(mask.sum()),
                                  float(mask.sum() / max(n_valid, 1)), _median_px(e, mask),
                                  mean_flow, n_valid, status)
                res.flow.append({"points": cur_pts.copy(), "vectors": d, "eigenvalues": eig,
                                 "residuals": resid_f, "valid": valid})
                # chỉ giữ các điểm hợp lệ và là inlier cho frame sau
                ref_pts = ref_pts[valid][mask]
                cur_pts = new_pts[valid][mask]

        res.homographies.append(H)
        res.corners.append(np.asarray(corners, dtype=np.float64))
        res.status.append(status)
        res.stats.append(st)
        prev_gray = gray
        prev_corners = corners if status == "ok" else None
    return res


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
    def _fires(value, limit, below=False):
        """So sánh an toàn: NaN / None không bao giờ kích hoạt."""
        if value is None:
            return False
        v = float(value)
        if not np.isfinite(v):
            return False
        return v < limit if below else v > limit

    out: List[Tuple[str, str]] = []
    run = 0                                          # số frame "có vấn đề" liên tiếp (luật 1 và 2)
    rule3 = [("median_residual", config.max_median_residual),
             ("corner_displacement", config.max_corner_displacement),
             ("mean_flow", config.max_mean_flow),
             ("brightness_change", config.max_brightness_change)]
    for s in stats:
        # luật 1: không có mô hình -> HIDE
        if s.get("status") != "ok":
            run += 1
            out.append((HIDE, "no_model"))
            continue
        # luật 2: thiếu inlier -> REINIT, lặp lại đủ lâu -> HIDE
        trig = None
        if _fires(s.get("inlier_count"), config.min_inlier_count, below=True):
            trig = "inlier_count"
        elif _fires(s.get("inlier_ratio"), config.min_inlier_ratio, below=True):
            trig = "inlier_ratio"
        if trig is not None:
            run += 1
            out.append((HIDE if run >= config.hide_after_consecutive else REINIT, trig))
            continue
        # luật 3: thống kê đầu tiên vượt ngưỡng -> REINIT
        trig = next((name for name, lim in rule3 if _fires(s.get(name), lim)), None)
        if trig is not None:
            out.append((REINIT, trig))
            continue
        # luật 4: OK, reset chuỗi
        run = 0
        out.append((OK, "none"))
    return out


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
