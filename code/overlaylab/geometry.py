"""Stage 1 Localise, geometry branch.

Implement the functions marked ``YOUR CODE HERE``. Allowed: numpy (including
``numpy.linalg.svd`` and ``numpy.linalg.inv``) and your Lab 1
``bilinear_sample`` (pasted into this file). ``cv2.findHomography`` and ``cv2.warpPerspective``
may appear only in the labelled sanity-check and comparison cells of Notebook 1.

Conventions: points are N x 2 arrays of (x, y) with x indexing columns; a
homography H maps source homogeneous coordinates to destination ones,
x' ~ H x, and is scaled so that H[2, 2] = 1 whenever that entry is not close
to zero. Pixel centres sit at integer coordinates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import RansacConfig


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D or H x W x C image at floating-point image coordinates (reuse Lab 1).

    Same signature as Lab 1 (``y`` first, then ``x``); extend it so that a
    colour image returns an array with a trailing channel axis. Coordinates
    are clamped to the valid closed interval, and an exact integer coordinate
    reproduces the corresponding pixel. ``y`` and ``x`` may be arrays of any
    matching shape so that a whole grid can be sampled at once.
    """
    # YOUR CODE HERE
    height, width = image.shape[:2]

    y = np.clip(np.asarray(y, dtype=np.float64), 0, height - 1)
    x = np.clip(np.asarray(x, dtype=np.float64), 0, width - 1)

    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)

    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)

    top_left = image[y0,x0]
    top_right = image[y0, x1]
    bottom_left = image[y1,x0]
    bottom_right = image[y1,x1]

    a = x - x0
    b = y - y0

    if image.ndim == 3:
        a = a[..., None]
        b = b[..., None]

    result = (1-a)*(1-b)*top_left + a*(1-b)*top_right + (1-a)*b*bottom_left + a*b*bottom_right

    return result



def rectangle_to_rectangle(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """Supplied helper: affine scaling that maps a (w, h) pixel rectangle onto another.

    Corner (w-1, h-1) of the source maps to corner (w'-1, h'-1) of the
    destination (pixel-centre convention). Used to compose the artwork
    rectangle with the board-to-frame homography.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return np.array([[(dw - 1) / max(sw - 1, 1), 0, 0],
                     [0, (dh - 1) / max(sh - 1, 1), 0],
                     [0, 0, 1.0]], dtype=np.float64)


def normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation.

    Returns (T, normalised) where T is the 3 x 3 similarity that translates the
    points to zero centroid and scales them so that the mean distance from the
    origin is sqrt(2), and ``normalised`` are the transformed N x 2 points.
    """
    # YOUR CODE HERE
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)                                   # trọng tâm
    centred = pts - c
    mean_dist = np.linalg.norm(centred, axis=1).mean()     # khoảng cách TB tới trọng tâm
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0   # tránh chia 0 khi mọi điểm trùng nhau
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    return T, centred * s


def dlt_homography(src: np.ndarray, dst: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Normalised direct linear transform.

    Builds the 2n x 9 system A h = 0 from normalised correspondences, solves
    for h as the right singular vector with the smallest singular value,
    denormalises H = T'^{-1} H_hat T, and scales so that H[2, 2] = 1 when that
    entry is not close to zero.

    Raises ``ValueError`` for fewer than four correspondences. Returns
    (H, singular_values, status); when A is rank deficient (e.g. collinear
    source points) ``status`` is False and ``H`` is None rather than a
    numerically meaningless matrix. ``singular_values`` are those of A, so
    that conditioning can be inspected.
    """
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4 or len(dst) != n:
        raise ValueError(f"need >= 4 matching correspondences, got {n} and {len(dst)}")

    # 1) chuẩn hoá Hartley cả hai phía
    T_src, p = normalise_points(src)
    T_dst, q = normalise_points(dst)

    # 2) dựng hệ A h = 0, mỗi cặp điểm cho 2 hàng
    x, y = p[:, 0], p[:, 1]
    u, v = q[:, 0], q[:, 1]
    zero, one = np.zeros(n), np.ones(n)
    A = np.empty((2 * n, 9))
    A[0::2] = np.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], axis=1)
    A[1::2] = np.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], axis=1)

    # 3) SVD: h = vector kỳ dị phải ứng với giá trị kỳ dị nhỏ nhất
    _, svals, Vt = np.linalg.svd(A)

    # 4) kiểm rank: cần rank(A) = 8, tức giá trị kỳ dị thứ 8 phải khác 0 rõ rệt
    if svals[0] <= 0 or svals[7] / svals[0] < 1e-8:
        return None, svals, False

    H_hat = Vt[-1].reshape(3, 3)

    # 5) khử chuẩn hoá: H = T_dst^-1 H_hat T_src
    H = np.linalg.inv(T_dst) @ H_hat @ T_src

    # 6) chuẩn tỉ lệ để H[2,2] = 1 (nếu H[2,2] không quá gần 0)
    if abs(H[2, 2]) > 1e-12 * np.abs(H).max():
        H = H / H[2, 2]
    if not np.all(np.isfinite(H)):
        return None, svals, False
    return H, svals, True


def transfer_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to N x 2 points and dehomogenise: pi(H x)."""
    # YOUR CODE HERE
    H = np.asarray(H, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])        # (x, y) -> (x, y, 1)
    mapped = hom @ H.T                                    # mỗi hàng = (H x)^T
    w = mapped[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return mapped[:, :2] / w                          # pi(.): chia cho thành phần thứ 3


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence e_i = |x' - pi(H x)|^2 + |x - pi(H^-1 x')|^2."""
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    fwd = dst - transfer_points(H, src)                   # x' - pi(H x)
    bwd = src - transfer_points(H_inv, dst)               # x  - pi(H^-1 x')
    err = (fwd ** 2).sum(axis=1) + (bwd ** 2).sum(axis=1)
    return np.where(np.isfinite(err), err, np.inf)        # điểm ra vô cực -> lỗi vô cùng lớn


def _three_collinear(p: np.ndarray, tol: float = 1e-6) -> bool:
    """Supplied helper: True if any three of the four points are (nearly) collinear."""
    from itertools import combinations
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    for i, j, k in combinations(range(len(p)), 3):
        a, b, c = p[i], p[j], p[k]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < tol * scale * scale:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray, config: RansacConfig
                      ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, bool]:
    """Robust homography with adaptive RANSAC and a final refit on all inliers.

    Each iteration draws four correspondences with ``config.rng()``, rejects
    the sample if any three source points are collinear, fits
    ``dlt_homography``, and scores every correspondence by the symmetric
    transfer error; correspondences under ``config.threshold`` are inliers.
    Keep the model with the most inliers (ties: smaller sum of inlier
    residuals); update N = log(1-p) / log(1-(1-eps)^4) from the current inlier
    fraction, capped by ``config.max_iterations``; refit on all inliers.

    Returns (H, inlier_mask, residuals, iterations_used, status) where status
    is False when no model reached ``config.min_inliers`` (H may then be None).
    """
    # YOUR CODE HERE
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, dtype=bool), np.full(n, np.inf), 0, False

    rng = config.rng()                      # generator mới mỗi lần gọi -> tất định theo seed
    p = float(config.confidence)
    N = int(config.max_iterations)          # số vòng cần chạy, sẽ được cập nhật thích nghi

    best_H, best_mask = None, np.zeros(n, dtype=bool)
    best_count, best_sum = -1, np.inf
    it = 0
    while it < N:
        it += 1
        idx = rng.choice(n, 4, replace=False)          # mẫu tối thiểu 4 cặp
        if _three_collinear(src[idx]):                 # mẫu suy biến -> bỏ
            continue
        H, _, ok = dlt_homography(src[idx], dst[idx])
        if not ok:
            continue
        e = symmetric_transfer_error(H, src, dst)
        mask = e < config.threshold
        count = int(mask.sum())
        rsum = float(e[mask].sum())
        if count > best_count or (count == best_count and rsum < best_sum):
            best_H, best_mask, best_count, best_sum = H, mask, count, rsum
            # cập nhật N = log(1-p) / log(1-w^4), w = tỉ lệ inlier hiện tại
            w4 = (count / n) ** 4
            if w4 >= 1.0 - 1e-12:
                N = it                                  # toàn inlier: dừng ngay
            elif w4 > 0:
                N_new = np.log(1.0 - p) / np.log(1.0 - w4)
                N = min(int(config.max_iterations), max(it, int(np.ceil(N_new))))

    if best_H is None or best_count < config.min_inliers:
        resid = symmetric_transfer_error(best_H, src, dst) if best_H is not None else np.full(n, np.inf)
        return best_H, best_mask, resid, it, False

    # refit trên toàn bộ inlier, rồi tính lại residual và inlier theo H cuối cùng
    H_ref, _, ok = dlt_homography(src[best_mask], dst[best_mask])
    H_final = H_ref if ok else best_H
    resid = symmetric_transfer_error(H_final, src, dst)
    mask = resid < config.threshold
    if mask.sum() < config.min_inliers:                # refit tệ hơn -> giữ mô hình tốt nhất
        H_final = best_H
        resid = symmetric_transfer_error(H_final, src, dst)
        mask = resid < config.threshold
    return H_final, mask, resid, it, bool(mask.sum() >= config.min_inliers)


def warp_artwork(artwork: np.ndarray, H: np.ndarray, frame_shape: Tuple[int, ...]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Destination-to-source warp of ``artwork`` into a frame of shape ``frame_shape``.

    ``H`` maps artwork pixel coordinates to frame pixel coordinates. For every
    frame pixel inside the projected artwork quadrilateral, map it back through
    H^{-1} and sample the artwork bilinearly. Returns (warped H x W x C float32,
    coverage H x W bool); pixels that map outside the artwork receive no
    coverage and are zero in ``warped``.
    """
    # YOUR CODE HERE
    art = np.asarray(artwork, dtype=np.float32)
    H = np.asarray(H, dtype=np.float64)
    Hf, Wf = int(frame_shape[0]), int(frame_shape[1])
    Ha, Wa = art.shape[:2]
    warped = np.zeros((Hf, Wf) + art.shape[2:], dtype=np.float32)
    coverage = np.zeros((Hf, Wf), dtype=bool)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return warped, coverage

    # 1) chỉ xét bounding box của tứ giác chiếu artwork (nhanh hơn duyệt cả frame)
    corners = np.array([[0, 0], [Wa - 1, 0], [Wa - 1, Ha - 1], [0, Ha - 1]], dtype=np.float64)
    ch = np.hstack([corners, np.ones((4, 1))]) @ H.T
    if np.all(ch[:, 2] > 0):
        q = ch[:, :2] / ch[:, 2:3]
        x0 = max(int(np.floor(q[:, 0].min())) - 1, 0); x1 = min(int(np.ceil(q[:, 0].max())) + 1, Wf - 1)
        y0 = max(int(np.floor(q[:, 1].min())) - 1, 0); y1 = min(int(np.ceil(q[:, 1].max())) + 1, Hf - 1)
    else:                                   # tứ giác đi qua "vô cực": xét cả frame
        x0, x1, y0, y1 = 0, Wf - 1, 0, Hf - 1
    if x1 < x0 or y1 < y0:
        return warped, coverage

    # 2) destination -> source: mỗi pixel frame đi ngược qua H^-1 về toạ độ artwork
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0).astype(np.float64)
    m = H_inv @ pts
    with np.errstate(divide="ignore", invalid="ignore"):
        u = m[0] / m[2]
        v = m[1] / m[2]

    # 3) coverage: điểm rơi vào hình chữ nhật artwork [0, Wa-1] x [0, Ha-1] (pixel-centre)
    eps = 1e-6
    ok = (np.isfinite(u) & np.isfinite(v) & (u >= -eps) & (u <= Wa - 1 + eps) &
          (v >= -eps) & (v <= Ha - 1 + eps))
    if not ok.any():
        return warped, coverage

    # 4) lấy mẫu bilinear artwork tại (v, u) cho các pixel được phủ
    yy, xx = ys.ravel()[ok], xs.ravel()[ok]
    warped[yy, xx] = bilinear_sample(art, v[ok], u[ok])
    coverage[yy, xx] = True
    return warped, coverage


__all__ = ["bilinear_sample", "rectangle_to_rectangle", "normalise_points", "dlt_homography", "transfer_points",
           "symmetric_transfer_error", "ransac_homography", "warp_artwork"]
