"""Stage 1 Localise, keypoint branch.

Implement the four functions marked ``YOUR CODE HERE``. Do not change the
signatures or return types; the tests and the grading harness call them.
Use only numpy and the helpers in ``overlaylab.supplied`` (Sobel gradients,
Gaussian smoothing, bilinear sampling). OpenCV feature functions
(``cv2.cornerHarris``, ``cv2.SIFT`` and friends) are not allowed here; the
SIFT baseline lives only in the labelled ablation cell of Notebook 1.

Conventions: grayscale images are float32 in [0, 1]; ``x`` indexes columns and
``y`` rows; keypoints are N x 2 arrays of (x, y); descriptors are N x 128.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import DetectorConfig, DescriptorConfig, MatchConfig
from .supplied import sobel_gradients, gaussian_smooth
from .geometry import bilinear_sample


def harris_response(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Harris response R = det(M) - kappa * tr(M)^2 at every pixel.

    M is the Gaussian-weighted (scale ``config.sigma_w``) second-moment matrix
    built from the supplied normalised Sobel gradients. Returns a float32 map
    with the same shape as ``gray``.
    """
    
    gx, gy, _, _ = sobel_gradients(gray)
    i_x = gx * gx
    i_y = gy * gy
    i_xy = gx * gy
    
    Sxx = gaussian_smooth(i_x, config.sigma_w)
    Syy = gaussian_smooth(i_y, config.sigma_w)
    Sxy = gaussian_smooth(i_xy, config.sigma_w)
    
    det = Sxx * Syy - Sxy**2 
    trace = Sxx + Syy
    R = det - config.kappa * (trace**2)
    
    return R.astype(np.float32)
    


def detect_corners(gray: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Harris corners with non-maximum suppression.

    Steps: compute ``harris_response``; discard responses below
    ``config.response_fraction * max(R)``; apply NMS in mode ``config.nms_mode``
    with radius ``config.nms_radius`` ("window": keep a pixel only if it is the
    maximum of its (2r+1)x(2r+1) neighbourhood; "radius": sort survivors by R
    and greedily accept the strongest while discarding weaker candidates within
    distance r); drop corners within ``config.border`` pixels of the image edge.

    Returns (keypoints, responses): an N x 2 float array of (x, y) ordered by
    decreasing response (N <= config.max_corners) and the matching responses.
    """
    # YOUR CODE HERE
    if config.nms_mode not in ("window", "radius"):
        raise ValueError(f"unknown nms_mode: {config.nms_mode!r}")

    R = harris_response(gray, config)
    candidate = R > config.response_fraction * R.max()

    # bỏ border (cả hàng lẫn cột) TRƯỚC khi NMS
    b = int(config.border)
    if b > 0:
        candidate[:b, :] = False
        candidate[-b:, :] = False
        candidate[:, :b] = False
        candidate[:, -b:] = False

    r = int(config.nms_radius)
    H, W = R.shape

    if config.nms_mode == "window":
        # giữ pixel là max trong ô (2r+1)x(2r+1)
        padded = np.pad(R, r, mode="constant", constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * r + 1, 2 * r + 1))
        maxR = windows.max(axis=(-1, -2))
        keep = candidate & (R == maxR)
        ys, xs = np.nonzero(keep)
    else:
        # "radius": duyệt tham lam từ mạnh đến yếu, chặn mọi pixel cách <= r
        cy, cx = np.nonzero(candidate)
        order = np.argsort(-R[cy, cx], kind="stable")
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (dx * dx + dy * dy) <= r * r
        blocked = np.zeros((H, W), dtype=bool)
        acc_y, acc_x = [], []
        for i in order:
            y, x = cy[i], cx[i]
            if blocked[y, x]:
                continue
            acc_y.append(y)
            acc_x.append(x)
            if len(acc_y) >= config.max_corners:
                break
            # cắt ô vuông quanh (y, x) và phần disk tương ứng khi sát biên
            y0, y1 = max(y - r, 0), min(y + r + 1, H)
            x0, x1 = max(x - r, 0), min(x + r + 1, W)
            blocked[y0:y1, x0:x1] |= disk[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
        ys = np.asarray(acc_y, dtype=int)
        xs = np.asarray(acc_x, dtype=int)

    # sort theo R giảm dần, cắt max_corners, đổi sang (x, y)
    vals = R[ys, xs]
    order = np.argsort(-vals, kind="stable")[:config.max_corners]
    keypoints = np.stack([xs[order], ys[order]], axis=1).astype(np.float64)
    return keypoints, vals[order].astype(np.float64)


def describe_keypoints(gray: np.ndarray, keypoints: np.ndarray,
                       config: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """128-D gradient-histogram descriptors with dominant-orientation normalisation.

    For each keypoint take a 16 x 16 grid of samples spaced ``config.patch_scale``
    pixels apart (so the window is 16 * patch_scale pixels wide),
    build a ``config.orientation_bins``-bin orientation histogram weighted by
    gradient magnitude, assign the dominant orientation, rotate the sampling
    grid so that the first histogram bin is aligned to it, and accumulate a
    ``cells x cells`` grid of ``cell_bins``-bin histograms (4 x 4 x 8 = 128).
    Apply Gaussian spatial weighting when ``config.gaussian_weighting`` is set,
    normalise to unit L2 norm, clamp to ``config.clamp``, and renormalise.

    Keypoints whose (rotated) window leaves the image are dropped. Returns
    (descriptors N' x 128, kept_indices N') where kept_indices indexes the input.
    """
    # YOUR CODE HERE
    gray = np.asarray(gray, dtype=np.float32)
    kps = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    H, W = gray.shape
    n_grid = 16
    nb_ori = int(config.orientation_bins)
    cells, cb = int(config.cells), int(config.cell_bins)
    dim = cells * cells * cb
    ps = float(config.patch_scale)

    # 1) lưới 16x16 tâm ở keypoint, bước patch_scale px; U theo x (cột), V theo y (hàng)
    u = (np.arange(n_grid) - (n_grid - 1) / 2.0) * ps
    U, V = np.meshgrid(u, u)                                   # V[i, j] = hàng i, U[i, j] = cột j

    # 2) bỏ keypoint có cửa sổ (xoay bất kỳ) rơi ra ngoài ảnh: dùng bán kính bao của lưới
    radius = np.sqrt(2.0) * (n_grid - 1) / 2.0 * ps
    inside = ((kps[:, 0] - radius >= 0) & (kps[:, 0] + radius <= W - 1) &
              (kps[:, 1] - radius >= 0) & (kps[:, 1] + radius <= H - 1))
    kept = np.nonzero(inside)[0]
    if kept.size == 0:
        return np.zeros((0, dim), dtype=np.float32), kept
    kx = kps[kept, 0][:, None, None]                           # K x 1 x 1 để broadcast với lưới
    ky = kps[kept, 1][:, None, None]
    K = kept.size

    gx, gy, _, _ = sobel_gradients(gray)

    # trọng số Gaussian theo không gian (sigma = nửa bề rộng cửa sổ)
    if config.gaussian_weighting:
        sig = 0.5 * n_grid * ps
        weight = np.exp(-(U ** 2 + V ** 2) / (2.0 * sig ** 2))
    else:
        weight = np.ones_like(U)

    # 3) hướng chủ đạo: histogram nb_ori bin trên lưới chưa xoay
    sx, sy = kx + U, ky + V
    ax, ay = bilinear_sample(gx, sy, sx), bilinear_sample(gy, sy, sx)
    mag = np.hypot(ax, ay) * weight
    ang = np.mod(np.arctan2(ay, ax), 2 * np.pi)
    bw = 2 * np.pi / nb_ori
    b = np.floor(ang / bw).astype(int) % nb_ori
    flat = (np.arange(K)[:, None, None] * nb_ori + b).ravel()
    hist = np.bincount(flat, weights=mag.ravel(), minlength=K * nb_ori).reshape(K, nb_ori)
    k = hist.argmax(axis=1)
    # nội suy parabol quanh đỉnh để hướng liên tục hơn (bin có tâm tại (k + 0.5) * bw)
    h0 = hist[np.arange(K), (k - 1) % nb_ori]
    h1 = hist[np.arange(K), k]
    h2 = hist[np.arange(K), (k + 1) % nb_ori]
    den = h0 - 2 * h1 + h2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (h0 - h2) / np.where(den == 0, 1, den), 0.0)
    theta = ((k + 0.5 + np.clip(delta, -0.5, 0.5)) * bw)[:, None, None]

    # 4) xoay lưới theo theta và lấy mẫu gradient trên lưới đã xoay
    c, s = np.cos(theta), np.sin(theta)
    rx = kx + c * U - s * V
    ry = ky + s * U + c * V
    gxs, gys = bilinear_sample(gx, ry, rx), bilinear_sample(gy, ry, rx)
    mag = np.hypot(gxs, gys) * weight
    rel = np.mod(np.arctan2(gys, gxs) - theta, 2 * np.pi)      # hướng tương đối so với theta

    # 5) cộng vào lưới cells x cells, mỗi ô cb bin (nội suy tuyến tính giữa 2 bin hướng)
    cell_size = n_grid // cells
    ii, jj = np.meshgrid(np.arange(n_grid), np.arange(n_grid), indexing="ij")
    cell = (ii // cell_size) * cells + (jj // cell_size)       # 16 x 16
    f = rel / (2 * np.pi / cb)
    b0 = np.floor(f).astype(int)
    frac = f - b0
    b0 %= cb
    b1 = (b0 + 1) % cb
    base = np.arange(K)[:, None, None] * dim + cell[None] * cb
    desc = (np.bincount((base + b0).ravel(), weights=(mag * (1 - frac)).ravel(), minlength=K * dim) +
            np.bincount((base + b1).ravel(), weights=(mag * frac).ravel(), minlength=K * dim)).reshape(K, dim)

    # 6) chuẩn hoá L2 -> clamp -> chuẩn hoá lại
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
    desc = np.minimum(desc, config.clamp)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
    return desc.astype(np.float32), kept


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      config: MatchConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matching with Lowe's ratio test (optionally mutual).

    For every descriptor in A find the nearest and second-nearest descriptor in
    B by Euclidean distance; accept the match when d1 / d2 < ``config.ratio``.
    With ``config.mutual`` the match must also be the nearest neighbour of the B
    descriptor in A. Returns (pairs M x 2 of (index_a, index_b), distances M,
    ratios M).
    """
    # YOUR CODE HERE
    if len(desc_a) == 0 or len(desc_b) == 0:
        pairs = np.zeros((0, 2), dtype = int)
        distances = np.zeros(0)
        ratios = np.zeros(0)
        return (pairs, distances, ratios)

    A = np.asarray(desc_a, dtype=np.float64)
    B = np.asarray(desc_b, dtype=np.float64)
    Na, Nb = len(A), len(B)

    # 1) ma trận khoảng cách Euclid Na x Nb: |a-b|^2 = |a|^2 + |b|^2 - 2 a.b
    aa = (A ** 2).sum(axis=1)
    bb = (B ** 2).sum(axis=1)
    D = np.sqrt(np.maximum(aa[:, None] + bb[None, :] - 2.0 * A @ B.T, 0.0))

    # 2) láng giềng gần nhất (j1, d1) và gần thứ hai (d2) trong B cho mỗi hàng A
    rows = np.arange(Na)
    if Nb >= 2:
        nn = np.argpartition(D, 1, axis=1)[:, :2]                 # 2 cột nhỏ nhất (chưa sắp xếp)
        nn = np.take_along_axis(nn, np.argsort(D[rows[:, None], nn], axis=1), axis=1)
        j1 = nn[:, 0]
        d1 = D[rows, j1]
        d2 = D[rows, nn[:, 1]]
    else:                                                          # B chỉ có 1 phần tử: không có d2
        j1 = np.zeros(Na, dtype=int)
        d1 = D[:, 0]
        d2 = np.full(Na, np.inf)

    # 3) Lowe's ratio test: d1/d2 < ratio (d1 = d2 = 0 -> ratio 1, coi là mơ hồ)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d2 > 0, d1 / d2, 1.0)
    ok = ratio < config.ratio

    # 4) mutual: B_j1 cũng phải chọn lại A_i là gần nhất
    if config.mutual:
        nn_ba = D.argmin(axis=0)
        ok &= nn_ba[j1] == rows

    # 5) đóng gói
    ia = np.nonzero(ok)[0]
    pairs = np.stack([ia, j1[ia]], axis=1).astype(int).reshape(-1, 2)
    return pairs, d1[ia], ratio[ia]


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
