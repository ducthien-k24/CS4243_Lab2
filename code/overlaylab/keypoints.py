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
    # YOUR CODE HERE
    raise NotImplementedError("YOUR CODE HERE")


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
    raise NotImplementedError("YOUR CODE HERE")


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
    raise NotImplementedError("YOUR CODE HERE")


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
    raise NotImplementedError("YOUR CODE HERE")


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
