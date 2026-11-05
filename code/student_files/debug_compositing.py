"""Synthetic Stage 3 checks; run from code/: python student_files/debug_compositing.py.

Uses only synthetic.py data, never common test/stress or personal audit frames.
Outputs are development evidence, not the final frozen thresholds. [AI-CODE]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

from overlaylab import metrics, synthetic as syn, supplied as sp, visualize as viz
from overlaylab.compositing import (
    compensated_difference, occlusion_mask, composite, select_difference_thresholds,
)
from overlaylab.config import CompositeConfig, OcclusionConfig
from overlaylab.geometry import warp_artwork


class RegressionChecks(unittest.TestCase):
    def test_gain_offset_ignores_a_large_colour_occluder(self):
        ref = np.random.default_rng(42).uniform(0.1, 0.7, (30, 40, 3)).astype(np.float32)
        observed = 0.8 * ref + 0.06
        observed[:8, :10] = (0.9, 0.05, 0.85)
        coverage = np.ones(ref.shape[:2], bool)
        coverage[-2:] = False
        difference, (a, b) = compensated_difference(observed, ref, coverage, OcclusionConfig())
        np.testing.assert_allclose((a, b), (0.8, 0.06), atol=1e-6)
        self.assertLess(float(difference[10:25].max()), 1e-6)
        self.assertGreater(float(difference[:8, :10].mean()), 0.2)
        self.assertTrue(np.all(difference[~coverage] == 0))

    def test_empty_coverage_and_constant_reference_are_finite(self):
        ref = np.full((8, 9, 3), 0.2, np.float32)
        observed = np.full_like(ref, 0.35)
        diff, fit = compensated_difference(observed, ref, np.zeros((8, 9), bool), OcclusionConfig())
        self.assertEqual(fit, (1.0, 0.0))
        self.assertFalse(diff.any())
        diff, fit = compensated_difference(observed, ref, np.ones((8, 9), bool), OcclusionConfig())
        np.testing.assert_allclose(fit, (1, 0.15), atol=1e-6)
        self.assertLess(float(diff.max()), 1e-6)

    def test_uint8_identity_and_empty_warp(self):
        ref = np.random.default_rng(4).integers(0, 256, (20, 24, 3), dtype=np.uint8)
        mask, diff, components = occlusion_mask(ref, np.eye(3), ref, OcclusionConfig())
        self.assertFalse(mask.any())
        self.assertLess(float(diff.max()), 1e-6)
        self.assertEqual(components.shape, (0, 6))
        H = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]], float)
        mask, diff, components = occlusion_mask(ref, H, ref, OcclusionConfig())
        self.assertFalse(mask.any() or diff.any())
        self.assertEqual(components.shape, (0, 6))

    def test_feather_preserves_occluder_and_image_boundary(self):
        frame = np.full((32, 40, 3), 0.2, np.float32)
        artwork = np.full_like(frame, 0.8)
        cov = np.ones(frame.shape[:2], bool)
        occ = np.zeros_like(cov)
        occ[12:20, 16:24] = True
        out, alpha = composite(frame, artwork, cov, occ, CompositeConfig(feather=4))
        np.testing.assert_array_equal(out[occ], frame[occ])
        self.assertEqual(float(alpha[occ].max()), 0)
        self.assertLess(float(alpha[11, 18]), 0.6)
        self.assertLess(float(alpha[0, 0]), 0.6)
        self.assertEqual(float(alpha[6, 6]), 1)

    def test_optional_gain_matching_does_not_use_foreground(self):
        frame = np.full((20, 30, 3), 0.4, np.float32)
        cov = np.ones((20, 30), bool)
        occ = np.zeros_like(cov)
        occ[:5] = True
        frame[occ] = 1
        out, _ = composite(frame, np.full_like(frame, 0.2), cov, occ,
                           CompositeConfig(feather=0, match_gain=True))
        np.testing.assert_allclose(out[~occ], 0.4, atol=1e-6)
        np.testing.assert_array_equal(out[occ], frame[occ])

    def test_search_uses_centre_of_tied_strong_range_without_mutation(self):
        ref = np.full((32, 48, 3), 0.2, np.float32)
        observed = ref.copy()
        observed[10:20, 15:30] = 0.9
        gt = np.zeros((32, 48), bool)
        gt[10:20, 15:30] = True
        cfg = OcclusionConfig(strong=0.25, weak=0.12, min_area=10, open_radius=0)
        original = asdict(cfg)
        clean = np.zeros_like(gt)
        frozen, log = select_difference_thresholds(
            [(observed, np.eye(3), ref), (ref, np.eye(3), ref)], [gt, clean], cfg)
        self.assertEqual(asdict(cfg), original)
        self.assertIsNot(frozen, cfg)
        self.assertAlmostEqual(frozen.strong, 0.25)
        self.assertTrue(all(row['mean_iou'] == 1 for row in log))
        self.assertTrue(all(row['n_scored'] == 1 for row in log))
        with self.assertRaises(ValueError):
            select_difference_thresholds([], [], cfg)
        with self.assertRaises(ValueError):
            select_difference_thresholds([(ref, np.eye(3), ref)], [], cfg)


def debug_synthetic(output: Path) -> None:
    """Exercise the supplied renderer under lighting, geometry and occluder changes."""
    output.mkdir(parents=True, exist_ok=True)
    board = syn.make_board_texture(800, 200, seed=1)
    background = syn.make_background(1400, 800, seed=2)
    artwork = syn.make_artwork(800, 200, seed=3)
    Hbg = np.diag([0.5, 0.5, 1.0])
    corners = syn.project_board(800, 200, (320, 180), 420, np.deg2rad(10), 0.05, 0)
    H = syn.homography_from_corners(sp.rect_corners(800, 200), corners)
    cfg = OcclusionConfig(min_area=60, open_radius=1)
    # Synthetic development selection only: do not save outputs/thresholds.json.
    training, masks = [], []
    for seed, cx in [(10, 290), (11, 350)]:
        frame, occ, _ = syn.render_frame(
            board, background, H, Hbg, (640, 360), gain=0.85, gradient=0.08,
            noise_std=0.01, occluders=[('ball', np.array([cx, 170, 30]))],
            rng=np.random.default_rng(seed))
        training.append((frame, H, board))
        masks.append(occ & sp.quad_mask((360, 640), corners))
    frozen, log = select_difference_thresholds(training, masks, cfg)
    cases = [
        ('clean_lighting', [], 0.85, 0.08, 0.0, 10),
        ('ball', [('ball', np.array([330, 170, 35]))], 0.85, 0.08, 0.0, 10),
        ('thin_pole', [('pole', np.array([300, 90, 335, 260, 2]))], 1.0, 0.0, 0.0, 10),
        ('hand', [('hand', np.array([330, 175, 35, 25]))], 0.85, 0.08, 0.0, 10),
        ('strong_perspective', [], 0.85, 0.08, 0.0, 60),
        ('blurred_ball', [('ball', np.array([330, 170, 35]))], 0.85, 0.08, 1.5, 10),
    ]
    rows = []
    for index, (name, shapes, gain, gradient, blur, yaw) in enumerate(cases):
        corners = syn.project_board(800, 200, (320, 180), 420, np.deg2rad(yaw), 0.05, 0)
        H = syn.homography_from_corners(sp.rect_corners(800, 200), corners)
        frame, occ, _ = syn.render_frame(
            board, background, H, Hbg, (640, 360), gain=gain, gradient=gradient,
            blur_sigma=blur, noise_std=0.01, occluders=shapes,
            rng=np.random.default_rng(100 + index))
        gt = occ & sp.quad_mask(frame.shape[:2], corners)
        mask, difference, components = occlusion_mask(frame, H, board, frozen)
        warped, coverage = warp_artwork(artwork, H, frame.shape)
        out, alpha = composite(frame, warped, coverage, mask, CompositeConfig())
        _, baseline = composite(frame, warped, coverage, np.zeros_like(mask), CompositeConfig())
        visible = coverage & ~gt
        row = dict(case=name, seed=100 + index, subset='synthetic-debug',
                   variant='known-homography', iou=metrics.mask_iou(mask, gt),
                   f1=metrics.mask_f1(mask, gt), leakage=metrics.leakage(alpha, gt),
                   baseline_leakage=metrics.leakage(baseline, gt),
                   false_occlusion=float((mask & visible).sum() / max(visible.sum(), 1)),
                   components=len(components), gt_pixels=int(gt.sum()))
        rows.append(row)
        assert not np.any(mask & ~coverage), name
        np.testing.assert_array_equal(out[gt & mask], frame[gt & mask])
        panels = [viz.label(frame, name), viz.label(viz.heatmap(difference, 0.6), 'difference'),
                  viz.label(viz.overlay_mask(frame, gt), 'synthetic ground truth'),
                  viz.label(viz.overlay_mask(frame, mask), 'predicted mask'),
                  viz.label(out, 'composite'), viz.label(viz.heatmap(alpha, 1), 'alpha')]
        sp.write_image(output / f'{name}.png', viz.montage(panels, cols=3))
        print(json.dumps(row))
    report = dict(seed_note='Board 1, background 2, artwork 3; fit seeds 10/11; case seeds 100..105',
                  runtime=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__),
                  thresholds=asdict(frozen), search_log=log, cases=rows,
                  note='Development only. Known synthetic homographies isolate Stage 3. '
                       'NaN metrics are undefined and retained, never replaced by zero.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Debug evidence:', output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'stage3_debug')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegressionChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    debug_synthetic(args.output)
