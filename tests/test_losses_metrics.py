import math

import numpy as np
import pytest

from vglab.errors import IndependentReferenceError
from vglab.evaluate.decisions import classify_h2, classify_h4, pair_status
from vglab.evaluate.metrics import coverage_fraction, geometric_mae_m, psnr_srgb
from vglab.optimize.losses import depth_supervision, expected_depth, rgb_loss
from vglab.optimize.runner import SharedOptimizer


def _scenes(lpips_delta: float, coverage_candidate: float = 0.9, **extra) -> list[dict]:
    rows = []
    for _ in range(4):
        row = {
            "lpips_baseline": 0.20,
            "lpips_candidate": 0.20 - lpips_delta,
            "psnr_baseline": 20.0,
            "psnr_candidate": 20.0,
            "coverage_baseline": 0.9,
            "coverage_candidate": coverage_candidate,
        }
        row.update(extra)
        rows.append(row)
    return rows


def test_expected_depth_and_empty_supervision_are_not_zero_loss():
    depth = expected_depth(np.array([1.0, 3.0]), np.array([0.5, 1.0]), eps=0.0)
    assert depth == pytest.approx(2.0)
    assert expected_depth(np.array([1.0]), np.array([0.0])) is None
    empty = depth_supervision(
        np.ones((2, 2)),
        np.full((2, 2), np.nan),
        np.zeros((2, 2), dtype=bool),
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
    )
    assert empty.loss is None
    assert empty.coverage is None
    assert empty.status == "skipped_empty_depth_mask"
    hidden = depth_supervision(
        np.full((2, 2), np.e),
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2)),
        np.zeros((2, 2), dtype=bool),
    )
    assert hidden.loss is None
    assert hidden.coverage == pytest.approx(0.0)
    supervised = depth_supervision(
        np.full((2, 2), np.e),
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
    )
    assert supervised.loss == pytest.approx(1.0)
    assert supervised.coverage == pytest.approx(1.0)


def test_masked_rgb_loss_ignores_pixels_outside_the_mask():
    image = np.full((8, 8, 3), 0.5)
    other = image.copy()
    other[:, -1] = 0.0
    mask = np.ones((8, 8), dtype=bool)
    mask[:, -1] = False
    result = rgb_loss(image, other, mask, lambda_l1=0.8, lambda_ssim=0.2, window=7, sigma=1.5)
    assert result.status == "computed"
    assert result.l1 == pytest.approx(0.0)
    assert result.ssim == pytest.approx(1.0)
    assert result.value == pytest.approx(0.0)
    unmasked = rgb_loss(image, other, np.ones((8, 8), dtype=bool), lambda_l1=1.0, lambda_ssim=0.0, window=7)
    assert unmasked.l1 > 0


def test_psnr_coverage_and_reference_source():
    image = np.zeros((2, 2, 3))
    shifted = image.copy()
    shifted[0, 0] = 1.0
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    score, count = psnr_srgb(image, shifted, mask)
    assert count == 1
    assert score == pytest.approx(-10.0 * math.log10(1.0))
    perfect, _ = psnr_srgb(image, image, np.ones((2, 2), dtype=bool))
    assert perfect == math.inf
    assert coverage_fraction(np.array([0.2, 0.8, 0.9]), 0.5) == pytest.approx(2 / 3)
    with pytest.raises(IndependentReferenceError):
        geometric_mae_m(np.array([1.0]), np.array([1.1]), "predicted")
    assert geometric_mae_m(np.array([1.0, np.nan]), np.array([1.2, 5.0]), "independent_reference") == pytest.approx(0.2)


def test_optimizer_does_not_move_poses_or_report_measurements():
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1, 0, 3] = 0.2
    result = SharedOptimizer("S1", poses).run()
    assert result.status == "not_run_no_verified_renderer"
    assert result.pose_updated is False
    assert result.measurements is None
    assert result.densification is False
    assert SharedOptimizer("S0", poses).run().status == "diagnostic_only"


def test_h2_rules_do_not_treat_partial_tables_as_success():
    assert pair_status(12) == "complete"
    assert pair_status(11) == "inconclusive"
    assert classify_h2(_scenes(0.02), g0="na").status == "advance_candidate"
    assert classify_h2(_scenes(0.02), g0="na").geometry == "not_evaluated"
    assert classify_h2(_scenes(0.02)[:3], g0="na").status == "inconclusive"
    assert classify_h2(_scenes(0.02, coverage_candidate=0.87), g0="na").status == "g1_failed"
    assert classify_h2(_scenes(0.004), g0="na").status == "no_clear_gain"
    assert classify_h2(_scenes(0.02), g0="na", seed_std=0.05).status == "inconclusive"
    degraded = _scenes(0.02, f_score_baseline=0.5, f_score_candidate=0.4)
    assert classify_h2(degraded, g0="available").geometry == "degraded"
    assert classify_h2(degraded, g0="available").status == "no_clear_gain"
    assert classify_h2(_scenes(0.02), g0="available").status == "inconclusive"
    held = _scenes(0.02, f_score_baseline=0.5, f_score_candidate=0.5)
    assert classify_h2(held, g0="available").status == "advance_candidate"
    assert classify_h2(held, g0="available").geometry == "no_degradation"
    partial = _scenes(0.02)
    partial[0] = {**partial[0], "f_score_baseline": 0.5, "f_score_candidate": 0.5}
    assert classify_h2(partial, g0="available").status == "inconclusive"
    assert classify_h4(partial, g0="available", cost_supports=True).status == "inconclusive"
    assert classify_h4(_scenes(0.02), g0="na", cost_supports=None).status == "inconclusive"
    assert classify_h4(_scenes(0.02), g0="na", cost_supports=False).status == "no_clear_gain"
    assert classify_h4(_scenes(0.02), g0="na", cost_supports=True).status == "advance_candidate"
    assert classify_h4(held, g0="available", cost_supports=True).status == "advance_candidate"
