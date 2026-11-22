import math

import numpy as np
import pytest

from vglab.errors import Dp2IsolationError, ScaleNotApplied
from vglab.initialization.candidates import UNFROZEN_SCALE_FIT
from vglab.initialization.scale import estimate_frame_scale, require_applicable_scale
from vglab.scene.admission import GaussianCandidate, admit
from vglab.scene.visibility import classify_visibility


def _candidate(gaussian_id: str, x: float, support: int, **overrides) -> GaussianCandidate:
    payload = dict(
        gaussian_id=gaussian_id,
        mean_world_m=np.array([x, 0.0, 2.0]),
        scale_std_m=np.array([0.01, 0.01, 0.01]),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        opacity=0.8,
        color_linear_rgb=np.array([0.2, 0.3, 0.4]),
        source_frame_id="src_000",
        source_time_s=0.0,
        layer_id=0,
        support_count=support,
        free_space_conflict=False,
    )
    payload.update(overrides)
    return GaussianCandidate(**payload)


def test_scale_is_log_median_and_failures_are_not_clipped():
    sharp = np.ones((8, 8))
    dp2 = np.full((8, 8), 2.0)
    mask = np.ones((8, 8), dtype=bool)
    fit = UNFROZEN_SCALE_FIT
    estimate = estimate_frame_scale(
        "src_000",
        sharp,
        dp2,
        mask,
        arm="S2",
        min_samples=fit["min_samples"],
        s_min=fit["s_min"],
        s_max=fit["s_max"],
        outlier_log_abs=fit["outlier_log_abs"],
    )
    assert estimate.status == "ok"
    assert estimate.s == pytest.approx(2.0)
    assert estimate.frozen is False
    assert require_applicable_scale(estimate) == pytest.approx(2.0)
    noisy = dp2.copy()
    noisy[0, 0] = 100.0
    robust = estimate_frame_scale(
        "src_000",
        sharp,
        noisy,
        mask,
        arm="S3",
        min_samples=fit["min_samples"],
        s_min=fit["s_min"],
        s_max=fit["s_max"],
        outlier_log_abs=fit["outlier_log_abs"],
    )
    assert robust.s == pytest.approx(2.0)
    too_few = estimate_frame_scale(
        "src_000",
        sharp,
        dp2,
        mask,
        arm="S2",
        min_samples=1000,
        s_min=0.25,
        s_max=4.0,
        outlier_log_abs=1.0,
    )
    assert too_few.status == "insufficient_samples"
    assert too_few.s is None
    with pytest.raises(ScaleNotApplied):
        require_applicable_scale(too_few)
    wide = estimate_frame_scale(
        "src_000",
        sharp,
        np.full_like(sharp, 10.0),
        mask,
        arm="S2",
        min_samples=8,
        s_min=0.25,
        s_max=4.0,
        outlier_log_abs=1.0,
    )
    assert wide.status == "scale_out_of_range"
    assert wide.s == pytest.approx(10.0)


def test_s0_and_s1_cannot_estimate_dp2_scale():
    mask = np.ones((8, 8), dtype=bool)
    with pytest.raises(Dp2IsolationError):
        estimate_frame_scale("src_000", np.ones((8, 8)), np.ones((8, 8)), mask, arm="S1", min_samples=1, s_min=0.1, s_max=10, outlier_log_abs=1)


def test_visibility_states_follow_the_tolerance():
    common = dict(width_px=10, height_px=8, alpha=0.5, image_valid=True, tau_abs_m=0.02, tau_rel=0.01, alpha_min=0.1)
    assert classify_visibility(-1.0, 1, 1, z_ref_m=2.0, **common) == "outside"
    assert classify_visibility(2.0, -1, 1, z_ref_m=2.0, **common) == "outside"
    assert classify_visibility(2.0, 1, 1, z_ref_m=math.nan, **common) == "unknown"
    assert classify_visibility(2.0, 1, 1, z_ref_m=2.0, **{**common, "alpha": 0.05}) == "unknown"
    tolerance = 0.02 + 0.01 * 2.0
    assert classify_visibility(2.0 + tolerance, 1, 1, z_ref_m=2.0, **common) == "supported"
    assert classify_visibility(2.0 + tolerance + 1e-6, 1, 1, z_ref_m=2.0, **common) == "occluded"
    assert classify_visibility(2.0 - tolerance - 1e-6, 1, 1, z_ref_m=2.0, **common) == "free_space_conflict"


def test_admission_reserves_new_regions_and_does_not_clone_or_average():
    supported = [_candidate(f"s{i}", float(i), support=2) for i in range(3)]
    novel = _candidate("new", 10.0, support=0, opacity=0.3)
    result = admit(
        supported + [novel],
        arm="S1",
        budget=2,
        new_region_fraction=0.5,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    kept_ids = [item.gaussian_id for item in result.kept]
    assert "new" in kept_ids
    assert len(result.kept) == 2
    assert result.cloned is False
    duplicate = _candidate("dup", 0.01, support=1, opacity=0.1, layer_id=0)
    original = _candidate("s0", 0.0, support=3, opacity=0.9, layer_id=0)
    other_layer = _candidate("back", 0.0, support=1, opacity=0.4, layer_id=1)
    deduped = admit(
        [duplicate, original, other_layer],
        arm="S2",
        budget=10,
        new_region_fraction=0.0,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    kept = {item.gaussian_id: item for item in deduped.kept}
    assert "dup" not in kept
    assert kept["s0"].opacity == pytest.approx(0.9)
    assert "back" in kept
    sparse = admit(
        [_candidate("only", 0.0, support=1)],
        arm="D0",
        budget=10,
        new_region_fraction=0.2,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    assert len(sparse.kept) == 1
    conflict = _candidate("bad", 3.0, support=0, free_space_conflict=True)
    blocked = admit(
        [conflict],
        arm="S0",
        budget=4,
        new_region_fraction=0.25,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    assert blocked.kept == ()
    assert blocked.rejected == (("bad", "free_space_conflict"),)
    opposite = _candidate(
        "opposite",
        0.0,
        support=4,
        layer_id=0,
        rotation_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    different_color = _candidate(
        "color",
        0.0,
        support=4,
        layer_id=0,
        color_linear_rgb=np.array([0.95, 0.05, 0.05]),
    )
    separated = admit(
        [_candidate("a", 0.0, support=1, scale_std_m=np.full(3, 0.001)), opposite, different_color],
        arm="D0",
        budget=10,
        new_region_fraction=0.0,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    assert {item.gaussian_id for item in separated.kept} == {"a", "opposite", "color"}
    far = [
        _candidate(f"p{index}", float(index), support=1, scale_std_m=np.full(3, 0.01))
        for index in range(3000)
    ]
    many = admit(
        far,
        arm="S1",
        budget=3000,
        new_region_fraction=0.0,
        geometry_source="arm_initialization",
        dedup_radius_m=0.05,
        max_scale_m=1.0,
    )
    assert len(many.kept) == 3000


def test_admission_rejects_direct_dp2_geometry_for_every_arm():
    candidate = _candidate("s0", 0.0, support=1)
    for arm in ("S1", "S2", "D0"):
        with pytest.raises(Dp2IsolationError):
            admit(
                [candidate],
                arm=arm,
                budget=4,
                new_region_fraction=0.0,
                geometry_source="dp2",
                dedup_radius_m=0.05,
                max_scale_m=1.0,
            )


def test_protocol_budget_must_be_one_of_the_registered_tiers():
    with pytest.raises(Exception):
        admit(
            [],
            arm="S1",
            budget=10,
            new_region_fraction=0.0,
            geometry_source="arm_initialization",
            dedup_radius_m=0.0,
            max_scale_m=1.0,
            protocol_budget=True,
        )
