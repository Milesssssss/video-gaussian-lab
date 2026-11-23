import numpy as np
import pytest

from tests.manifest_factory import synthetic_manifest
from vglab.depth.aggregate import DepthPrediction
from vglab.depth.teacher import DepthView, attach_camera, build_depth_teacher
from vglab.errors import ContractError, LeakageError

K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
POSE = np.eye(4)


def _prediction(depth: np.ndarray, frame_id: str = "src_000") -> DepthPrediction:
    return attach_camera(
        DepthPrediction(frame_id, depth, ("src_000", "src_001"), representation="z_depth", unit="m"),
        K,
        POSE,
    )


def test_teacher_mask_records_invalid_and_static_exclusions():
    depth = np.array([[1.0, np.nan], [-0.5, 2.0]])
    static = np.array([[True, True], [True, False]])
    teacher = build_depth_teacher(_prediction(depth), synthetic_manifest(), static_mask=static)
    assert teacher.mask.tolist() == [[True, False], [False, False]]
    assert teacher.diagnostics["depth_invalid_px"] == 2
    assert teacher.diagnostics["teacher_valid_px"] == 1
    assert teacher.reliability[1, 0] == 0.0
    assert 0.0 < teacher.reliability[0, 0] <= 1.0


def test_teacher_reliability_is_heuristic_and_bounded():
    depth = np.tile(np.linspace(0.4, 15.0, 8), (4, 1))
    teacher = build_depth_teacher(_prediction(depth), synthetic_manifest())
    assert np.all((teacher.reliability >= 0.05) & (teacher.reliability <= 1.0))
    assert teacher.diagnostics["supported_px"] == 0
    assert teacher.config_name == "heuristic_reliability_v0"
    assert teacher.frozen is False


def test_shifted_source_view_tags_surface_as_supported():
    manifest = synthetic_manifest()
    depth = np.full((4, 4), 2.0)
    pose_other = np.eye(4)
    pose_other[0, 3] = 0.05
    view = DepthView("src_001", np.full((4, 4), 2.0), K, pose_other)
    teacher = build_depth_teacher(_prediction(depth), manifest, views=(view,))
    assert teacher.diagnostics["free_space_conflict_px"] == 0
    assert teacher.diagnostics["supported_px"] > 0
    assert teacher.diagnostics["views_checked"] == 1
    assert np.all(teacher.mask)


def test_surface_in_front_of_other_view_is_a_free_space_conflict():
    manifest = synthetic_manifest()
    depth = np.full((4, 4), 1.0)
    pose_other = np.eye(4)
    pose_other[0, 3] = 0.05
    view = DepthView("src_001", np.full((4, 4), 2.0), K, pose_other)
    teacher = build_depth_teacher(_prediction(depth), manifest, views=(view,))
    assert teacher.diagnostics["free_space_conflict_px"] > 0
    assert teacher.mask.sum() < depth.size
    assert np.all(teacher.reliability[~teacher.mask] == 0.0)


def test_support_check_rejects_heldout_view_and_self_view():
    manifest = synthetic_manifest()
    depth = np.full((4, 4), 2.0)
    heldout = DepthView("val_000", np.full((4, 4), 2.0), K, np.eye(4))
    with pytest.raises(LeakageError):
        build_depth_teacher(_prediction(depth), manifest, views=(heldout,))
    self_view = DepthView("src_000", np.full((4, 4), 2.0), K, np.eye(4))
    with pytest.raises(ContractError):
        build_depth_teacher(_prediction(depth), manifest, views=(self_view,))


def test_temporal_reliability_penalizes_inconsistent_depth():
    manifest = synthetic_manifest()
    depth = np.full((4, 4), 2.0)
    noisy = np.full((4, 4), 4.0)
    stable = build_depth_teacher(_prediction(depth), manifest, temporal_depth_m=depth)
    inconsistent = build_depth_teacher(_prediction(depth), manifest, temporal_depth_m=noisy)
    assert stable.reliability.mean() > inconsistent.reliability.mean()


def test_teacher_requires_attached_camera_for_support_checks():
    plain = DepthPrediction("src_000", np.full((4, 4), 2.0), ("src_000",))
    view = DepthView("src_001", np.full((4, 4), 2.0), K, np.eye(4))
    with pytest.raises(ContractError):
        build_depth_teacher(plain, synthetic_manifest(), views=(view,))
    no_views = build_depth_teacher(plain, synthetic_manifest())
    assert no_views.diagnostics["views_checked"] == 0


def test_one_pixel_depth_map_does_not_break_edge_reliability():
    tiny = DepthPrediction("src_000", np.array([[2.0]]), ("src_000",))
    teacher = build_depth_teacher(tiny, synthetic_manifest())
    assert teacher.mask.tolist() == [[True]]
    assert 0.0 < teacher.reliability[0, 0] <= 1.0
