import numpy as np
import pytest

from tests.manifest_factory import synthetic_manifest
from vglab.depth.aggregate import (
    DepthPrediction,
    aggregate_frame_predictions,
    log_median_depth,
    scalar_focal_px,
    source_windows,
)
from vglab.errors import ContractError, LeakageError


def test_log_median_ignores_invalid_and_preserves_empty_pixels():
    first = np.array([[1.0, np.nan], [4.0, 4.0]])
    second = np.array([[np.e**2, -1.0], [0.0, 1.0]])
    third = np.array([[np.e**4, np.nan], [np.nan, np.nan]])
    merged = log_median_depth([first, second, third])
    assert merged[0, 0] == pytest.approx(np.e**2)
    assert np.isnan(merged[0, 1])
    assert merged[1, 0] == pytest.approx(4.0)
    assert merged[1, 1] == pytest.approx(2.0)


def test_windows_use_source_frames_only_and_do_not_pad(repo_root):
    manifest = synthetic_manifest()
    manifest["frames"] = []
    ids = []
    for index in range(20):
        frame = synthetic_manifest()["frames"][0]
        frame = {**frame, "frame_id": f"src_{index:03d}", "timestamp_s": index * 0.1, "split": "source"}
        manifest["frames"].append(frame)
        ids.append(frame["frame_id"])
    manifest["split_policy"]["source_frame_ids"] = ids
    manifest["split_policy"]["validation_frame_ids"] = []
    manifest["split_policy"]["test_frame_ids"] = []
    windows = source_windows(ids, manifest, window_size=16, overlap=4)
    assert [len(item.context_frame_ids) for item in windows] == [16, 8]
    assert windows[0].context_frame_ids[-1] == "src_015"
    assert windows[1].context_frame_ids[0] == "src_012"
    assert windows[1].context_frame_ids[-1] == "src_019"
    assert len(set(windows[1].context_frame_ids)) == 8
    with pytest.raises(LeakageError):
        source_windows(["src_000", "val_000"], synthetic_manifest())


def test_short_sequence_is_not_duplicated_to_fill_the_window():
    manifest = synthetic_manifest()
    ids = ["src_000", "src_001"]
    windows = source_windows(ids, manifest, window_size=16, overlap=4)
    assert len(windows) == 1
    assert windows[0].context_frame_ids == ("src_000", "src_001")


def test_prediction_context_must_stay_on_source_frames():
    manifest = synthetic_manifest()
    depth = np.ones((4, 4))
    good = DepthPrediction("src_000", depth, ("src_000", "src_001"))
    assert aggregate_frame_predictions([good], manifest, "src_000")[0, 0] == pytest.approx(1.0)
    leaked = DepthPrediction("src_000", depth, ("src_000", "test_000"))
    with pytest.raises(LeakageError):
        aggregate_frame_predictions([leaked], manifest, "src_000")


def test_scalar_focal_refuses_to_average_fx_and_fy():
    square = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    assert scalar_focal_px(square) == pytest.approx(10.0)
    skewed = np.array([[10.0, 0.0, 1.0], [0.0, 12.0, 1.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ContractError):
        scalar_focal_px(skewed)


def test_range_labeled_depth_is_rejected():
    manifest = synthetic_manifest()
    prediction = DepthPrediction("src_000", np.ones((4, 4)), ("src_000",), representation="range")
    with pytest.raises(ContractError):
        aggregate_frame_predictions([prediction], manifest, "src_000")
