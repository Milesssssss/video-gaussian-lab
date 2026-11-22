"""合成清单。kind 明确不是真实采集。"""

from __future__ import annotations


def minimal_frame(frame_id: str, split: str, timestamp_s: float, tx: float = 0.0) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_s": timestamp_s,
        "split": split,
        "rgb_path": f"data/{frame_id}.png",
        "rgb_sha256": "ab" * 32,
        "width_px": 4,
        "height_px": 4,
        "K": [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
        "T_world_camera": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "depth": {
            "path": None,
            "source": "none",
            "representation": "z_depth",
            "unit": "m",
            "sha256": None,
        },
        "timestamp_kind": "synthetic_exact",
        "pose_timestamp_s": timestamp_s,
        "pose_interpolation": "none",
        "tracking_status": "synthetic",
        "distortion_model": "none_synthetic_pinhole",
        "distortion_params": [],
        "image_transform": {"operations": ["identity"], "synthetic": True},
    }


def synthetic_manifest() -> dict:
    frames = [
        minimal_frame("src_000", "source", 0.0, 0.0),
        minimal_frame("src_001", "source", 0.2, 0.05),
        minimal_frame("val_000", "validation", 1.0, 0.2),
        minimal_frame("test_000", "test", 2.0, 0.4),
    ]
    return {
        "schema_version": "0.1",
        "kind": "synthetic_geometry_fixture",
        "scene_id": "synthetic_geometry_fixture_01",
        "scene_split": "dev",
        "units": {"distance": "m", "time": "s"},
        "camera_axes": "opencv_optical",
        "transform": "T_world_camera",
        "matrix_storage": "row_major",
        "vector_convention": "column_vector",
        "pixel_coordinates": "integer_center",
        "depth_representation": "z_depth",
        "camera_id": "synthetic_camera",
        "world_frame_id": "synthetic_world",
        "clock_domain": "synthetic_monotonic",
        "pose_status": "measured_or_estimated_fixed",
        "calibration_sha256": "cd" * 32,
        "pose_provider": {
            "name": "synthetic_fixture",
            "metric_scale": True,
            "used_eval_rgb": False,
            "timestamp_alignment_method": "synthetic_exact",
            "reference_geometry_available": False,
        },
        "preprocess": {
            "undistort": True,
            "operation_order": ["orientation", "undistort", "crop", "resize"],
            "canonical_width_px": 4,
            "canonical_height_px": 4,
            "K_is_for": "canonical_image",
            "color_space": "srgb",
        },
        "split_policy": {
            "status": "synthetic_fixture_not_a_capture",
            "frozen_before_inference": True,
            "source_frame_ids": ["src_000", "src_001"],
            "validation_frame_ids": ["val_000"],
            "test_frame_ids": ["test_000"],
            "eval_frames_in_network_context": False,
            "split_sha256": None,
        },
        "frames": frames,
    }
