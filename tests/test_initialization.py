import numpy as np
import pytest

from vglab.errors import ContractError, Dp2IsolationError
from vglab.initialization.ordinary import initialize_ordinary_gaussians, srgb_to_linear
from vglab.initialization.sharp_scene import (
    SHARP_INTERNAL_SQUARE_PX,
    CameraGaussian,
    place_sharp_gaussians,
    sampling_record,
)


def test_ordinary_init_uses_pixel_footprint_and_does_not_scale_translation():
    depth = np.full((4, 4), np.nan)
    depth[2, 2] = 2.0
    depth[0, 0] = -1.0
    rgb = np.zeros((4, 4, 3))
    rgb[2, 2] = (1.0, 0.0, 0.0)
    intrinsics = np.array([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[0, 3] = 1.0
    valid = np.zeros((4, 4), dtype=bool)
    valid[2, 2] = True
    valid[0, 0] = True
    scene = initialize_ordinary_gaussians(depth, rgb, intrinsics, pose, valid, "src_000")
    assert scene.frozen is False
    assert scene.config_name == "pixel_footprint_v0"
    assert scene.skipped_invalid == 1
    assert len(scene.gaussian_id) == 1
    assert scene.mean_world_m[0] == pytest.approx(np.array([1.0, 0.0, 2.0]))
    assert scene.scale_std_m[0] == pytest.approx(np.array([0.5, 0.5, 0.5]))
    assert scene.opacity[0] == pytest.approx(0.8)
    assert scene.color_linear_rgb[0, 0] == pytest.approx(1.0)
    assert scene.rotation_wxyz[0] == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ContractError):
        initialize_ordinary_gaussians(depth, rgb, intrinsics, pose, valid, "src_000", representation="range")


def test_srgb_breakpoint_matches_iec_conversion():
    assert srgb_to_linear(np.array([0.0, 0.04045, 1.0])) == pytest.approx(
        np.array([0.0, 0.04045 / 12.92, 1.0])
    )


def test_sharp_sampling_does_not_replace_canonical_size():
    record = sampling_record(1280, 720)
    assert record.canonical_width_px == 1280
    assert record.internal_square_px == SHARP_INTERNAL_SQUARE_PX == 1536
    assert record.gaussian_grid == (768, 768)


def test_sharp_placement_scales_s2_and_blocks_s1():
    gaussian = CameraGaussian(
        gaussian_id="g0",
        frame_id="src_000",
        layer_id=0,
        mean_camera_m=np.array([1.0, 0.0, 2.0]),
        scale_std_m=np.ones(3),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        opacity=0.5,
        color_linear_rgb=np.array([0.1, 0.2, 0.3]),
        source_grid_uv=(1, 2),
        source_grid_shape=(4, 4),
        original_index=7,
    )
    pose = np.eye(4)
    pose[0, 3] = 10.0
    placed = place_sharp_gaussians([gaussian], {"src_000": pose}, arm="S2", scale_by_frame={"src_000": 2.0})
    assert placed[0].mean_world_m == pytest.approx(np.array([12.0, 0.0, 4.0]))
    assert placed[0].scale_std_m == pytest.approx(np.array([2.0, 2.0, 2.0]))
    assert placed[0].opacity == pytest.approx(0.5)
    assert placed[0].layer_id == 0
    with pytest.raises(Dp2IsolationError):
        place_sharp_gaussians([gaussian], {"src_000": pose}, arm="S1", scale_by_frame={"src_000": 2.0})
    unchanged = place_sharp_gaussians([gaussian], {"src_000": pose}, arm="S0")
    assert unchanged[0].applied_scale == pytest.approx(1.0)
    assert unchanged[0].mean_world_m == pytest.approx(np.array([11.0, 0.0, 2.0]))
