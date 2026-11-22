import numpy as np
import pytest

from vglab.calibration.camera import (
    arkit_to_optical,
    backproject,
    camera_from_world,
    image_rotation_clockwise_90,
    invert_world_camera,
    pose_after_clockwise_90,
    project,
    range_from_z_depth,
    rotate_intrinsics_clockwise_90,
    scale_intrinsics_half_pixel,
    world_from_camera,
    z_depth_from_range,
)
from vglab.errors import ContractError


def _k() -> np.ndarray:
    return np.array([[100.0, 0.0, 2.0], [0.0, 80.0, 1.0], [0.0, 0.0, 1.0]])


def test_backproject_roundtrip_and_image_right_is_positive_x():
    intrinsics = _k()
    point = backproject(intrinsics, 12.0, 5.0, 2.0)
    u, v, z_depth = project(intrinsics, point)
    assert (u, v, z_depth) == pytest.approx((12.0, 5.0, 2.0))
    right = backproject(intrinsics, intrinsics[0, 2] + 10.0, intrinsics[1, 2], 1.0)
    assert right[0] > 0
    assert right[2] == pytest.approx(1.0)
    down = backproject(intrinsics, intrinsics[0, 2], intrinsics[1, 2] + 8.0, 1.0)
    assert down[1] > 0


def test_world_camera_inverse_matches_column_vector_convention():
    pose = np.array(
        [
            [0.0, 0.0, 1.0, 0.4],
            [0.0, 1.0, 0.0, -0.2],
            [-1.0, 0.0, 0.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    point_c = np.array([0.1, -0.2, 2.0])
    point_w = world_from_camera(pose, point_c)
    assert camera_from_world(pose, point_w) == pytest.approx(point_c)
    inverse = invert_world_camera(pose)
    assert inverse @ pose == pytest.approx(np.eye(4))


def test_arkit_conversion_flips_y_and_z_once():
    pose = np.eye(4)
    optical = arkit_to_optical(pose)
    assert optical[:3, :3] == pytest.approx(np.diag([1.0, -1.0, -1.0]))
    front = world_from_camera(optical, np.array([0.0, 0.0, 1.0]))
    assert front == pytest.approx(np.array([0.0, 0.0, -1.0]))
    image_down = world_from_camera(optical, np.array([0.0, 1.0, 1.0]))
    assert image_down[1] < 0


def test_clockwise_rotation_preserves_k_q_equals_a_k():
    intrinsics = _k()
    height = 6
    rotated = rotate_intrinsics_clockwise_90(intrinsics, height)
    image_map, camera_rotation = image_rotation_clockwise_90(height)
    assert rotated @ camera_rotation == pytest.approx(image_map @ intrinsics)
    assert rotated[0, 0] == pytest.approx(intrinsics[1, 1])
    assert rotated[1, 1] == pytest.approx(intrinsics[0, 0])
    assert rotated[0, 2] == pytest.approx(height - 1 - intrinsics[1, 2])
    pose = pose_after_clockwise_90(np.eye(4), height)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)


def test_half_pixel_resize_uses_the_contract_formula():
    intrinsics = np.array([[10.0, 0.0, 3.0], [0.0, 12.0, 4.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics_half_pixel(intrinsics, 2.0, 2.0, origin_x=1.0, origin_y=0.0)
    assert scaled[0, 0] == pytest.approx(20.0)
    assert scaled[1, 1] == pytest.approx(24.0)
    assert scaled[0, 2] == pytest.approx(2.0 * 3.0 + (2.0 - 1.0) / 2.0 - 1.0)
    assert scaled[1, 2] == pytest.approx(2.0 * 4.0 + 0.5)


def test_range_equals_z_on_axis_and_is_larger_off_axis():
    intrinsics = _k()
    center_u, center_v = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    assert range_from_z_depth(intrinsics, center_u, center_v, 2.0) == pytest.approx(2.0)
    corner = range_from_z_depth(intrinsics, center_u + 30.0, center_v + 20.0, 2.0)
    assert corner > 2.0
    recovered = z_depth_from_range(intrinsics, center_u + 30.0, center_v + 20.0, corner)
    assert recovered == pytest.approx(2.0)


def test_non_positive_z_and_reflection_are_rejected():
    with pytest.raises(ContractError):
        backproject(_k(), 0.0, 0.0, 0.0)
    reflected = np.diag([1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ContractError):
        world_from_camera(reflected, np.array([0.0, 0.0, 1.0]))
