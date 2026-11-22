import numpy as np
import pytest

from vglab.errors import ContractError
from vglab.scene.gaussians import (
    apply_metric_scale_and_pose,
    covariance,
    matrix_to_quaternion,
    physical_from_raw,
    quaternion_to_matrix,
    raw_from_physical,
)


def test_scale_updates_mean_and_covariance_but_not_camera_translation():
    mean_c = np.array([1.0, 0.0, 2.0])
    scale_c = np.array([0.1, 0.2, 0.3])
    rotation = np.array([1.0, 0.0, 0.0, 0.0])
    pose = np.eye(4)
    pose[0, 3] = 10.0
    mean_w, scale_w, rotation_w = apply_metric_scale_and_pose(mean_c, scale_c, rotation, 2.0, pose)
    assert mean_w[0] == pytest.approx(np.array([12.0, 0.0, 4.0]))
    assert scale_w[0] == pytest.approx(np.array([0.2, 0.4, 0.6]))
    assert rotation_w[0] == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]))
    origin_w, _, _ = apply_metric_scale_and_pose(np.zeros(3), scale_c, rotation, 2.0, pose)
    assert origin_w[0] == pytest.approx(np.array([10.0, 0.0, 0.0]))
    forbidden = 2.0 * (mean_c + pose[:3, 3])
    assert not np.allclose(mean_w[0], forbidden)
    cov_c = covariance(scale_c, rotation)[0]
    cov_w = covariance(scale_w[0], rotation_w[0])[0]
    assert cov_w == pytest.approx(4.0 * cov_c)


def test_quaternion_roundtrip_and_raw_parameterization_is_single_use():
    quat = np.array([0.5, 0.5, 0.5, 0.5])
    quat = quat / np.linalg.norm(quat)
    recovered = matrix_to_quaternion(quaternion_to_matrix(quat))[0]
    assert recovered == pytest.approx(quat)
    log_scale, logit = raw_from_physical(np.array([0.2, 0.3, 0.4]), np.array([0.2]))
    scale, opacity = physical_from_raw(log_scale, logit)
    assert scale == pytest.approx(np.array([0.2, 0.3, 0.4]))
    assert opacity == pytest.approx(np.array([0.2]))
    doubled_scale, _ = physical_from_raw(scale, logit)
    assert not np.allclose(doubled_scale, scale)


def test_non_positive_scale_is_rejected():
    with pytest.raises(ContractError):
        apply_metric_scale_and_pose(
            np.zeros(3),
            np.ones(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            0.0,
            np.eye(4),
        )
