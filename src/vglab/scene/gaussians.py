"""物理高斯参数。尺度是标准差，米；四元数顺序 w、x、y、z。"""

from __future__ import annotations

import numpy as np

from vglab.calibration.camera import as_pose
from vglab.errors import ContractError


def quaternion_to_matrix(rotation_wxyz: np.ndarray) -> np.ndarray:
    quats = np.atleast_2d(np.asarray(rotation_wxyz, dtype=np.float64))
    _validate_quaternions(quats)
    w, x, y, z = quats.T
    matrices = np.empty((len(quats), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - w * z)
    matrices[:, 0, 2] = 2 * (x * z + w * y)
    matrices[:, 1, 0] = 2 * (x * y + w * z)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - w * x)
    matrices[:, 2, 0] = 2 * (x * z - w * y)
    matrices[:, 2, 1] = 2 * (y * z + w * x)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotation, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    quats = np.empty((len(matrices), 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        quats[index] = _one_matrix_to_quaternion(matrix)
    return quats


def covariance(scale_std: np.ndarray, rotation_wxyz: np.ndarray) -> np.ndarray:
    scales = np.atleast_2d(np.asarray(scale_std, dtype=np.float64))
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ContractError("scale_std_m 必须是有限正数，单位 m")
    rotations = quaternion_to_matrix(rotation_wxyz)
    variance = scales**2
    return rotations @ (variance[:, :, None] * np.transpose(rotations, (0, 2, 1)))


def apply_metric_scale_and_pose(
    mean_camera_m: np.ndarray,
    scale_std_m: np.ndarray,
    rotation_wxyz: np.ndarray,
    scale: float | np.ndarray,
    t_world_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mu_w = R (s mu_c) + t，Sigma_w = s^2 R Sigma_c R^T。不缩放 t。"""
    means = np.atleast_2d(np.asarray(mean_camera_m, dtype=np.float64))
    scales = np.atleast_2d(np.asarray(scale_std_m, dtype=np.float64))
    scale_values = np.asarray(scale, dtype=np.float64)
    if scale_values.ndim == 0:
        scale_values = np.full(len(means), float(scale_values))
    if scale_values.shape != (len(means),) or np.any(scale_values <= 0) or not np.isfinite(scale_values).all():
        raise ContractError("尺度 s 必须是有限正数，且不能作用于相机平移")
    if not np.isfinite(means).all():
        raise ContractError("相机坐标均值必须有限")
    pose = as_pose(t_world_camera)
    rotation_wc = pose[:3, :3]
    translation = pose[:3, 3]
    mean_world = (means * scale_values[:, None]) @ rotation_wc.T + translation
    local = quaternion_to_matrix(rotation_wxyz)
    world_rotation = np.einsum("ij,njk->nik", rotation_wc, local)
    return mean_world, scales * scale_values[:, None], matrix_to_quaternion(world_rotation)


def raw_from_physical(scale_std_m: np.ndarray, opacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.asarray(scale_std_m, dtype=np.float64)
    alphas = np.asarray(opacity, dtype=np.float64)
    if np.any(scales <= 0) or np.any((alphas <= 0) | (alphas >= 1)) or not np.isfinite(scales).all():
        raise ContractError("raw 参数化要求正尺度和 (0,1) 内的 opacity")
    return np.log(scales), np.log(alphas / (1.0 - alphas))


def physical_from_raw(log_scale: np.ndarray, opacity_logit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logs = np.asarray(log_scale, dtype=np.float64)
    logits = np.asarray(opacity_logit, dtype=np.float64)
    if not np.isfinite(logs).all() or not np.isfinite(logits).all():
        raise ContractError("raw 参数必须有限；导入层只做一次 exp 和 sigmoid")
    return np.exp(logs), 1.0 / (1.0 + np.exp(-logits))


def _validate_quaternions(quats: np.ndarray) -> None:
    if quats.ndim != 2 or quats.shape[1] != 4 or not np.isfinite(quats).all():
        raise ContractError("rotation_wxyz 必须是有限的 (N,4)")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-5):
        raise ContractError("rotation_wxyz 必须是归一化四元数")


def _one_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat
