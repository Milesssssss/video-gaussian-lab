"""OpenCV 光学坐标下的投影、刚体位姿和图像网格变换。

矩阵按行存储，点是列向量。`T_world_camera` 把相机坐标变到世界坐标。
"""

from __future__ import annotations

import numpy as np

from vglab.errors import ContractError


def as_pose(transform: np.ndarray) -> np.ndarray:
    pose = np.asarray(transform, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ContractError("T_world_camera 必须是有限的 4×4 矩阵")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ContractError("T_world_camera 底行必须是 [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ContractError("T_world_camera 的旋转不是正交矩阵")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise ContractError("T_world_camera 必须是右手旋转，det(R)=+1")
    return pose


def as_intrinsics(k_matrix: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(k_matrix, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ContractError("K 必须是有限的 3×3 矩阵")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ContractError("K 的 fx、fy 必须为正")
    if abs(intrinsics[0, 1]) > 1e-8 or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ContractError("K 必须是无斜切针孔矩阵")
    return intrinsics


def invert_world_camera(t_world_camera: np.ndarray) -> np.ndarray:
    """返回 T_camera_world。传给 view matrix 时使用这个逆，而不是猜测 pose 方向。"""
    pose = as_pose(t_world_camera)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def project(k_matrix: np.ndarray, point_camera: np.ndarray) -> tuple[float, float, float]:
    intrinsics = as_intrinsics(k_matrix)
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all() or point[2] == 0:
        raise ContractError("相机坐标点必须有限且 z 不为 0")
    pixel = intrinsics @ point
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), float(point[2])


def backproject(k_matrix: np.ndarray, u: float, v: float, z_depth: float) -> np.ndarray:
    """由像素中心和光轴 Z-depth 反投影到相机坐标，单位米。"""
    intrinsics = as_intrinsics(k_matrix)
    if not np.isfinite(z_depth) or z_depth <= 0:
        raise ContractError("反投影要求有限正值 z_depth，单位 m")
    ray = np.linalg.inv(intrinsics) @ np.array([u, v, 1.0], dtype=np.float64)
    return z_depth * ray


def world_from_camera(t_world_camera: np.ndarray, point_camera: np.ndarray) -> np.ndarray:
    pose = as_pose(t_world_camera)
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    return pose[:3, :3] @ point + pose[:3, 3]


def camera_from_world(t_world_camera: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    inverse = invert_world_camera(t_world_camera)
    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    return inverse[:3, :3] @ point + inverse[:3, 3]


def arkit_to_optical(t_world_arkit_camera: np.ndarray) -> np.ndarray:
    """C=diag(1,-1,-1,1)，T_wc_optical = T_wc_ARKit @ C。

    只转换一次。调用方不得再对图像、位姿和渲染器重复翻转 y/z。
    """
    pose = as_pose(t_world_arkit_camera)
    optical_from_arkit = np.diag([1.0, -1.0, -1.0, 1.0])
    return pose @ optical_from_arkit


def rotate_intrinsics_clockwise_90(k_matrix: np.ndarray, height_px: int) -> np.ndarray:
    """原图顺时针转 90° 后的针孔 K。u' = H-1-v，v' = u。"""
    intrinsics = as_intrinsics(k_matrix)
    if not isinstance(height_px, int) or isinstance(height_px, bool) or height_px <= 0:
        raise ContractError("height_px 必须是正整数")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.array(
        [
            [fy, 0.0, height_px - 1 - cy],
            [0.0, fx, cx],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def image_rotation_clockwise_90(height_px: int) -> tuple[np.ndarray, np.ndarray]:
    """返回图像齐次变换 A 和相机旋转 Q，满足 K' Q = A K。"""
    if not isinstance(height_px, int) or isinstance(height_px, bool) or height_px <= 0:
        raise ContractError("height_px 必须是正整数")
    image_map = np.array(
        [
            [0.0, -1.0, height_px - 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return image_map, camera_rotation


def pose_after_clockwise_90(t_world_camera: np.ndarray, height_px: int) -> np.ndarray:
    """T'_wc = T_wc @ diag(Q^T, 1)。"""
    pose = as_pose(t_world_camera)
    _, camera_rotation = image_rotation_clockwise_90(height_px)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = camera_rotation.T
    return pose @ delta


def scale_intrinsics_half_pixel(
    k_matrix: np.ndarray,
    scale_x: float,
    scale_y: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> np.ndarray:
    """half-pixel 约定：u' = sx(u+0.5)-0.5，再从 (origin_x, origin_y) 裁剪。"""
    intrinsics = as_intrinsics(k_matrix)
    if scale_x <= 0 or scale_y <= 0 or not np.isfinite([scale_x, scale_y, origin_x, origin_y]).all():
        raise ContractError("缩放比例必须是有限正数")
    transform = np.array(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0 - origin_x],
            [0.0, scale_y, (scale_y - 1.0) / 2.0 - origin_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ intrinsics


def unit_ray(k_matrix: np.ndarray, u: float, v: float) -> np.ndarray:
    """K^{-1}[u,v,1]^T，并归一化到 r_z=1。"""
    intrinsics = as_intrinsics(k_matrix)
    ray = np.linalg.inv(intrinsics) @ np.array([u, v, 1.0], dtype=np.float64)
    if ray[2] == 0:
        raise ContractError("射线 r_z 不能为 0")
    return ray / ray[2]


def range_from_z_depth(k_matrix: np.ndarray, u: float, v: float, z_depth: float) -> float:
    if not np.isfinite(z_depth) or z_depth <= 0:
        raise ContractError("z_depth 必须是有限正值，单位 m")
    return float(z_depth * np.linalg.norm(unit_ray(k_matrix, u, v)))


def z_depth_from_range(k_matrix: np.ndarray, u: float, v: float, range_m: float) -> float:
    if not np.isfinite(range_m) or range_m <= 0:
        raise ContractError("range 必须是有限正值，单位 m")
    return float(range_m / np.linalg.norm(unit_ray(k_matrix, u, v)))
