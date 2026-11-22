"""由 Z-depth 反投影的普通高斯。规则是未冻结候选，不是已测得的最优初始化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.calibration.camera import as_intrinsics, as_pose, backproject
from vglab.errors import ContractError
from vglab.initialization.candidates import UNFROZEN_ORDINARY_GAUSSIAN
from vglab.scene.gaussians import matrix_to_quaternion


@dataclass(frozen=True)
class OrdinaryInitialization:
    mean_world_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: np.ndarray
    color_linear_rgb: np.ndarray
    gaussian_id: np.ndarray
    frame_id: str
    config_name: str
    frozen: bool
    skipped_invalid: int


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    color = np.asarray(values, dtype=np.float64)
    if np.any((color < 0) | (color > 1)) or not np.isfinite(color).all():
        raise ContractError("sRGB 输入必须在 [0,1]")
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def initialize_ordinary_gaussians(
    z_depth_m: np.ndarray,
    rgb_srgb: np.ndarray,
    k_matrix: np.ndarray,
    t_world_camera: np.ndarray,
    valid: np.ndarray,
    frame_id: str,
    *,
    representation: str = "z_depth",
    unit: str = "m",
    config: dict | None = None,
) -> OrdinaryInitialization:
    if representation != "z_depth" or unit != "m":
        raise ContractError("普通高斯初始化只接受 unit=m 的 z_depth，range 必须先显式转换")
    chosen = UNFROZEN_ORDINARY_GAUSSIAN if config is None else config
    if chosen.get("frozen") is not False or chosen.get("normal_rule") != "camera_axes":
        raise ContractError("普通高斯配置必须显式 frozen=false，且当前只实现 camera_axes")
    depth = np.asarray(z_depth_m, dtype=np.float64)
    rgb = np.asarray(rgb_srgb, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or rgb.shape != (depth.shape[0], depth.shape[1], 3) or mask.shape != depth.shape:
        raise ContractError("深度、RGB 和有效掩码的网格必须一致")
    intrinsics = as_intrinsics(k_matrix)
    pose = as_pose(t_world_camera)
    usable = mask & np.isfinite(depth) & (depth > 0)
    vs, us = np.nonzero(usable)
    skipped = int(mask.sum() - usable.sum())
    means = []
    scales = []
    ids = []
    colors = []
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    footprint = float(chosen["footprint_px"])
    min_scale = float(chosen["min_scale_m"])
    if footprint <= 0 or min_scale <= 0:
        raise ContractError("footprint 和最小尺度必须为正")
    for v, u in zip(vs.tolist(), us.tolist(), strict=True):
        z_value = float(depth[v, u])
        point_c = backproject(intrinsics, float(u), float(v), z_value)
        means.append(pose[:3, :3] @ point_c + pose[:3, 3])
        lateral = np.array(
            [
                max(footprint * z_value / fx, min_scale),
                max(footprint * z_value / fy, min_scale),
                max(footprint * z_value / fx, min_scale),
            ]
        )
        scales.append(lateral)
        colors.append(srgb_to_linear(rgb[v, u]))
        ids.append(f"{frame_id}:{v}:{u}")
    count = len(means)
    rotation = np.repeat(matrix_to_quaternion(pose[:3, :3]), count, axis=0) if count else np.zeros((0, 4))
    opacity = np.full(count, float(chosen["opacity"]), dtype=np.float64)
    if np.any((opacity <= 0) | (opacity > 1)):
        raise ContractError("opacity 必须在 (0,1]，避免不可渲染的空点")
    return OrdinaryInitialization(
        mean_world_m=np.asarray(means, dtype=np.float64).reshape(count, 3),
        scale_std_m=np.asarray(scales, dtype=np.float64).reshape(count, 3),
        rotation_wxyz=rotation,
        opacity=opacity,
        color_linear_rgb=np.asarray(colors, dtype=np.float64).reshape(count, 3),
        gaussian_id=np.asarray(ids),
        frame_id=frame_id,
        config_name=str(chosen["name"]),
        frozen=False,
        skipped_invalid=skipped,
    )
