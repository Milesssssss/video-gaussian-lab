"""把 SHARP 相机坐标高斯放进世界。S0/S1 不接受尺度字典。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError, Dp2IsolationError
from vglab.scene.gaussians import apply_metric_scale_and_pose

SHARP_INTERNAL_SQUARE_PX = 1536
SHARP_GAUSSIAN_GRID = (768, 768)
SHARP_LAYER_COUNT = 2


@dataclass(frozen=True)
class SharpSamplingRecord:
    canonical_width_px: int
    canonical_height_px: int
    internal_square_px: int = SHARP_INTERNAL_SQUARE_PX
    gaussian_grid: tuple[int, int] = SHARP_GAUSSIAN_GRID


@dataclass(frozen=True)
class CameraGaussian:
    gaussian_id: str
    frame_id: str
    layer_id: int
    mean_camera_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: float
    color_linear_rgb: np.ndarray
    source_grid_uv: tuple[int, int]
    source_grid_shape: tuple[int, int]
    original_index: int


@dataclass(frozen=True)
class PlacedGaussian:
    gaussian_id: str
    frame_id: str
    layer_id: int
    mean_world_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: float
    color_linear_rgb: np.ndarray
    applied_scale: float


def sampling_record(canonical_width_px: int, canonical_height_px: int) -> SharpSamplingRecord:
    if canonical_width_px <= 0 or canonical_height_px <= 0:
        raise ContractError("canonical 尺寸必须为正")
    return SharpSamplingRecord(canonical_width_px, canonical_height_px)


def place_sharp_gaussians(
    gaussians: list[CameraGaussian],
    t_world_camera_by_frame: dict[str, np.ndarray],
    *,
    arm: str,
    scale_by_frame: dict[str, float] | None = None,
) -> list[PlacedGaussian]:
    if arm in {"S0", "S1"} and scale_by_frame is not None:
        raise Dp2IsolationError(f"{arm} 不能使用 DP2 尺度")
    if arm in {"S2", "S3"} and scale_by_frame is None:
        raise ContractError(f"{arm} 需要每个 source 帧的已接受尺度")
    if arm not in {"S0", "S1", "S2", "S3"}:
        raise ContractError("SHARP 放置只用于 S0/S1/S2/S3")
    placed: list[PlacedGaussian] = []
    for gaussian in gaussians:
        if gaussian.layer_id not in {0, 1}:
            raise ContractError("SHARP layer_id 只能是 0 或 1")
        scale = 1.0 if scale_by_frame is None else float(scale_by_frame[gaussian.frame_id])
        mean_w, scale_w, rotation_w = apply_metric_scale_and_pose(
            gaussian.mean_camera_m,
            gaussian.scale_std_m,
            gaussian.rotation_wxyz,
            scale,
            t_world_camera_by_frame[gaussian.frame_id],
        )
        placed.append(
            PlacedGaussian(
                gaussian_id=gaussian.gaussian_id,
                frame_id=gaussian.frame_id,
                layer_id=gaussian.layer_id,
                mean_world_m=mean_w[0],
                scale_std_m=scale_w[0],
                rotation_wxyz=rotation_w[0],
                opacity=gaussian.opacity,
                color_linear_rgb=np.asarray(gaussian.color_linear_rgb, dtype=np.float64),
                applied_scale=scale,
            )
        )
    return placed
