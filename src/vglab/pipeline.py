"""把已实现的几何步骤接到实验臂上。

正式协议运行仍走 protocol.assert_official_run_allowed。这里的结果不是 P1a/P1b 记录。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError, Dp2IsolationError, OfficialRunRefused
from vglab.initialization.scale import estimate_frame_scale, require_applicable_scale
from vglab.initialization.sharp_scene import CameraGaussian, PlacedGaussian, place_sharp_gaussians
from vglab.optimize.runner import OptimizationResult, SharedOptimizer
from vglab.protocol import ArmSpec, assert_official_run_allowed


@dataclass(frozen=True)
class ArmInitialization:
    arm: str
    placed: tuple[PlacedGaussian, ...]
    optimization: OptimizationResult
    measurements: None = None


def run_official(protocol: dict) -> None:
    assert_official_run_allowed(protocol)
    raise OfficialRunRefused("正式运行入口没有已验证的模型执行器")


def initialize_sharp_arm(
    spec: ArmSpec,
    gaussians: list[CameraGaussian],
    poses_by_frame: dict[str, np.ndarray],
    *,
    z_sharp: dict[str, np.ndarray] | None = None,
    z_dp2: dict[str, np.ndarray] | None = None,
    scale_mask: dict[str, np.ndarray] | None = None,
    min_samples: int = 1,
    s_min: float = 0.25,
    s_max: float = 4.0,
    outlier_log_abs: float = 1.0,
) -> ArmInitialization:
    if spec.initializer != "sharp":
        raise ContractError("initialize_sharp_arm 只接受 SHARP 臂")
    if spec.sharp_only and (z_dp2 is not None or z_sharp is not None):
        raise Dp2IsolationError(f"{spec.arm_id} 不能接收深度图")
    scale_by_frame = None
    if spec.depth_scale:
        if z_sharp is None or z_dp2 is None or scale_mask is None:
            raise ContractError(f"{spec.arm_id} 缺少尺度拟合输入")
        scale_by_frame = {}
        for frame_id, sharp_depth in z_sharp.items():
            estimate = estimate_frame_scale(
                frame_id,
                sharp_depth,
                z_dp2[frame_id],
                scale_mask[frame_id],
                arm=spec.arm_id,
                min_samples=min_samples,
                s_min=s_min,
                s_max=s_max,
                outlier_log_abs=outlier_log_abs,
            )
            scale_by_frame[frame_id] = require_applicable_scale(estimate)
    placed = place_sharp_gaussians(gaussians, poses_by_frame, arm=spec.arm_id, scale_by_frame=scale_by_frame)
    poses = np.stack([np.asarray(poses_by_frame[item.frame_id], dtype=np.float64) for item in placed]) if placed else np.zeros((0, 4, 4))
    if spec.rgb_optimization:
        optimization = SharedOptimizer(spec.arm_id, poses).run()
    else:
        optimization = OptimizationResult("diagnostic_only", spec.arm_id, False, False, None)
    return ArmInitialization(spec.arm_id, tuple(placed), optimization, None)
