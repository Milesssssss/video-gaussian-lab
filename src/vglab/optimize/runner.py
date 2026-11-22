"""共享优化的执行守卫。

可训练参数不含 K/T。未注入已验证渲染器时，状态是未运行，不是优化成功。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError

TRAINABLE = ("mean", "log_scale", "rotation", "opacity_logit", "color")


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    arm: str
    densification: bool
    pose_updated: bool
    measurements: None = None


class SharedOptimizer:
    def __init__(self, arm: str, poses: np.ndarray):
        if arm not in {"S0", "S1", "S2", "S3", "D0", "D1"}:
            raise ContractError(f"未知实验臂 {arm}")
        self.arm = arm
        self.poses = np.array(poses, dtype=np.float64, copy=True)
        self._pose_snapshot = self.poses.copy()
        self.densification = False
        self.trainable = TRAINABLE

    def run(self) -> OptimizationResult:
        if not np.array_equal(self.poses, self._pose_snapshot):
            raise ContractError("优化过程中相机位姿被改写")
        if self.densification:
            raise ContractError("Phase 1 主矩阵关闭 densification")
        if set(self.trainable) & {"pose", "T", "K"}:
            raise ContractError("K/T 不可训练")
        if self.arm == "S0":
            return OptimizationResult("diagnostic_only", self.arm, False, False, None)
        return OptimizationResult("not_run_no_verified_renderer", self.arm, False, False, None)
