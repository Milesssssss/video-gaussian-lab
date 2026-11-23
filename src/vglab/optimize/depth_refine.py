"""S3/D1 的开发用深度约束精化。

只更新光学轴方向的高斯均值（可选 opacity logit），不更新 K/T、尺度、旋转、颜色，
不做 densification。使用参考渲染器与固定的 teacher 掩码/q。

状态字符串只说"参考链路跑过"，不是实测结果。正式实验仍被协议拒绝。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from vglab.calibration.camera import as_pose
from vglab.errors import ContractError, Dp2IsolationError
from vglab.initialization.candidates import UNFROZEN_DEPTH_REFINER
from vglab.optimize.losses import depth_supervision_with_gradient
from vglab.render.reference import (
    ReferenceCamera,
    ReferenceScene,
    depth_render_gradients,
    depth_render_step,
    render_reference,
)

DEPTH_LOSS_ARMS = ("S3", "D1")


@dataclass(frozen=True)
class DepthRefineConfig:
    lr_log_step: float = 0.35
    log_step_cap: float = 0.5
    alpha_min: float = 0.1
    max_steps: int = 25
    min_coverage: float = 0.5
    patience: int = 5
    update_opacity: bool = False
    lr_opacity: float = 0.02
    opacity_logit_clip: float = 12.0
    name: str = "log_step_optical_axis_v0"
    frozen: bool = False

    @classmethod
    def from_mapping(cls, payload: dict) -> DepthRefineConfig:
        if payload.get("frozen") is not False:
            raise ContractError("深度精化配置必须显式 frozen=false")
        return cls(
            lr_log_step=float(payload["lr_log_step"]),
            log_step_cap=float(payload["log_step_cap"]),
            alpha_min=float(payload["alpha_min"]),
            max_steps=int(payload["max_steps"]),
            min_coverage=float(payload["min_coverage"]),
            patience=int(payload["patience"]),
            update_opacity=bool(payload["update_opacity"]),
            lr_opacity=float(payload["lr_opacity"]),
            opacity_logit_clip=float(payload["opacity_logit_clip"]),
            name=str(payload["name"]),
            frozen=False,
        )


@dataclass(frozen=True)
class DepthRefineResult:
    status: str
    arm: str
    steps_run: int
    initial_loss: float | None
    final_loss: float | None
    initial_coverage: float | None
    final_coverage: float | None
    coverage_history: tuple[float, ...]
    loss_history: tuple[float | None, ...]
    stop_reason: str
    teacher_mask_sha256: str
    gaussian_count: int
    pose_updated: bool = False
    densification: bool = False
    measurements: None = None


class DepthConstraintRefiner:
    """固定 teacher、固定相机、固定其它高斯参数的深度残差精化。"""

    def __init__(
        self,
        arm: str,
        scene: ReferenceScene,
        camera: ReferenceCamera,
        teacher,
        *,
        config: DepthRefineConfig | None = None,
    ):
        if arm in {"S0", "S1"}:
            raise Dp2IsolationError(f"{arm} 是 SHARP-only/无深度对照，不能读取 DP2 teacher")
        if arm not in DEPTH_LOSS_ARMS:
            raise ContractError(f"{arm} 没有 depth loss；只有 S3/D1 可以进入深度精化")
        chosen = UNFROZEN_DEPTH_REFINER if config is None else config
        if isinstance(chosen, dict):
            chosen = DepthRefineConfig.from_mapping(chosen)
        self._validate(chosen)
        self.arm = arm
        self.config = chosen
        self.camera = ReferenceCamera(
            t_world_camera=np.array(camera.t_world_camera, dtype=np.float64, copy=True),
            k_matrix=np.array(camera.k_matrix, dtype=np.float64, copy=True),
            width_px=camera.width_px,
            height_px=camera.height_px,
        )
        self.means = np.array(scene.mean_world_m, dtype=np.float64, copy=True)
        self.scales = np.array(scene.scale_std_m, dtype=np.float64, copy=True)
        self.rotations = np.array(scene.rotation_wxyz, dtype=np.float64, copy=True)
        self.colors = np.array(scene.color_linear_rgb, dtype=np.float64, copy=True)
        opacity = np.asarray(scene.opacity, dtype=np.float64).reshape(-1)
        if np.any((opacity <= 0) | (opacity >= 1)):
            raise ContractError("深度精化要求 opacity 在 (0,1)，避免 alpha=1 后无梯度")
        self.opacity_logit = np.log(opacity / (1.0 - opacity))
        self.teacher_depth = np.array(teacher.depth_m, dtype=np.float64, copy=True)
        self.teacher_mask = np.array(teacher.mask, dtype=bool, copy=True)
        self.teacher_reliability = np.array(teacher.reliability, dtype=np.float64, copy=True)
        if self.teacher_depth.shape != (self.camera.height_px, self.camera.width_px):
            raise ContractError("teacher 网格必须与参考相机一致")
        if self.teacher_mask.shape != self.teacher_depth.shape or self.teacher_reliability.shape != self.teacher_depth.shape:
            raise ContractError("teacher 掩码与可靠性必须与深度同网格")
        self._teacher_mask_sha256 = _digest_mask(self.teacher_mask)
        self._pose_snapshot = np.array(self.camera.t_world_camera, dtype=np.float64, copy=True)
        self._k_snapshot = np.array(self.camera.k_matrix, dtype=np.float64, copy=True)

    def _validate(self, config: DepthRefineConfig) -> None:
        if config.lr_log_step <= 0 or config.log_step_cap <= 0:
            raise ContractError("深度精化步长必须为正")
        if not 0 <= config.alpha_min <= 1 or not 0 <= config.min_coverage <= 1:
            raise ContractError("alpha 与覆盖率门槛必须在 [0,1]")
        if config.max_steps <= 0 or config.patience < 0:
            raise ContractError("步数与耐心参数非法")
        if config.lr_opacity < 0 or config.opacity_logit_clip <= 0:
            raise ContractError("opacity 更新参数非法")

    def _scene(self) -> ReferenceScene:
        return ReferenceScene(
            mean_world_m=self.means,
            scale_std_m=self.scales,
            rotation_wxyz=self.rotations,
            opacity=1.0 / (1.0 + np.exp(-self.opacity_logit)),
            color_linear_rgb=self.colors,
        )

    def current_scene(self) -> ReferenceScene:
        """只读快照，供调用方渲染最终场景；不暴露可变数组。"""
        return ReferenceScene(
            mean_world_m=np.array(self.means, copy=True),
            scale_std_m=np.array(self.scales, copy=True),
            rotation_wxyz=np.array(self.rotations, copy=True),
            opacity=1.0 / (1.0 + np.exp(-self.opacity_logit)),
            color_linear_rgb=np.array(self.colors, copy=True),
        )

    def run(self) -> DepthRefineResult:
        config = self.config
        axis_world = as_pose(self.camera.t_world_camera)[:3, 2]
        coverage_history: list[float] = []
        loss_history: list[float | None] = []
        initial_loss: float | None = None
        initial_coverage: float | None = None
        final_loss: float | None = None
        final_coverage: float | None = None
        steps_run = 0
        low_streak = 0
        stop_reason = "max_steps"
        for step in range(config.max_steps):
            render = render_reference(self._scene(), self.camera, collect_records=True)
            renderer_mask = np.asarray(render.alpha >= config.alpha_min, dtype=bool)
            outcome = depth_supervision_with_gradient(
                render.z_expected_m,
                self.teacher_depth,
                self.teacher_mask,
                self.teacher_reliability,
                renderer_mask,
            )
            supervision = outcome.supervision
            if step == 0:
                initial_loss = supervision.loss
                initial_coverage = supervision.coverage
            if supervision.loss is None:
                loss_history.append(None)
                coverage_history.append(float(supervision.coverage or 0.0))
                stop_reason = "empty_supervision"
                break
            loss_history.append(supervision.loss)
            coverage = float(supervision.coverage or 0.0)
            coverage_history.append(coverage)
            low_streak = low_streak + 1 if coverage < config.min_coverage else 0

            residual = np.zeros_like(self.teacher_depth)
            supervised = (
                self.teacher_mask
                & np.isfinite(self.teacher_depth)
                & (self.teacher_depth > 0)
                & np.isfinite(render.z_expected_m)
                & (render.z_expected_m > 0)
                & renderer_mask
                & np.isfinite(self.teacher_reliability)
                & (self.teacher_reliability > 0)
            )
            residual[supervised] = np.log(render.z_expected_m[supervised]) - np.log(self.teacher_depth[supervised])
            log_step = np.zeros_like(residual)
            log_step[supervised] = -config.lr_log_step * np.clip(
                residual[supervised], -config.log_step_cap, config.log_step_cap
            )
            desired_delta = log_step * np.where(np.isfinite(render.z_expected_m), render.z_expected_m, 0.0)
            delta_z = depth_render_step(
                render.records,
                desired_delta,
                num_gaussians=len(self.means),
            )
            self.means += delta_z[:, None] * axis_world[None, :]
            if config.update_opacity:
                _, gradient_alpha = depth_render_gradients(
                    render.records,
                    outcome.grad_z_render,
                    num_gaussians=len(self.means),
                    z_render_m=render.z_expected_m,
                )
                alpha = 1.0 / (1.0 + np.exp(-self.opacity_logit))
                self.opacity_logit -= config.lr_opacity * gradient_alpha * alpha * (1.0 - alpha)
                self.opacity_logit = np.clip(
                    self.opacity_logit, -config.opacity_logit_clip, config.opacity_logit_clip
                )
            steps_run += 1
            if low_streak > config.patience:
                stop_reason = "low_coverage"
                break

        if stop_reason != "empty_supervision":
            final = render_reference(self._scene(), self.camera)
            final_outcome = depth_supervision_with_gradient(
                final.z_expected_m,
                self.teacher_depth,
                self.teacher_mask,
                self.teacher_reliability,
                np.asarray(final.alpha >= config.alpha_min, dtype=bool),
            )
            final_loss = final_outcome.supervision.loss
            final_coverage = final_outcome.supervision.coverage
        if np.any(self.camera.t_world_camera != self._pose_snapshot) or np.any(self.camera.k_matrix != self._k_snapshot):
            raise ContractError("深度精化过程中固定 K/T 被改写")
        if _digest_mask(self.teacher_mask) != self._teacher_mask_sha256:
            raise ContractError("teacher 固定掩码在优化中被改写")
        if stop_reason == "empty_supervision":
            status = "skipped_empty_depth_supervision" if steps_run == 0 else "stopped_empty_depth_supervision"
        elif stop_reason == "low_coverage":
            status = "stopped_low_coverage"
        elif steps_run == 0:
            status = "skipped_low_coverage"
        else:
            status = "ran_reference_depth_pass"
        return DepthRefineResult(
            status=status,
            arm=self.arm,
            steps_run=steps_run,
            initial_loss=initial_loss,
            final_loss=final_loss,
            initial_coverage=initial_coverage,
            final_coverage=final_coverage,
            coverage_history=tuple(coverage_history),
            loss_history=tuple(loss_history),
            stop_reason=stop_reason,
            teacher_mask_sha256=self._teacher_mask_sha256,
            gaussian_count=len(self.means),
        )


def _digest_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()
