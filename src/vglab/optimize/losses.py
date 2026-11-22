"""协议中的 RGB 与深度损失。空监督不会被记成零损失。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError


@dataclass(frozen=True)
class DepthSupervision:
    loss: float | None
    status: str
    teacher_valid_px: int
    supervised_px: int
    coverage: float | None
    exclusion_counts: dict[str, int]


@dataclass(frozen=True)
class RgbLoss:
    value: float | None
    l1: float | None
    ssim: float | None
    status: str
    valid_px: int


def expected_depth(z_m: np.ndarray, alpha: np.ndarray, *, eps: float = 1e-8) -> float | None:
    """front-to-back 归一化期望深度。没有正权重时返回 None，而不是 0。"""
    depth = np.asarray(z_m, dtype=np.float64).reshape(-1)
    opacity = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if depth.shape != opacity.shape:
        raise ContractError("深度和 alpha 数量必须一致")
    order = np.argsort(depth, kind="mergesort")
    transmittance = 1.0
    mass = 0.0
    accum = 0.0
    for index in order:
        z_value = float(depth[index])
        alpha_value = float(opacity[index])
        if not np.isfinite(z_value) or z_value <= 0 or not np.isfinite(alpha_value) or alpha_value <= 0:
            continue
        alpha_value = min(alpha_value, 1.0)
        weight = transmittance * alpha_value
        accum += weight * z_value
        mass += weight
        transmittance *= 1.0 - alpha_value
    if mass <= 0.0:
        return None
    return float(accum / (mass + eps))


def depth_supervision(
    z_render_m: np.ndarray,
    z_dp2_m: np.ndarray,
    teacher_mask: np.ndarray,
    reliability: np.ndarray,
    renderer_mask: np.ndarray,
) -> DepthSupervision:
    render = np.asarray(z_render_m, dtype=np.float64)
    teacher_depth = np.asarray(z_dp2_m, dtype=np.float64)
    teacher = np.asarray(teacher_mask, dtype=bool)
    renderer = np.asarray(renderer_mask, dtype=bool)
    weight = np.asarray(reliability, dtype=np.float64)
    if render.shape != teacher_depth.shape or teacher.shape != render.shape or renderer.shape != render.shape or weight.shape != render.shape:
        raise ContractError("深度监督的各张图必须同网格")
    finite_teacher = teacher & np.isfinite(teacher_depth) & (teacher_depth > 0)
    finite_render = np.isfinite(render) & (render > 0)
    positive_weight = np.isfinite(weight) & (weight > 0)
    teacher_count = int(finite_teacher.sum())
    supervised = finite_teacher & finite_render & renderer & positive_weight
    exclusions = {
        "teacher_invalid_or_nonpositive": int((teacher & ~finite_teacher).sum()),
        "render_invalid_or_nonpositive": int((finite_teacher & ~finite_render).sum()),
        "renderer_mask_excluded": int((finite_teacher & finite_render & ~renderer).sum()),
        "nonpositive_reliability": int((finite_teacher & finite_render & renderer & ~positive_weight).sum()),
    }
    if teacher_count == 0:
        return DepthSupervision(None, "skipped_empty_depth_mask", 0, 0, None, exclusions)
    coverage = float(supervised.sum() / teacher_count)
    if int(supervised.sum()) == 0:
        return DepthSupervision(None, "skipped_empty_depth_mask", teacher_count, 0, coverage, exclusions)
    residual = np.abs(np.log(render[supervised]) - np.log(teacher_depth[supervised]))
    used = weight[supervised]
    loss = float(np.sum(used * residual) / np.sum(used))
    return DepthSupervision(loss, "computed", teacher_count, int(supervised.sum()), coverage, exclusions)


def rgb_loss(
    prediction_srgb: np.ndarray,
    target_srgb: np.ndarray,
    mask: np.ndarray,
    *,
    lambda_l1: float,
    lambda_ssim: float,
    window: int = 7,
    sigma: float = 1.5,
) -> RgbLoss:
    prediction = np.asarray(prediction_srgb, dtype=np.float64)
    target = np.asarray(target_srgb, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3 or valid.shape != prediction.shape[:2]:
        raise ContractError("RGB 损失要求 H×W×3 图像和 H×W 掩码")
    if np.any((prediction < 0) | (prediction > 1) | (target < 0) | (target > 1)):
        raise ContractError("PSNR/RGB 损失的 sRGB 输入必须在 [0,1]，无效像素不能填黑后冒充背景")
    if int(valid.sum()) == 0:
        return RgbLoss(None, None, None, "not_computed", 0)
    l1 = float(np.abs(prediction - target)[valid].mean())
    ssim_value = _masked_ssim(prediction, target, valid, window=window, sigma=sigma)
    if ssim_value is None:
        return RgbLoss(None, l1, None, "ssim_undefined", int(valid.sum()))
    value = float(lambda_l1 * l1 + lambda_ssim * (1.0 - ssim_value))
    return RgbLoss(value, l1, ssim_value, "computed", int(valid.sum()))


def _masked_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    window: int,
    sigma: float,
) -> float | None:
    if window < 3 or window % 2 == 0:
        raise ContractError("SSIM 窗口必须是不小于 3 的奇数")
    axis = np.arange(window, dtype=np.float64) - window // 2
    kernel_1d = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)
    scores: list[float] = []
    for channel in range(prediction.shape[2]):
        score = _ssim_channel(prediction[:, :, channel], target[:, :, channel], mask, kernel)
        if score is None:
            return None
        scores.append(score)
    return float(np.mean(scores))


def _ssim_channel(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> float | None:
    window = kernel.shape[0]
    height, width = prediction.shape
    if height < window or width < window:
        return None
    out_h = height - window + 1
    out_w = width - window + 1
    mu_x = np.zeros((out_h, out_w))
    mu_y = np.zeros((out_h, out_w))
    mu_x2 = np.zeros((out_h, out_w))
    mu_y2 = np.zeros((out_h, out_w))
    mu_xy = np.zeros((out_h, out_w))
    valid = np.zeros((out_h, out_w))
    for y in range(window):
        for x in range(window):
            weight = kernel[y, x]
            pred = prediction[y : y + out_h, x : x + out_w]
            ref = target[y : y + out_h, x : x + out_w]
            mu_x += weight * pred
            mu_y += weight * ref
            mu_x2 += weight * pred * pred
            mu_y2 += weight * ref * ref
            mu_xy += weight * pred * ref
            valid += mask[y : y + out_h, x : x + out_w]
    usable = valid == window * window
    if not np.any(usable):
        return None
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    sigma_x = mu_x2 - mu_x**2
    sigma_y = mu_y2 - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator
    return float(score[usable].mean())
