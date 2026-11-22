"""每个 source 帧一个标量尺度。失败时不截断、不改写高斯。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError, Dp2IsolationError, ScaleNotApplied

SHARP_SCALE_ARMS = ("S2", "S3")


@dataclass(frozen=True)
class ScaleEstimate:
    frame_id: str
    status: str
    sample_count: int
    s: float | None
    log_residual_mae: float | None
    outlier_fraction: float | None
    frozen: bool = False


def estimate_frame_scale(
    frame_id: str,
    z_sharp_m: np.ndarray,
    z_dp2_m: np.ndarray,
    mask: np.ndarray,
    *,
    arm: str,
    min_samples: int,
    s_min: float,
    s_max: float,
    outlier_log_abs: float,
) -> ScaleEstimate:
    if arm in {"S0", "S1"}:
        raise Dp2IsolationError(f"{arm} 是 SHARP-only 对照，不能读取 DP2 深度来估计尺度")
    if arm not in SHARP_SCALE_ARMS:
        raise ContractError(f"{arm} 不使用 SHARP 逐帧标量尺度")
    sharp = np.asarray(z_sharp_m, dtype=np.float64)
    dp2 = np.asarray(z_dp2_m, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(sharp) & np.isfinite(dp2) & (sharp > 0) & (dp2 > 0)
    count = int(valid.sum())
    if count < min_samples:
        return ScaleEstimate(frame_id, "insufficient_samples", count, None, None, None)
    log_ratio = np.log(dp2[valid]) - np.log(sharp[valid])
    ell = float(np.median(log_ratio))
    estimate = float(np.exp(ell))
    residual = np.abs(log_ratio - ell)
    mae = float(np.mean(residual))
    outlier_fraction = float(np.mean(residual > outlier_log_abs))
    if not np.isfinite(estimate) or estimate < s_min or estimate > s_max:
        reported = estimate if np.isfinite(estimate) else None
        return ScaleEstimate(frame_id, "scale_out_of_range", count, reported, mae, outlier_fraction)
    return ScaleEstimate(frame_id, "ok", count, estimate, mae, outlier_fraction)


def require_applicable_scale(estimate: ScaleEstimate) -> float:
    if estimate.status != "ok" or estimate.s is None:
        raise ScaleNotApplied(f"{estimate.frame_id} 尺度状态为 {estimate.status}，不改写高斯")
    return estimate.s
