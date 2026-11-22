"""渲染指标。模型预测不能充当独立几何真值。"""

from __future__ import annotations

import math

import numpy as np

from vglab.errors import ContractError, IndependentReferenceError


def psnr_srgb(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float | None, int]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if pred.shape != ref.shape or valid.shape != pred.shape[:2]:
        raise ContractError("PSNR 的图像和掩码网格不一致")
    if np.any((pred < 0) | (pred > 1) | (ref < 0) | (ref > 1)):
        raise ContractError("PSNR 在 [0,1] sRGB 上计算")
    count = int(valid.sum())
    if count == 0:
        return None, 0
    mse = float(np.mean((pred - ref)[valid] ** 2))
    if mse == 0.0:
        return math.inf, count
    return float(-10.0 * math.log10(mse)), count


def coverage_fraction(opacity: np.ndarray, threshold: float) -> float:
    values = np.asarray(opacity, dtype=np.float64)
    if not np.isfinite(values).all() or not 0 <= threshold <= 1:
        raise ContractError("覆盖率阈值必须在 [0,1]，opacity 必须有限")
    return float(np.mean(values >= threshold))


def geometric_mae_m(prediction_m: np.ndarray, reference_m: np.ndarray, reference_source: str) -> float:
    if reference_source != "independent_reference":
        raise IndependentReferenceError("DP2 或 SHARP 预测不能作为独立几何真值")
    pred = np.asarray(prediction_m, dtype=np.float64)
    ref = np.asarray(reference_m, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(valid):
        raise ContractError("独立几何比较没有有效像素")
    return float(np.mean(np.abs(pred[valid] - ref[valid])))
