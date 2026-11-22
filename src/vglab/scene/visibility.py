"""中心点可见性。occluded 不是错误，free-space conflict 才是反对证据。"""

from __future__ import annotations

import math

import numpy as np

from vglab.errors import ContractError


def classify_visibility(
    z_cam_m: float,
    u: float,
    v: float,
    *,
    width_px: int,
    height_px: int,
    z_ref_m: float,
    alpha: float,
    image_valid: bool,
    tau_abs_m: float,
    tau_rel: float,
    alpha_min: float,
) -> str:
    if tau_abs_m < 0 or tau_rel < 0 or alpha_min < 0 or alpha_min > 1:
        raise ContractError("可见性容差必须非负，alpha 门槛必须在 [0,1]")
    if (
        not image_valid
        or not math.isfinite(z_cam_m)
        or z_cam_m <= 0
        or not math.isfinite(u)
        or not math.isfinite(v)
        or u < 0
        or v < 0
        or u > width_px - 1
        or v > height_px - 1
    ):
        return "outside"
    if not math.isfinite(z_ref_m) or z_ref_m <= 0 or not math.isfinite(alpha) or alpha < alpha_min:
        return "unknown"
    tolerance = tau_abs_m + tau_rel * z_ref_m
    if z_cam_m < z_ref_m - tolerance:
        return "free_space_conflict"
    if z_cam_m > z_ref_m + tolerance:
        return "occluded"
    return "supported"


def tolerance_m(z_ref_m: float, tau_abs_m: float, tau_rel: float) -> float:
    if not np.isfinite(z_ref_m) or z_ref_m <= 0:
        raise ContractError("容差参考深度必须是有限正值")
    return float(tau_abs_m + tau_rel * z_ref_m)
