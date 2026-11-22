"""确定性准入。几何来源只能是该臂自己的初始化，不能直接读 DP2。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vglab.errors import ContractError, Dp2IsolationError

PROTOCOL_BUDGETS = (250000, 500000, 1000000)


@dataclass(frozen=True)
class GaussianCandidate:
    gaussian_id: str
    mean_world_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: float
    color_linear_rgb: np.ndarray
    source_frame_id: str
    source_time_s: float
    layer_id: int | None
    support_count: int
    free_space_conflict: bool


@dataclass(frozen=True)
class AdmissionResult:
    kept: tuple[GaussianCandidate, ...]
    rejected: tuple[tuple[str, str], ...]
    count_before: int
    count_after: int

    @property
    def cloned(self) -> bool:
        return len({item.gaussian_id for item in self.kept}) != len(self.kept)


def admit(
    candidates: list[GaussianCandidate],
    *,
    arm: str,
    budget: int,
    new_region_fraction: float,
    geometry_source: str,
    dedup_radius_m: float,
    max_scale_m: float,
    protocol_budget: bool = False,
    orientation_dot_min: float = 0.9659258262890683,
    color_l2_max: float = 0.15,
) -> AdmissionResult:
    if geometry_source != "arm_initialization":
        raise Dp2IsolationError(f"{arm} 的准入几何来源是 {geometry_source}；只能使用 arm_initialization，不能读取 DP2")
    if arm not in {"S0", "S1", "S2", "S3", "D0", "D1"}:
        raise ContractError(f"未知实验臂 {arm}")
    if budget <= 0:
        raise ContractError("budget 必须为正")
    if protocol_budget and budget not in PROTOCOL_BUDGETS:
        raise ContractError("正式预算只能是 250000、500000 或 1000000")
    if not 0 <= new_region_fraction <= 1 or dedup_radius_m < 0 or max_scale_m <= 0:
        raise ContractError("准入参数非法")
    if not 0 <= orientation_dot_min <= 1 or color_l2_max < 0:
        raise ContractError("对应确认阈值非法；这些数值尚未冻结")
    rejected: list[tuple[str, str]] = []
    valid: list[GaussianCandidate] = []
    for candidate in candidates:
        reason = _invalid_reason(candidate, max_scale_m)
        if reason is not None:
            rejected.append((candidate.gaussian_id, reason))
        elif candidate.free_space_conflict:
            rejected.append((candidate.gaussian_id, "free_space_conflict"))
        else:
            valid.append(candidate)
    ranked = sorted(valid, key=_rank_key)
    deduped: list[GaussianCandidate] = []
    cell = dedup_radius_m if dedup_radius_m > 0 else 1.0
    buckets: dict[tuple[int, int, int], list[GaussianCandidate]] = {}
    for candidate in ranked:
        index = _cell_index(candidate.mean_world_m, cell)
        neighbors: list[GaussianCandidate] = []
        for key in _neighbor_cells(index):
            neighbors.extend(buckets.get(key, ()))
        if any(
            _confirmed_same_surface(
                candidate,
                kept,
                dedup_radius_m,
                orientation_dot_min,
                color_l2_max,
            )
            for kept in neighbors
        ):
            rejected.append((candidate.gaussian_id, "duplicate_represented"))
        else:
            deduped.append(candidate)
            buckets.setdefault(index, []).append(candidate)
    selected = _select_with_new_region_quota(deduped, budget, new_region_fraction)
    selected_ids = {item.gaussian_id for item in selected}
    for candidate in deduped:
        if candidate.gaussian_id not in selected_ids:
            rejected.append((candidate.gaussian_id, "budget"))
    return AdmissionResult(tuple(selected), tuple(rejected), len(candidates), len(selected))


def _select_with_new_region_quota(
    deduped: list[GaussianCandidate],
    budget: int,
    new_region_fraction: float,
) -> list[GaussianCandidate]:
    quota = min(budget, int(math.floor(budget * new_region_fraction)))
    supported = [item for item in deduped if item.support_count >= 1]
    novel = [item for item in deduped if item.support_count < 1]
    picked = supported[: budget - quota]
    picked.extend(novel[:quota])
    picked_ids = {item.gaussian_id for item in picked}
    leftovers = [item for item in deduped if item.gaussian_id not in picked_ids]
    picked.extend(leftovers[: budget - len(picked)])
    return picked


def _rank_key(candidate: GaussianCandidate) -> tuple:
    group = 0 if candidate.support_count >= 1 else 1
    return (group, -candidate.support_count, candidate.source_time_s, candidate.source_frame_id, candidate.gaussian_id)


def _invalid_reason(candidate: GaussianCandidate, max_scale_m: float) -> str | None:
    mean = np.asarray(candidate.mean_world_m, dtype=np.float64)
    scale = np.asarray(candidate.scale_std_m, dtype=np.float64)
    quat = np.asarray(candidate.rotation_wxyz, dtype=np.float64)
    color = np.asarray(candidate.color_linear_rgb, dtype=np.float64)
    if mean.shape != (3,) or scale.shape != (3,) or quat.shape != (4,) or color.shape != (3,):
        return "invalid_parameter"
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(color).all():
        return "invalid_parameter"
    if np.any(scale <= 0) or np.any(scale > max_scale_m):
        return "degenerate_or_huge"
    if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-5:
        return "invalid_parameter"
    if not math.isfinite(candidate.opacity) or not 0.0 <= candidate.opacity <= 1.0:
        return "invalid_parameter"
    if candidate.support_count < 0:
        return "invalid_parameter"
    return None


def _cell_index(mean_world_m: np.ndarray, cell_m: float) -> tuple[int, int, int]:
    coords = np.floor(np.asarray(mean_world_m, dtype=np.float64) / cell_m).astype(np.int64)
    return int(coords[0]), int(coords[1]), int(coords[2])


def _neighbor_cells(index: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    x, y, z = index
    return tuple((x + dx, y + dy, z + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))


def _confirmed_same_surface(
    left: GaussianCandidate,
    right: GaussianCandidate,
    radius_m: float,
    orientation_dot_min: float,
    color_l2_max: float,
) -> bool:
    """距离只进入候选邻域。层、朝向、颜色和尺度范围都一致才抑制其中一个。"""
    if left.layer_id != right.layer_id:
        return False
    delta = np.asarray(left.mean_world_m, dtype=np.float64) - np.asarray(right.mean_world_m, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance > radius_m:
        return False
    left_scale = np.asarray(left.scale_std_m, dtype=np.float64)
    right_scale = np.asarray(right.scale_std_m, dtype=np.float64)
    if distance > float(np.min(left_scale) + np.min(right_scale)):
        return False
    left_q = np.asarray(left.rotation_wxyz, dtype=np.float64)
    right_q = np.asarray(right.rotation_wxyz, dtype=np.float64)
    if abs(float(np.dot(left_q, right_q))) < orientation_dot_min:
        return False
    color_delta = np.asarray(left.color_linear_rgb, dtype=np.float64) - np.asarray(right.color_linear_rgb, dtype=np.float64)
    return float(np.linalg.norm(color_delta)) <= color_l2_max
