"""由 DP2 预测构造固定的深度教师信号：teacher 掩码与启发式可靠性 q。

这里的掩码与权重只由 source 数据、固定 K/T 和其它 source 深度计算，并在优化前固定。
q 是明确声明的启发式值，不是 DP2 官方置信度，也不是已校准概率。数值来自
`candidates.UNFROZEN_DEPTH_TEACHER`，未在开发集冻结。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.calibration.camera import as_intrinsics, as_pose, invert_world_camera
from vglab.depth.aggregate import DepthPrediction, assert_prediction_context
from vglab.errors import ContractError, LeakageError
from vglab.ingest.split import frames_by_id
from vglab.initialization.candidates import UNFROZEN_DEPTH_TEACHER


@dataclass(frozen=True)
class DepthView:
    """另一个 source 视角的深度与固定相机，用于自由空间冲突与支持检查。"""

    frame_id: str
    depth_m: np.ndarray
    k_matrix: np.ndarray
    t_world_camera: np.ndarray


@dataclass(frozen=True)
class DepthTeacher:
    frame_id: str
    depth_m: np.ndarray
    mask: np.ndarray
    reliability: np.ndarray
    diagnostics: dict[str, int | float]
    config_name: str
    frozen: bool = False


def build_depth_teacher(
    prediction: DepthPrediction,
    manifest: dict,
    *,
    static_mask: np.ndarray | None = None,
    views: tuple[DepthView, ...] | list[DepthView] = (),
    temporal_depth_m: np.ndarray | None = None,
    config: dict | None = None,
) -> DepthTeacher:
    """构造 teacher。失败与排除都显式记录，不做静默修补。"""
    assert_prediction_context(prediction, manifest)
    chosen = UNFROZEN_DEPTH_TEACHER if config is None else config
    if chosen.get("frozen") is not False:
        raise ContractError("深度教师配置必须显式 frozen=false")
    depth = np.asarray(prediction.depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ContractError("teacher 深度必须是 H×W")
    height, width = depth.shape
    if static_mask is None:
        static = np.ones((height, width), dtype=bool)
    else:
        static = np.asarray(static_mask, dtype=bool)
        if static.shape != depth.shape:
            raise ContractError("static_mask 必须与 teacher 深度同网格")
    finite = np.isfinite(depth) & (depth > 0)
    source_valid = finite & static

    q_edge = _edge_reliability(depth, source_valid, float(chosen["edge_k"]))
    q_range = _range_reliability(
        depth,
        float(chosen["range_near_m"]),
        float(chosen["range_far_m"]),
        float(chosen["range_rolloff_m"]),
        float(chosen["range_floor"]),
    )
    q_temporal = _temporal_reliability(depth, temporal_depth_m, float(chosen["temporal_tau"]), source_valid)

    visibility = chosen["visibility"]
    tau_abs = float(visibility["tau_abs_m"])
    tau_rel = float(visibility["tau_rel"])
    supported_count, conflict_count, occluded_count, unknown_count, outside_count = _support_totals(
        depth,
        prediction,
        manifest,
        views,
        tau_abs,
        tau_rel,
    )
    support_min = int(chosen["support_min_views"])
    if support_min < 0:
        raise ContractError("support_min_views 不能为负")
    supported = supported_count >= support_min if views else np.zeros_like(source_valid)
    q_support = np.where(supported, 1.0, float(chosen["q_unsupported"]))
    if not views:
        q_support = np.full_like(depth, float(chosen["q_unsupported"]))

    weights = chosen["weights"]
    q = (
        float(weights["edge"]) * q_edge
        + float(weights["range"]) * q_range
        + float(weights["support"]) * q_support
        + float(weights["temporal"]) * q_temporal
    )
    mask = source_valid.copy()
    if chosen.get("conflict_excludes") is True:
        mask &= conflict_count == 0
    q = np.where(mask, np.clip(q, float(chosen["q_floor"]), 1.0), 0.0)

    diagnostics: dict[str, int | float] = {
        "pixels": int(mask.size),
        "teacher_valid_px": int(mask.sum()),
        "static_excluded_px": int((finite & ~static).sum()),
        "depth_invalid_px": int((~finite).sum()),
        "free_space_conflict_px": int((source_valid & (conflict_count > 0)).sum()),
        "supported_px": int((mask & supported).sum()),
        "occluded_px": int((mask & (occluded_count > 0) & (conflict_count == 0)).sum()),
        "unknown_px": int((mask & (unknown_count > 0) & ~supported).sum()),
        "outside_px": int((mask & (outside_count > 0) & ~supported).sum()),
        "mean_reliability": float(q[mask].mean()) if np.any(mask) else 0.0,
        "views_checked": int(len(views)),
    }
    return DepthTeacher(
        frame_id=prediction.frame_id,
        depth_m=depth,
        mask=mask,
        reliability=q,
        diagnostics=diagnostics,
        config_name=str(chosen["name"]),
    )


def _edge_reliability(depth: np.ndarray, valid: np.ndarray, k: float) -> np.ndarray:
    height, width = depth.shape
    if height < 2 or width < 2:
        return np.ones_like(depth)
    safe = np.where(valid, depth, 1.0)
    log_depth = np.log(safe)
    gy, gx = np.gradient(log_depth)
    gradient = np.sqrt(gx * gx + gy * gy)
    return np.exp(-k * gradient)


def _range_reliability(
    depth: np.ndarray,
    near: float,
    far: float,
    rolloff: float,
    floor: float,
) -> np.ndarray:
    if not (0 < near < far) or rolloff <= 0 or not 0 <= floor <= 1:
        raise ContractError("深度范围可靠性参数非法")
    lower = (depth - near + rolloff) / rolloff
    upper = (far + rolloff - depth) / rolloff
    return np.clip(np.minimum(lower, upper), floor, 1.0)


def _temporal_reliability(
    depth: np.ndarray,
    temporal_depth_m: np.ndarray | None,
    tau: float,
    valid: np.ndarray,
) -> np.ndarray:
    if temporal_depth_m is None:
        return np.ones_like(depth)
    previous = np.asarray(temporal_depth_m, dtype=np.float64)
    if previous.shape != depth.shape:
        raise ContractError("temporal_depth_m 必须与 teacher 深度同网格")
    if tau <= 0:
        raise ContractError("temporal_tau 必须为正")
    previous_valid = np.isfinite(previous) & (previous > 0)
    safe_delta = np.log(np.where(valid, depth, 1.0)) - np.log(np.where(previous_valid, previous, 1.0))
    return np.where(previous_valid & valid, np.exp(-np.abs(safe_delta) / tau), 1.0)


def _support_totals(
    depth: np.ndarray,
    prediction: DepthPrediction,
    manifest: dict,
    views: tuple[DepthView, ...] | list[DepthView],
    tau_abs: float,
    tau_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    supported = np.zeros(depth.shape, dtype=np.int64)
    conflict = np.zeros(depth.shape, dtype=np.int64)
    occluded = np.zeros(depth.shape, dtype=np.int64)
    unknown = np.zeros(depth.shape, dtype=np.int64)
    outside = np.zeros(depth.shape, dtype=np.int64)
    if not views:
        return supported, conflict, occluded, unknown, outside
    index = frames_by_id(manifest)
    for view in views:
        frame = index.get(view.frame_id)
        if frame is None or frame.get("split") != "source":
            raise LeakageError(f"{view.frame_id} 不是 manifest 中的 source 帧")
        if view.frame_id == prediction.frame_id:
            raise ContractError("支持检查不能使用被预测帧自身冒充其他视角")
        part_supported, part_conflict, part_occluded, part_unknown, part_outside = _support_counts(
            depth,
            prediction,
            view,
            tau_abs,
            tau_rel,
        )
        supported += part_supported
        conflict += part_conflict
        occluded += part_occluded
        unknown += part_unknown
        outside += part_outside
    return supported, conflict, occluded, unknown, outside


def _support_counts(
    depth: np.ndarray,
    prediction: DepthPrediction,
    view: DepthView,
    tau_abs: float,
    tau_rel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把本帧像素投影到另一 source 视角，按 `classify_visibility` 的同一阈值计数。"""
    own_intrinsics = _prediction_intrinsics(prediction)
    own_pose = _prediction_pose(prediction)
    other_depth = np.asarray(view.depth_m, dtype=np.float64)
    if other_depth.ndim != 2:
        raise ContractError("支持视角深度必须是 H×W")
    other_intrinsics = as_intrinsics(view.k_matrix)
    other_pose = as_pose(view.t_world_camera)
    height, width = depth.shape
    other_height, other_width = other_depth.shape
    vv, uu = np.mgrid[0:height, 0:width]
    x = (uu - own_intrinsics[0, 2]) / own_intrinsics[0, 0] * depth
    y = (vv - own_intrinsics[1, 2]) / own_intrinsics[1, 1] * depth
    camera = np.stack([x, y, depth], axis=-1)
    world = camera @ own_pose[:3, :3].T + own_pose[:3, 3]
    other_from_world = invert_world_camera(other_pose)
    camera_other = world @ other_from_world[:3, :3].T + other_from_world[:3, 3]
    z_other = camera_other[..., 2]
    fx, fy = float(other_intrinsics[0, 0]), float(other_intrinsics[1, 1])
    cx, cy = float(other_intrinsics[0, 2]), float(other_intrinsics[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        u_other = fx * camera_other[..., 0] / z_other + cx
        v_other = fy * camera_other[..., 1] / z_other + cy
    in_bounds = (
        np.isfinite(u_other)
        & np.isfinite(v_other)
        & np.isfinite(z_other)
        & (z_other > 0)
        & (u_other >= 0)
        & (u_other <= other_width - 1)
        & (v_other >= 0)
        & (v_other <= other_height - 1)
    )
    ui = np.clip(np.rint(np.where(in_bounds, u_other, 0)).astype(np.int64), 0, other_width - 1)
    vi = np.clip(np.rint(np.where(in_bounds, v_other, 0)).astype(np.int64), 0, other_height - 1)
    reference = other_depth[vi, ui]
    reference_valid = np.isfinite(reference) & (reference > 0)
    tolerance = tau_abs + tau_rel * np.where(reference_valid, reference, 0.0)
    supported = in_bounds & reference_valid & (np.abs(z_other - reference) <= tolerance)
    conflict = in_bounds & reference_valid & (z_other < reference - tolerance)
    occluded = in_bounds & reference_valid & (z_other > reference + tolerance)
    unknown = in_bounds & ~reference_valid
    outside = ~in_bounds
    return (
        supported.astype(np.int64),
        conflict.astype(np.int64),
        occluded.astype(np.int64),
        unknown.astype(np.int64),
        outside.astype(np.int64),
    )


def _prediction_intrinsics(prediction: DepthPrediction) -> np.ndarray:
    if prediction.k_matrix is None:
        raise ContractError("支持检查需要被预测帧的 k_matrix 附件；请用 attach_camera 构造预测")
    return as_intrinsics(prediction.k_matrix)


def _prediction_pose(prediction: DepthPrediction) -> np.ndarray:
    if prediction.t_world_camera is None:
        raise ContractError("支持检查需要被预测帧的 t_world_camera 附件；请用 attach_camera 构造预测")
    return as_pose(prediction.t_world_camera)


def attach_camera(prediction: DepthPrediction, k_matrix: np.ndarray, t_world_camera: np.ndarray) -> DepthPrediction:
    """把固定相机附加到预测上，供多视角支持检查使用；不改动深度与上下文。"""
    as_intrinsics(k_matrix)
    as_pose(t_world_camera)
    return DepthPrediction(
        frame_id=prediction.frame_id,
        depth_m=prediction.depth_m,
        context_frame_ids=prediction.context_frame_ids,
        representation=prediction.representation,
        unit=prediction.unit,
        k_matrix=np.array(k_matrix, dtype=np.float64, copy=True),
        t_world_camera=np.array(t_world_camera, dtype=np.float64, copy=True),
    )
