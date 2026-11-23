"""Depth Pro 2 风格的 source 窗口与逐像素 log-median。

这些函数不调用模型。窗口参数来自协议登记，不是已测得的最优值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.calibration.camera import as_intrinsics
from vglab.errors import ContractError, LeakageError
from vglab.ingest.split import assert_context_is_source_only


@dataclass(frozen=True)
class DepthWindow:
    window_id: str
    context_frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class DepthPrediction:
    frame_id: str
    depth_m: np.ndarray
    context_frame_ids: tuple[str, ...]
    representation: str = "z_depth"
    unit: str = "m"
    k_matrix: np.ndarray | None = None
    t_world_camera: np.ndarray | None = None


def scalar_focal_px(k_matrix: np.ndarray) -> float:
    """DP2 公开接口只接收一个焦距。fx 与 fy 不一致时拒绝，不取平均。"""
    intrinsics = as_intrinsics(k_matrix)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    if abs(fx - fy) > 1e-6:
        raise ContractError("同一针孔 K 的 fx 与 fy 不一致，不能平均成 DP2 焦距；需要拆开序列或先矫正")
    return fx


def source_windows(
    frame_ids: list[str],
    manifest: dict,
    *,
    window_size: int = 16,
    overlap: int = 4,
) -> list[DepthWindow]:
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ContractError("窗口长度必须为正，且重叠必须小于窗口长度")
    assert_context_is_source_only(frame_ids, manifest)
    if len(frame_ids) != len(set(frame_ids)):
        raise ContractError("source 窗口输入不能重复帧来伪造多视图")
    if not frame_ids:
        return []
    stride = window_size - overlap
    windows: list[DepthWindow] = []
    start = 0
    while start < len(frame_ids):
        chunk = tuple(frame_ids[start : start + window_size])
        if windows and set(chunk) <= set(windows[-1].context_frame_ids):
            break
        windows.append(DepthWindow(window_id=f"w{len(windows):03d}:{chunk[0]}:{chunk[-1]}", context_frame_ids=chunk))
        if start + window_size >= len(frame_ids):
            break
        start += stride
    return windows


def assert_prediction_context(prediction: DepthPrediction, manifest: dict) -> None:
    if prediction.representation != "z_depth" or prediction.unit != "m":
        raise ContractError("深度预测必须声明 representation=z_depth 且 unit=m")
    try:
        assert_context_is_source_only(prediction.context_frame_ids, manifest)
    except LeakageError as exc:
        raise LeakageError(f"{prediction.frame_id} 的上下文泄漏: {exc}") from exc
    if prediction.frame_id not in prediction.context_frame_ids:
        raise ContractError("context_frame_ids 必须包含被预测的 source 帧")


def log_median_depth(depths: list[np.ndarray]) -> np.ndarray:
    """对同一像素的有效正深度做 log-median。没有有效值时保持 NaN。"""
    if not depths:
        raise ContractError("log-median 至少需要一份深度图")
    stack = np.stack([np.asarray(depth, dtype=np.float64) for depth in depths], axis=0)
    if stack.ndim != 3:
        raise ContractError("深度图必须是 H×W")
    valid = np.isfinite(stack) & (stack > 0)
    logs = np.log(np.where(valid, stack, 1.0))
    masked = np.ma.array(logs, mask=~valid)
    merged_log = np.ma.median(masked, axis=0).filled(np.nan)
    return np.exp(merged_log)


def aggregate_frame_predictions(predictions: list[DepthPrediction], manifest: dict, frame_id: str) -> np.ndarray:
    selected = [item for item in predictions if item.frame_id == frame_id]
    if not selected:
        raise ContractError(f"{frame_id} 没有可合并的深度预测")
    for item in selected:
        assert_prediction_context(item, manifest)
    return log_median_depth([item.depth_m for item in selected])
