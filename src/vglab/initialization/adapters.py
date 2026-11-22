"""SHARP 与 Depth Pro 2 的只读接口。

当前实现只检查帧划分并报告许可/权重阻塞，不下载权重，也不把阻塞记成一次推理。
"""

from __future__ import annotations

import inspect
import shutil
from dataclasses import dataclass

import numpy as np

from vglab.errors import LeakageError, ModelPreconditionError
from vglab.provenance import canonical_json_sha256


@dataclass(frozen=True)
class ModelBlock:
    backend: str
    reasons: tuple[str, ...]


def gpu_visible() -> bool:
    return shutil.which("nvidia-smi") is not None


def environment_blockers(protocol: dict, *, gpu: bool | None = None) -> list[str]:
    from vglab.protocol import official_execution_blockers

    blockers = list(official_execution_blockers(protocol))
    if not (gpu_visible() if gpu is None else gpu):
        blockers.append("未检测到 nvidia-smi，本环境不能执行 SHARP 或 DP2 推理")
    blockers.append("examples/scene_manifest.json 是 format_example，不能当作真实采集")
    blockers.append("P1b 的 60 次锁定测试不在本次执行范围内")
    return blockers


def sharp_precondition_block(protocol: dict, *, gpu: bool | None = None) -> ModelBlock:
    reasons = ["SHARP 推理未执行"]
    digest = ((protocol.get("candidate_sources") or {}).get("sharp") or {}).get("weight_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        reasons.append("SHARP weight_sha256 未记录，见 docs/THIRD_PARTY.md")
    if (protocol.get("entry_conditions") or {}).get("model_use_conditions_resolved") is not True:
        reasons.append("没有覆盖本项目用途的 SHARP LICENSE_MODEL 授权记录，见 docs/THIRD_PARTY.md")
    if ((protocol.get("candidate_sources") or {}).get("sharp") or {}).get("use_optional_depth_hook") is not False:
        reasons.append("use_optional_depth_hook 必须保持 false")
    if not (gpu_visible() if gpu is None else gpu):
        reasons.append("未检测到可用 GPU，不能把未发生的 SHARP 推理记为已运行")
    return ModelBlock("sharp", tuple(reasons))


def dp2_precondition_block(protocol: dict, *, gpu: bool | None = None) -> ModelBlock:
    reasons = ["Depth Pro 2 推理未执行"]
    digest = ((protocol.get("candidate_sources") or {}).get("depth") or {}).get("weight_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        reasons.append("DP2 weight_sha256 未记录，见 docs/THIRD_PARTY.md")
    if (protocol.get("entry_conditions") or {}).get("model_use_conditions_resolved") is not True:
        reasons.append("没有覆盖本项目用途的 Depth Pro 2 LICENSE_MODEL 授权记录，见 docs/THIRD_PARTY.md")
    if not (gpu_visible() if gpu is None else gpu):
        reasons.append("未检测到可用 GPU，不能把未发生的 DP2 推理记为已运行")
    return ModelBlock("depth_pro_2", tuple(reasons))


def _require_source_frame(frame: dict, backend: str) -> None:
    if frame.get("split") != "source":
        frame_id = frame.get("frame_id")
        raise LeakageError(f"{frame_id} split={frame.get('split')} 不能进入 {backend}")


class BlockedSharpAdapter:
    """Phase 1 SHARP 入口。签名不接收 depth。"""

    def __init__(self, protocol: dict, *, gpu: bool | None = None):
        self.block = sharp_precondition_block(protocol, gpu=gpu)

    def predict_source_frame(self, frame: dict, k_matrix: np.ndarray) -> None:
        _require_source_frame(frame, "SHARP")
        raise ModelPreconditionError("；".join(self.block.reasons))


class BlockedDepthAdapter:
    def __init__(self, protocol: dict, *, gpu: bool | None = None):
        self.block = dp2_precondition_block(protocol, gpu=gpu)

    def predict_window(self, frames: list[dict], focal_px: float) -> None:
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise ModelPreconditionError("DP2 焦距必须是显式正值；该检查先于权重阻塞之外的推理")
        for frame in frames:
            _require_source_frame(frame, "DP2")
        raise ModelPreconditionError("；".join(self.block.reasons))


def assert_sharp_api_has_no_depth_argument() -> None:
    parameters = inspect.signature(BlockedSharpAdapter.predict_source_frame).parameters
    if "depth" in parameters or "depth_pro" in parameters:
        raise ModelPreconditionError("SHARP Phase 1 接口不能接收外部 depth")


def sharp_cache_key(
    *,
    weights_sha256: str,
    frame_id: str,
    image_sha256: str,
    k_matrix: np.ndarray,
    transform_sha256: str,
    config_sha256: str,
    **extra: object,
) -> str:
    if extra:
        raise LeakageError("SHARP 缓存键不能包含 DP2 或其他额外深度字段")
    return canonical_json_sha256(
        {
            "backend": "sharp",
            "weights_sha256": weights_sha256,
            "frame_id": frame_id,
            "image_sha256": image_sha256,
            "K": np.asarray(k_matrix, dtype=np.float64).round(10).tolist(),
            "transform_sha256": transform_sha256,
            "config_sha256": config_sha256,
        }
    )


def dp2_cache_key(
    *,
    weights_sha256: str,
    context_frame_ids: list[str] | tuple[str, ...],
    image_sha256_by_frame: dict[str, str],
    focal_px: float,
    config_sha256: str,
    manifest: dict,
) -> str:
    from vglab.ingest.split import assert_context_is_source_only

    assert_context_is_source_only(context_frame_ids, manifest)
    return canonical_json_sha256(
        {
            "backend": "depth_pro_2",
            "weights_sha256": weights_sha256,
            "context_frame_ids": list(context_frame_ids),
            "image_sha256_by_frame": {frame_id: image_sha256_by_frame[frame_id] for frame_id in context_frame_ids},
            "focal_px": focal_px,
            "config_sha256": config_sha256,
        }
    )
