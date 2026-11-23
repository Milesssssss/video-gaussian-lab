"""SHARP 与 Depth Pro 2 的只读接口。

分两层：

- `Blocked*`：正式协议入口。只检查帧划分与许可/权重/硬件阻塞，不执行推理。
- `SharpAdapter` / `DepthPro2Adapter`：执行层。后端由调用方注入（真实 torch 桥或测试替身），
  前置条件门、source-only 检查、输出转换与缓存键在这里统一执行。权重、许可与加速器
  任一缺失时同样拒绝，不会把未发生的推理记为已运行。
"""

from __future__ import annotations

import inspect
import shutil
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vglab.depth.aggregate import DepthPrediction, scalar_focal_px
from vglab.errors import ContractError, LeakageError, ModelPreconditionError
from vglab.initialization.sharp_scene import SHARP_GAUSSIAN_GRID, SHARP_LAYER_COUNT, CameraGaussian
from vglab.provenance import canonical_json_sha256


@dataclass(frozen=True)
class ModelBlock:
    backend: str
    reasons: tuple[str, ...]


def detect_accelerators() -> tuple[str, ...]:
    """诚实报告可用加速器：nvidia-smi、torch CUDA、torch MPS。CPU 总是存在。"""
    names: list[str] = []
    if shutil.which("nvidia-smi") is not None:
        names.append("cuda")
    try:
        import torch  # noqa: F401
    except Exception:
        torch = None
    if torch is not None:
        try:
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available() and "cuda" not in names:
                names.append("cuda")
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                names.append("mps")
        except Exception:
            pass
    names.append("cpu")
    return tuple(names)


def gpu_visible() -> bool:
    return any(name in {"cuda", "mps"} for name in detect_accelerators())


def environment_blockers(protocol: dict, *, gpu: bool | None = None) -> list[str]:
    from vglab.protocol import official_execution_blockers

    blockers = list(official_execution_blockers(protocol))
    if not (gpu_visible() if gpu is None else gpu):
        blockers.append("未检测到 nvidia-smi 或 MPS 加速器，本环境不能执行 SHARP 或 DP2 推理")
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
    """Phase 1 SHARP 正式入口。签名不接收 depth。"""

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


@dataclass(frozen=True)
class ModelPreconditions:
    backend: str
    weights_sha256: str | None
    model_use_conditions_resolved: bool
    accelerators: tuple[str, ...]


def resolve_model_preconditions(
    protocol: dict,
    backend: str,
    *,
    weights_sha256: str | None = None,
    model_use_conditions_resolved: bool | None = None,
    accelerators: tuple[str, ...] | None = None,
) -> ModelPreconditions:
    source_key = {"sharp": "sharp", "depth": "depth", "depth_pro_2": "depth"}.get(backend)
    if source_key is None:
        raise ContractError(f"未知后端 {backend}")
    recorded = ((protocol.get("candidate_sources") or {}).get(source_key) or {}).get("weight_sha256")
    digest = weights_sha256 if weights_sha256 is not None else recorded
    if model_use_conditions_resolved is None:
        resolved = (protocol.get("entry_conditions") or {}).get("model_use_conditions_resolved") is True
    else:
        resolved = model_use_conditions_resolved
    chosen = detect_accelerators() if accelerators is None else tuple(accelerators)
    return ModelPreconditions(
        backend=backend,
        weights_sha256=str(digest) if digest is not None else None,
        model_use_conditions_resolved=bool(resolved),
        accelerators=chosen,
    )


def precondition_blockers(preconditions: ModelPreconditions) -> tuple[str, ...]:
    reasons: list[str] = []
    if not isinstance(preconditions.weights_sha256, str) or len(preconditions.weights_sha256) != 64:
        reasons.append(f"{preconditions.backend} weight_sha256 未记录；不能把未发生的推理记为已运行")
    if not preconditions.model_use_conditions_resolved:
        reasons.append(f"{preconditions.backend} 没有覆盖本项目用途的 LICENSE_MODEL 授权记录，见 docs/THIRD_PARTY.md")
    if not any(item in {"cuda", "mps"} for item in preconditions.accelerators):
        reasons.append(f"{preconditions.backend} 没有可用加速器（CUDA/MPS），不能把未发生的推理记为已运行")
    return tuple(reasons)


@dataclass(frozen=True)
class SharpLayerOutput:
    mean_camera_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: np.ndarray
    color_linear_rgb: np.ndarray
    grid_uv: np.ndarray
    grid_shape: tuple[int, int]


@dataclass(frozen=True)
class SharpImageOutput:
    """后端契约：相机坐标物理高斯，层顺序即 SHARP 的层 0/1。"""

    layers: tuple[SharpLayerOutput, ...]
    internal_depth_z_m: tuple[np.ndarray, ...] = ()


class SharpImagePredictor(Protocol):
    def predict_image(self, image_srgb: np.ndarray, focal_px: float, config: dict) -> SharpImageOutput: ...


class DepthWindowPredictor(Protocol):
    def predict_window(self, images_srgb: list[np.ndarray], focal_px: float, config: dict) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class SharpFrameResult:
    frame_id: str
    gaussians: tuple[CameraGaussian, ...]
    focal_px: float
    layer_count: int
    internal_depth_shapes: tuple[tuple[int, ...], ...]


def camera_gaussians_from_sharp_output(output: SharpImageOutput, *, frame_id: str) -> list[CameraGaussian]:
    layers = tuple(output.layers)
    if len(layers) != SHARP_LAYER_COUNT:
        raise ContractError(f"SHARP 输出必须恰好 {SHARP_LAYER_COUNT} 层，实际 {len(layers)}")
    gaussians: list[CameraGaussian] = []
    grid_shape: tuple[int, int] | None = None
    for layer_id, layer in enumerate(layers):
        means = np.asarray(layer.mean_camera_m, dtype=np.float64)
        scales = np.asarray(layer.scale_std_m, dtype=np.float64)
        rotations = np.asarray(layer.rotation_wxyz, dtype=np.float64)
        opacity = np.asarray(layer.opacity, dtype=np.float64).reshape(-1)
        colors = np.asarray(layer.color_linear_rgb, dtype=np.float64)
        grid_uv = np.asarray(layer.grid_uv)
        count = len(means)
        if means.shape != (count, 3) or scales.shape != (count, 3) or rotations.shape != (count, 4):
            raise ContractError("SHARP 层输出形状非法")
        if opacity.shape != (count,) or colors.shape != (count, 3) or grid_uv.shape != (count, 2):
            raise ContractError("SHARP 层的 opacity/颜色/grid 数量必须一致")
        if not np.isfinite(means).all() or not np.isfinite(scales).all() or not np.isfinite(colors).all():
            raise ContractError("SHARP 输出包含非有限参数")
        if np.any(scales <= 0):
            raise ContractError("SHARP 输出必须给正尺度")
        if np.any((opacity <= 0) | (opacity > 1)):
            raise ContractError("SHARP 输出 opacity 必须在 (0,1]")
        norms = np.linalg.norm(rotations, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise ContractError("SHARP 输出四元数必须归一化；后端转换必须显式处理")
        if np.any((colors < 0) | (colors > 1)):
            raise ContractError("SHARP 输出线性 RGB 必须在 [0,1]")
        shape = layer.grid_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
        ):
            raise ContractError("grid_shape 必须是两个正整数")
        if grid_shape is None:
            grid_shape = shape
        elif grid_shape != shape:
            raise ContractError("SHARP 两层必须共享同一内部 grid")
        integer_uv = np.rint(grid_uv).astype(np.int64)
        if not np.array_equal(integer_uv, grid_uv.astype(np.int64)):
            raise ContractError("grid_uv 必须是整数像素坐标")
        if np.any(integer_uv < 0) or np.any(integer_uv[:, 0] >= shape[1]) or np.any(integer_uv[:, 1] >= shape[0]):
            raise ContractError("grid_uv 越出声明的 grid")
        for original_index in range(count):
            u = int(integer_uv[original_index, 0])
            v = int(integer_uv[original_index, 1])
            gaussians.append(
                CameraGaussian(
                    gaussian_id=f"{frame_id}:L{layer_id}:{original_index:06d}",
                    frame_id=frame_id,
                    layer_id=layer_id,
                    mean_camera_m=means[original_index],
                    scale_std_m=scales[original_index],
                    rotation_wxyz=rotations[original_index],
                    opacity=float(opacity[original_index]),
                    color_linear_rgb=colors[original_index],
                    source_grid_uv=(u, v),
                    source_grid_shape=shape,
                    original_index=original_index,
                )
            )
    return gaussians


class SharpAdapter:
    """执行层 SHARP 入口。后端注入；前置条件不满足时在执行前拒绝。"""

    def __init__(
        self,
        backend: SharpImagePredictor,
        preconditions: ModelPreconditions,
        *,
        config: dict | None = None,
    ):
        if preconditions.backend != "sharp":
            raise ContractError("SharpAdapter 需要 backend=sharp 的前置条件")
        self.backend = backend
        self.preconditions = preconditions
        self.block = ModelBlock("sharp", precondition_blockers(preconditions))
        self.config = dict(config or {})

    def predict_source_frame(
        self,
        frame: dict,
        image_srgb: np.ndarray,
        k_matrix: np.ndarray,
    ) -> SharpFrameResult:
        _require_source_frame(frame, "SHARP")
        if self.block.reasons:
            raise ModelPreconditionError("；".join(self.block.reasons))
        image = np.asarray(image_srgb, dtype=np.float64)
        _validate_image(image, frame, "SHARP")
        focal_px = scalar_focal_px(k_matrix)
        output = self.backend.predict_image(image, focal_px, dict(self.config))
        gaussians = camera_gaussians_from_sharp_output(output, frame_id=str(frame["frame_id"]))
        shapes = tuple(np.asarray(item).shape for item in output.internal_depth_z_m)
        return SharpFrameResult(
            frame_id=str(frame["frame_id"]),
            gaussians=tuple(gaussians),
            focal_px=focal_px,
            layer_count=len(output.layers),
            internal_depth_shapes=shapes,
        )

    def cache_key(
        self,
        frame: dict,
        *,
        image_sha256: str,
        transform_sha256: str,
        config_sha256: str,
    ) -> str:
        if self.preconditions.weights_sha256 is None:
            raise ModelPreconditionError("SHARP 权重哈希未记录，不能生成缓存键")
        return sharp_cache_key(
            weights_sha256=self.preconditions.weights_sha256,
            frame_id=str(frame["frame_id"]),
            image_sha256=image_sha256,
            k_matrix=np.asarray(frame["K"], dtype=np.float64),
            transform_sha256=transform_sha256,
            config_sha256=config_sha256,
        )


class DepthPro2Adapter:
    """执行层 DP2 入口。一个窗口一个显式焦距；窗口内焦距不一致时拒绝。"""

    def __init__(
        self,
        backend: DepthWindowPredictor,
        preconditions: ModelPreconditions,
        *,
        config: dict | None = None,
    ):
        if preconditions.backend not in {"depth", "depth_pro_2"}:
            raise ContractError("DepthPro2Adapter 需要 backend=depth 的前置条件")
        self.backend = backend
        self.preconditions = preconditions
        self.block = ModelBlock("depth_pro_2", precondition_blockers(preconditions))
        self.config = dict(config or {})

    def predict_window(
        self,
        frames: list[dict],
        images_by_frame: dict[str, np.ndarray],
        k_by_frame: dict[str, np.ndarray],
        pose_by_frame: dict[str, np.ndarray] | None = None,
    ) -> dict[str, DepthPrediction]:
        if not frames:
            raise ContractError("DP2 窗口不能为空")
        frame_ids = [str(frame["frame_id"]) for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ContractError("DP2 窗口不能包含重复帧")
        if self.block.reasons:
            raise ModelPreconditionError("；".join(self.block.reasons))
        images: list[np.ndarray] = []
        focals: list[float] = []
        for frame in frames:
            _require_source_frame(frame, "DP2")
            frame_id = str(frame["frame_id"])
            if frame_id not in images_by_frame or frame_id not in k_by_frame:
                raise ContractError(f"{frame_id} 缺少图像或 K")
            image = np.asarray(images_by_frame[frame_id], dtype=np.float64)
            _validate_image(image, frame, "DP2")
            images.append(image)
            focals.append(scalar_focal_px(k_by_frame[frame_id]))
        focal_px = focals[0]
        if any(abs(value - focal_px) > 1e-6 for value in focals[1:]):
            raise ContractError("同一 DP2 窗口的焦距必须一致；不兼容焦距需要拆成不同窗口")
        depths = self.backend.predict_window(images, focal_px, dict(self.config))
        if len(depths) != len(frames):
            raise ContractError("DP2 后端必须为窗口内每一帧返回一份深度")
        predictions: dict[str, DepthPrediction] = {}
        for position, (frame, frame_id, depth) in enumerate(zip(frames, frame_ids, depths, strict=True)):
            value = np.asarray(depth, dtype=np.float64)
            if value.shape != images[position].shape[:2]:
                raise ContractError(f"{frame_id} 的 DP2 深度网格与输入图像不一致")
            if np.any(np.isfinite(value) & (value < 0)) or np.any(np.isposinf(value)):
                raise ContractError("DP2 深度只能是非负有限值或 NaN；负值与 +inf 不进入管线")
            pose = None
            if pose_by_frame is not None and frame_id in pose_by_frame:
                pose = pose_by_frame[frame_id]
            predictions[frame_id] = DepthPrediction(
                frame_id=frame_id,
                depth_m=value,
                context_frame_ids=tuple(frame_ids),
                representation="z_depth",
                unit="m",
                k_matrix=np.array(k_by_frame[frame_id], dtype=np.float64, copy=True),
                t_world_camera=None if pose is None else np.array(pose, dtype=np.float64, copy=True),
            )
        return predictions

    def cache_key(
        self,
        *,
        context_frame_ids: list[str] | tuple[str, ...],
        image_sha256_by_frame: dict[str, str],
        focal_px: float,
        config_sha256: str,
        manifest: dict,
    ) -> str:
        if self.preconditions.weights_sha256 is None:
            raise ModelPreconditionError("DP2 权重哈希未记录，不能生成缓存键")
        return dp2_cache_key(
            weights_sha256=self.preconditions.weights_sha256,
            context_frame_ids=context_frame_ids,
            image_sha256_by_frame=image_sha256_by_frame,
            focal_px=focal_px,
            config_sha256=config_sha256,
            manifest=manifest,
        )


def _validate_image(image: np.ndarray, frame: dict, backend: str) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContractError(f"{backend} 输入必须是 H×W×3 的 sRGB 图像")
    if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ContractError(f"{backend} 输入 sRGB 必须在 [0,1]")
    height = frame.get("height_px")
    width = frame.get("width_px")
    if isinstance(height, int) and isinstance(width, int):
        if image.shape[0] != height or image.shape[1] != width:
            raise ContractError(f"{backend} 图像网格与 manifest 声明不一致")


def assert_sharp_api_has_no_depth_argument() -> None:
    for target in (BlockedSharpAdapter.predict_source_frame, SharpAdapter.predict_source_frame):
        parameters = inspect.signature(target).parameters
        if "depth" in parameters or "depth_pro" in parameters:
            raise ModelPreconditionError("SHARP Phase 1 接口不能接收外部 depth")
    parameters = inspect.signature(camera_gaussians_from_sharp_output).parameters
    if "depth" in parameters or "depth_pro" in parameters:
        raise ModelPreconditionError("SHARP 输出转换不能接收外部 depth")


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
