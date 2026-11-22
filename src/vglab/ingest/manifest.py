"""scene manifest 的 schema 与执行资格。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vglab.ids import split_sha256

ALLOWED_KINDS = ("format_example", "synthetic_geometry_fixture", "captured_scene")
FRAME_SPLITS = ("source", "validation", "test")
DEPTH_SOURCES = ("sensor", "predicted", "reference", "none")
POSE_STATUSES = ("measured_or_estimated_fixed", "reference")
_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class SchemaReport:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scene manifest 必须是 JSON object")
    return payload


def validate_schema(manifest: dict) -> SchemaReport:
    errors: list[str] = []
    _expect(errors, manifest.get("schema_version") == "0.1", "schema_version 必须是 0.1")
    _expect(errors, manifest.get("kind") in ALLOWED_KINDS, "kind 不在允许集合内")
    _expect(errors, isinstance(manifest.get("scene_id"), str) and manifest.get("scene_id"), "scene_id 不能为空")
    _expect(errors, manifest.get("scene_split") in {"dev", "test"}, "scene_split 只能是 dev 或 test")
    units = manifest.get("units") or {}
    _expect(errors, units.get("distance") == "m", "units.distance 必须是 m")
    _expect(errors, units.get("time") == "s", "units.time 必须是 s")
    _expect(errors, manifest.get("camera_axes") == "opencv_optical", "camera_axes 必须是 opencv_optical")
    _expect(errors, manifest.get("transform") == "T_world_camera", "transform 必须是 T_world_camera")
    _expect(errors, manifest.get("matrix_storage") == "row_major", "matrix_storage 必须是 row_major")
    _expect(errors, manifest.get("vector_convention") == "column_vector", "vector_convention 必须是 column_vector")
    _expect(errors, manifest.get("pixel_coordinates") == "integer_center", "pixel_coordinates 必须是 integer_center")
    _expect(errors, manifest.get("depth_representation") == "z_depth", "depth_representation 必须是 z_depth")
    preprocess = manifest.get("preprocess") or {}
    _expect(errors, preprocess.get("color_space") == "srgb", "preprocess.color_space 必须是 srgb")
    _expect(errors, preprocess.get("K_is_for") == "canonical_image", "K_is_for 必须声明 canonical_image")
    policy = manifest.get("split_policy") or {}
    _expect(errors, policy.get("eval_frames_in_network_context") is False, "eval_frames_in_network_context 必须为 false")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("frames 必须是非空数组")
        return SchemaReport(tuple(errors))
    source_ids = list(policy.get("source_frame_ids") or [])
    validation_ids = list(policy.get("validation_frame_ids") or [])
    test_ids = list(policy.get("test_frame_ids") or [])
    _expect_disjoint(errors, source_ids, validation_ids, test_ids)
    seen: set[str] = set()
    by_split: dict[str, list[str]] = {name: [] for name in FRAME_SPLITS}
    previous_timestamp = -math.inf
    canonical_w = preprocess.get("canonical_width_px")
    canonical_h = preprocess.get("canonical_height_px")
    for frame in frames:
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in seen:
            errors.append(f"frame_id 缺失或重复: {frame_id!r}")
            continue
        seen.add(frame_id)
        split = frame.get("split")
        if split not in FRAME_SPLITS:
            errors.append(f"{frame_id} split 非法")
        else:
            by_split[split].append(frame_id)
        timestamp = frame.get("timestamp_s")
        if not _finite_number(timestamp) or timestamp <= previous_timestamp:
            errors.append(f"{frame_id} timestamp_s 必须按清单顺序严格递增")
        else:
            previous_timestamp = float(timestamp)
        _expect_matrix(errors, frame.get("K"), (3, 3), f"{frame_id} K")
        _expect_matrix(errors, frame.get("T_world_camera"), (4, 4), f"{frame_id} T_world_camera")
        if _is_matrix(frame.get("K"), (3, 3)):
            _validate_k(errors, np.asarray(frame["K"], dtype=np.float64), frame_id)
        if _is_matrix(frame.get("T_world_camera"), (4, 4)):
            _validate_pose(errors, np.asarray(frame["T_world_camera"], dtype=np.float64), frame_id)
        _validate_depth(errors, frame)
        _validate_frame_size(errors, frame, canonical_w, canonical_h)
        if not _finite_number(frame.get("pose_timestamp_s")):
            errors.append(f"{frame_id} pose_timestamp_s 必须是有限数值")
    if by_split["source"] != source_ids or by_split["validation"] != validation_ids or by_split["test"] != test_ids:
        errors.append("split_policy 的帧清单必须与 frames[].split 的出现顺序一致且不相交")
    return SchemaReport(tuple(errors))


def execution_blockers(manifest: dict, *, require_files: bool = False) -> list[str]:
    blockers: list[str] = []
    schema = validate_schema(manifest)
    blockers.extend(schema.errors)
    kind = manifest.get("kind")
    if kind == "format_example":
        blockers.append("kind=format_example 不可作为数据集或可执行实验，不能进入执行器")
    if kind != "captured_scene":
        blockers.append("kind 必须是 captured_scene 才能进入模型执行")
    if manifest.get("pose_status") not in POSE_STATUSES:
        blockers.append("pose_status 必须是 measured_or_estimated_fixed 或有独立证据的 reference")
    provider = manifest.get("pose_provider") or {}
    if provider.get("used_eval_rgb") not in (True, False, "unknown"):
        blockers.append("pose_provider.used_eval_rgb 必须显式为 true、false 或 unknown")
    if manifest.get("pose_status") == "reference" and provider.get("reference_geometry_available") is not True:
        blockers.append("pose_status=reference 需要 reference_geometry_available=true")
    calibration = manifest.get("calibration_sha256")
    if not _is_sha256(calibration):
        blockers.append("calibration_sha256 缺失")
    policy = manifest.get("split_policy") or {}
    if policy.get("frozen_before_inference") is not True:
        blockers.append("split_policy.frozen_before_inference 必须为 true")
    hashes: dict[str, str | None] = {}
    for frame in manifest.get("frames") or []:
        frame_id = frame.get("frame_id")
        digest = frame.get("rgb_sha256")
        hashes[frame_id] = digest
        if not _is_sha256(digest):
            blockers.append(f"{frame_id} rgb_sha256 缺失")
        if require_files:
            rgb_path = frame.get("rgb_path")
            if not isinstance(rgb_path, str) or not Path(rgb_path).is_file():
                blockers.append(f"{frame_id} rgb_path 不存在")
    if schema.ok:
        expected = split_sha256(
            manifest["scene_id"],
            list(policy.get("source_frame_ids") or []),
            list(policy.get("validation_frame_ids") or []),
            list(policy.get("test_frame_ids") or []),
            hashes,
        )
        if manifest.get("split_policy", {}).get("split_sha256") != expected:
            blockers.append("split_sha256 缺失或与划分内容不一致")
    elif not _is_sha256(policy.get("split_sha256")):
        blockers.append("split_sha256 缺失")
    return blockers


def _validate_k(errors: list[str], k: np.ndarray, frame_id: str) -> None:
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        errors.append(f"{frame_id} K 的焦距必须是有限正数")
        return
    if abs(k[0, 1]) > 1e-8 or not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-8):
        errors.append(f"{frame_id} K 必须是无斜切的针孔矩阵")


def _validate_pose(errors: list[str], transform: np.ndarray, frame_id: str) -> None:
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        errors.append(f"{frame_id} T_world_camera 底行必须是 [0,0,0,1]")
        return
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        errors.append(f"{frame_id} T_world_camera 不是右手刚体，判定为非刚体或反射")


def _validate_depth(errors: list[str], frame: dict) -> None:
    depth = frame.get("depth")
    frame_id = frame.get("frame_id")
    if not isinstance(depth, dict):
        errors.append(f"{frame_id} depth 记录缺失")
        return
    if depth.get("representation") != "z_depth" or depth.get("unit") != "m":
        errors.append(f"{frame_id} depth 必须是 unit=m 的 z_depth")
    source = depth.get("source")
    if source not in DEPTH_SOURCES:
        errors.append(f"{frame_id} depth.source 非法")
    if source == "none":
        if depth.get("path") is not None or depth.get("sha256") is not None:
            errors.append(f"{frame_id} depth.source=none 时 path 与 sha256 必须为 null")
    elif depth.get("path") in (None, ""):
        errors.append(f"{frame_id} 有深度来源时必须给出 path")


def _validate_frame_size(errors: list[str], frame: dict, canonical_w: object, canonical_h: object) -> None:
    frame_id = frame.get("frame_id")
    width = frame.get("width_px")
    height = frame.get("height_px")
    if not _positive_int(width) or not _positive_int(height):
        errors.append(f"{frame_id} width_px/height_px 必须是正整数")
        return
    if width != canonical_w or height != canonical_h:
        errors.append(f"{frame_id} 尺寸必须与 canonical 图像网格一致")


def _expect_disjoint(errors: list[str], source: list[str], validation: list[str], test: list[str]) -> None:
    groups = (source, validation, test)
    if any(len(group) != len(set(group)) for group in groups):
        errors.append("同一 split 内 frame_id 不能重复")
    if set(source) & set(validation) or set(source) & set(test) or set(validation) & set(test):
        errors.append("source、validation、test 必须互斥")


def _expect_matrix(errors: list[str], value: object, shape: tuple[int, int], label: str) -> None:
    if not _is_matrix(value, shape):
        errors.append(f"{label} 形状必须是 {shape[0]}×{shape[1]}")


def _is_matrix(value: object, shape: tuple[int, int]) -> bool:
    if not isinstance(value, list) or len(value) != shape[0]:
        return False
    return all(isinstance(row, list) and len(row) == shape[1] for row in value)


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX
