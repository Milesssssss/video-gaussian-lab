"""source / validation / test 隔离。"""

from __future__ import annotations

from vglab.errors import LeakageError


def frames_by_id(manifest: dict) -> dict[str, dict]:
    return {frame["frame_id"]: frame for frame in manifest["frames"]}


def assert_context_is_source_only(frame_ids: list[str] | tuple[str, ...], manifest: dict) -> None:
    index = frames_by_id(manifest)
    leaked: list[str] = []
    for frame_id in frame_ids:
        frame = index.get(frame_id)
        if frame is None or frame.get("split") != "source":
            leaked.append(frame_id)
    if leaked:
        joined = ", ".join(leaked)
        raise LeakageError(f"重建上下文包含非 source 帧: {joined}")


def context_frame_ids(frame_ids: list[str] | tuple[str, ...], manifest: dict) -> tuple[str, ...]:
    assert_context_is_source_only(frame_ids, manifest)
    return tuple(frame_ids)
