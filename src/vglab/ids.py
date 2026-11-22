"""实验 ID 与帧划分哈希。"""

from __future__ import annotations

from dataclasses import dataclass

from vglab.errors import ContractError
from vglab.provenance import canonical_json_sha256

OFFICIAL_ARMS = ("S0", "S1", "S2", "S3", "D0", "D1")
SCENE_SPLITS = ("dev", "test")


@dataclass(frozen=True)
class ExperimentId:
    scene_id: str
    scene_split: str
    arm: str
    seed: int
    config_sha256: str

    def text(self) -> str:
        if self.arm not in OFFICIAL_ARMS:
            raise ContractError(f"未知实验臂 {self.arm}")
        if self.scene_split not in SCENE_SPLITS:
            raise ContractError("scene_split 只能是 dev 或 test")
        if not isinstance(self.config_sha256, str) or len(self.config_sha256) != 64:
            raise ContractError("config_sha256 必须是 64 位十六进制摘要")
        if not self.scene_id or "/" in self.scene_id or ":" in self.scene_id:
            raise ContractError("scene_id 不能为空且不能包含分隔符")
        prefix = self.config_sha256[:12]
        return f"p1:{self.scene_split}:{self.scene_id}:{self.arm}:seed{self.seed}:{prefix}"


def split_sha256(
    scene_id: str,
    source_frame_ids: list[str],
    validation_frame_ids: list[str],
    test_frame_ids: list[str],
    rgb_sha256_by_frame_id: dict[str, str | None],
) -> str:
    """哈希冻结的划分内容。不读取 split_sha256 字段本身。"""
    ordered = list(source_frame_ids) + list(validation_frame_ids) + list(test_frame_ids)
    payload = {
        "scene_id": scene_id,
        "source_frame_ids": list(source_frame_ids),
        "validation_frame_ids": list(validation_frame_ids),
        "test_frame_ids": list(test_frame_ids),
        "rgb_sha256_by_frame_id": {frame_id: rgb_sha256_by_frame_id[frame_id] for frame_id in ordered},
    }
    return canonical_json_sha256(payload)
