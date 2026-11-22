"""读取 Phase 1 协议登记，并拒绝尚未满足前置条件的正式运行。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vglab.errors import OfficialRunRefused

ENTRY_CONDITION_KEYS = (
    "data_available",
    "camera_contract_verified",
    "independent_geometry_available",
    "model_use_conditions_resolved",
    "hardware_profiled",
    "runtime_versions_pinned",
    "weight_checksums_recorded",
)
P1B_SEEDS = (17, 29, 43)
CAPACITY_CANDIDATES = (250000, 500000, 1000000)


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    initializer: str
    depth_scale: bool
    rgb_optimization: bool
    depth_loss: bool

    @property
    def consumes_dp2(self) -> bool:
        return self.initializer != "sharp" or self.depth_scale or self.depth_loss

    @property
    def sharp_only(self) -> bool:
        return self.initializer == "sharp" and not self.depth_scale and not self.depth_loss


def load_protocol(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0.1":
        raise ValueError("protocol schema_version must be 0.1")
    if payload.get("kind") != "research_protocol_registration":
        raise ValueError("protocol kind must be research_protocol_registration")
    return payload


def arm_spec(protocol: dict, arm_id: str) -> ArmSpec:
    for arm in protocol["arms"]:
        if arm["id"] == arm_id:
            return ArmSpec(
                arm_id=arm_id,
                initializer=arm["initializer"],
                depth_scale=bool(arm["depth_scale"]),
                rgb_optimization=bool(arm["rgb_optimization"]),
                depth_loss=bool(arm["depth_loss"]),
            )
    raise KeyError(f"unknown arm {arm_id}")


def official_execution_blockers(protocol: dict) -> list[str]:
    blockers: list[str] = []
    if protocol.get("execution_enabled") is not True:
        blockers.append("execution_enabled 不是 true；正式协议运行保持关闭")
    if protocol.get("status") != "frozen":
        blockers.append(f"status={protocol.get('status')!r}，协议尚未冻结")
    conditions = protocol.get("entry_conditions") or {}
    for key in ENTRY_CONDITION_KEYS:
        if conditions.get(key) is not True:
            blockers.append(f"entry_conditions.{key} 未满足")
    if conditions.get("model_use_conditions_resolved") is not True:
        blockers.append("模型使用条件未解决，见 docs/THIRD_PARTY.md")
    sources = protocol.get("candidate_sources") or {}
    for name in ("sharp", "depth"):
        digest = (sources.get(name) or {}).get("weight_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            blockers.append(f"candidate_sources.{name}.weight_sha256 缺失")
    sharp = sources.get("sharp") or {}
    if sharp.get("use_optional_depth_hook") is not False:
        blockers.append("use_optional_depth_hook 必须为 false；Phase 1 不使用 SHARP 可选 depth 参数")
    shared = protocol.get("shared_design") or {}
    capacity = shared.get("frozen_capacity")
    if capacity not in CAPACITY_CANDIDATES:
        blockers.append("frozen_capacity 尚未从 {250000, 500000, 1000000} 冻结")
    if not isinstance(shared.get("frozen_optimization_steps"), int) or shared.get("frozen_optimization_steps", 0) <= 0:
        blockers.append("frozen_optimization_steps 尚未冻结为正整数")
    return blockers


def assert_official_run_allowed(protocol: dict) -> None:
    blockers = official_execution_blockers(protocol)
    if blockers:
        detail = "；".join(blockers)
        raise OfficialRunRefused(f"正式 Phase 1 运行被拒绝：{detail}")


def validate_seed_for_stage(stage: str, seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise OfficialRunRefused(f"{stage} seed 必须是非负整数")
    if stage == "P1b" and seed not in P1B_SEEDS:
        raise OfficialRunRefused("P1b seeds 只允许 17、29、43")
    if stage not in {"P1a", "P1b"}:
        raise OfficialRunRefused(f"未知阶段 {stage}")
