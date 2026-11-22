"""区分已实现的执行基础和尚未发生的模型实验。"""

from __future__ import annotations

from vglab.initialization.adapters import environment_blockers
from vglab.paths import protocol_path
from vglab.protocol import load_protocol


def foundation_status(protocol: dict | None = None) -> dict:
    chosen = protocol if protocol is not None else load_protocol(protocol_path())
    return {
        "design": "v0.1",
        "execution_foundations": "implemented_and_unit_tested_on_synthetic_fixtures",
        "official_execution_enabled": bool(chosen.get("execution_enabled")),
        "implementation_exists": bool(chosen.get("implementation_exists")),
        "model_runs": "not_run",
        "measurements": None,
        "p1a_runs": "not_started",
        "p1b_runs": "not_started",
        "blockers": environment_blockers(chosen),
    }


def assert_no_fabricated_measurement(status: dict) -> None:
    if status.get("measurements") is not None:
        raise ValueError("执行基础状态不能携带实测质量数字")
    if status.get("model_runs") != "not_run":
        raise ValueError("模型运行状态只能在真实运行记录中改变")
