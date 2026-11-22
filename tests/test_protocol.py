import json

import pytest

from vglab.errors import OfficialRunRefused
from vglab.protocol import (
    arm_spec,
    assert_official_run_allowed,
    load_protocol,
    official_execution_blockers,
    validate_seed_for_stage,
)


def test_repository_protocol_is_not_executable(repo_root):
    protocol = load_protocol(repo_root / "configs" / "phase1_protocol.json")
    assert protocol["schema_version"] == "0.1"
    assert protocol["kind"] == "research_protocol_registration"
    assert protocol["execution_enabled"] is False
    assert protocol["status"] == "draft_not_frozen"
    assert protocol["implementation_exists"] is False
    blockers = official_execution_blockers(protocol)
    assert any("execution_enabled" in item for item in blockers)
    assert any("THIRD_PARTY.md" in item for item in blockers)
    assert any("weight_sha256" in item for item in blockers)
    with pytest.raises(OfficialRunRefused) as caught:
        assert_official_run_allowed(protocol)
    assert "execution_enabled" in str(caught.value)


def test_arm_table_matches_protocol_file(repo_root):
    protocol = load_protocol(repo_root / "configs" / "phase1_protocol.json")
    assert arm_spec(protocol, "S0").sharp_only
    assert arm_spec(protocol, "S1").sharp_only
    assert not arm_spec(protocol, "S0").consumes_dp2
    assert not arm_spec(protocol, "S1").consumes_dp2
    s2 = arm_spec(protocol, "S2")
    assert s2.depth_scale and s2.rgb_optimization and not s2.depth_loss
    assert s2.consumes_dp2
    s3 = arm_spec(protocol, "S3")
    assert s3.depth_scale and s3.depth_loss and s3.consumes_dp2
    d0 = arm_spec(protocol, "D0")
    assert d0.initializer == "depth_backprojected_gaussians"
    assert not d0.depth_scale and d0.rgb_optimization and not d0.depth_loss
    assert d0.consumes_dp2
    d1 = arm_spec(protocol, "D1")
    assert d1.depth_loss and d1.consumes_dp2
    assert protocol["shared_design"]["baseline_S0_S1_may_read_DP2"] is False


def test_flipping_execution_flag_still_blocks_unmet_entry_conditions(repo_root):
    protocol = load_protocol(repo_root / "configs" / "phase1_protocol.json")
    protocol["execution_enabled"] = True
    protocol["status"] = "frozen"
    blockers = official_execution_blockers(protocol)
    assert any("data_available" in item for item in blockers)
    assert any("weight_sha256" in item for item in blockers)
    with pytest.raises(OfficialRunRefused):
        assert_official_run_allowed(protocol)


def test_frozen_protocol_with_preconditions_has_no_blockers():
    protocol = {
        "schema_version": "0.1",
        "kind": "research_protocol_registration",
        "status": "frozen",
        "execution_enabled": True,
        "entry_conditions": {
            "data_available": True,
            "camera_contract_verified": True,
            "independent_geometry_available": True,
            "model_use_conditions_resolved": True,
            "hardware_profiled": True,
            "runtime_versions_pinned": True,
            "weight_checksums_recorded": True,
        },
        "candidate_sources": {
            "sharp": {"weight_sha256": "a" * 64, "use_optional_depth_hook": False},
            "depth": {"weight_sha256": "b" * 64},
        },
        "shared_design": {
            "baseline_S0_S1_may_read_DP2": False,
            "frozen_capacity": 250000,
            "frozen_optimization_steps": 100,
            "capacity_candidates": [250000, 500000, 1000000],
        },
        "arms": [
            {"id": "S0", "initializer": "sharp", "depth_scale": False, "rgb_optimization": False, "depth_loss": False},
            {"id": "S1", "initializer": "sharp", "depth_scale": False, "rgb_optimization": True, "depth_loss": False},
        ],
    }
    assert official_execution_blockers(protocol) == []
    assert_official_run_allowed(protocol)


def test_optional_depth_hook_cannot_be_enabled_for_official_runs():
    protocol = {
        "schema_version": "0.1",
        "kind": "research_protocol_registration",
        "status": "frozen",
        "execution_enabled": True,
        "entry_conditions": {key: True for key in (
            "data_available",
            "camera_contract_verified",
            "independent_geometry_available",
            "model_use_conditions_resolved",
            "hardware_profiled",
            "runtime_versions_pinned",
            "weight_checksums_recorded",
        )},
        "candidate_sources": {
            "sharp": {"weight_sha256": "a" * 64, "use_optional_depth_hook": True},
            "depth": {"weight_sha256": "b" * 64},
        },
        "shared_design": {
            "frozen_capacity": 250000,
            "frozen_optimization_steps": 10,
            "capacity_candidates": [250000, 500000, 1000000],
        },
        "arms": [],
    }
    blockers = official_execution_blockers(protocol)
    assert any("use_optional_depth_hook" in item for item in blockers)


def test_p1b_seeds_are_fixed_and_p1a_seed_is_not_invented():
    validate_seed_for_stage("P1b", 17)
    validate_seed_for_stage("P1b", 29)
    validate_seed_for_stage("P1b", 43)
    with pytest.raises(OfficialRunRefused):
        validate_seed_for_stage("P1b", 1)
    validate_seed_for_stage("P1a", 1)
    with pytest.raises(OfficialRunRefused):
        validate_seed_for_stage("P1b", -1)


def test_load_protocol_rejects_unknown_kind(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"schema_version": "0.1", "kind": "experiment_result"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(path)
