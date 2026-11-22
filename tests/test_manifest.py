import copy
import json

import pytest

from tests.manifest_factory import synthetic_manifest
from vglab.errors import LeakageError
from vglab.ids import split_sha256
from vglab.ingest.manifest import execution_blockers, load_manifest, validate_schema
from vglab.ingest.split import assert_context_is_source_only, context_frame_ids


def test_format_example_parses_but_cannot_execute(repo_root):
    manifest = load_manifest(repo_root / "examples" / "scene_manifest.json")
    assert manifest["kind"] == "format_example"
    schema = validate_schema(manifest)
    assert schema.ok
    blockers = execution_blockers(manifest)
    assert any("format_example" in item for item in blockers)
    assert any("rgb_sha256" in item for item in blockers)
    assert any("不可作为" in item or "不能进入执行" in item for item in blockers)


def test_synthetic_fixture_passes_schema_and_stays_non_capture():
    manifest = synthetic_manifest()
    assert validate_schema(manifest).ok
    blockers = execution_blockers(manifest)
    assert any("captured_scene" in item for item in blockers)
    assert all("非刚体" not in item for item in blockers)


def test_split_lists_must_match_frames_and_stay_disjoint():
    manifest = synthetic_manifest()
    manifest["frames"][0]["split"] = "test"
    report = validate_schema(manifest)
    assert not report.ok
    manifest = synthetic_manifest()
    manifest["split_policy"]["source_frame_ids"].append("val_000")
    report = validate_schema(manifest)
    assert not report.ok


def test_non_rigid_pose_and_wrong_units_fail_schema():
    manifest = synthetic_manifest()
    manifest["frames"][0]["T_world_camera"][0][1] = 0.2
    assert not validate_schema(manifest).ok
    manifest = synthetic_manifest()
    manifest["units"]["distance"] = "cm"
    assert not validate_schema(manifest).ok
    manifest = synthetic_manifest()
    manifest["camera_axes"] = "arkit"
    assert not validate_schema(manifest).ok


def test_depth_none_must_not_carry_a_file_and_z_depth_is_required():
    manifest = synthetic_manifest()
    manifest["frames"][0]["depth"]["path"] = "depth.npy"
    assert not validate_schema(manifest).ok
    manifest = synthetic_manifest()
    manifest["frames"][0]["depth"]["representation"] = "range"
    assert not validate_schema(manifest).ok
    manifest = synthetic_manifest()
    manifest["depth_representation"] = "range"
    assert not validate_schema(manifest).ok


def test_reflection_pose_is_rejected():
    manifest = synthetic_manifest()
    manifest["frames"][0]["T_world_camera"][1][1] = -1.0
    manifest["frames"][0]["T_world_camera"][2][2] = -1.0
    # det of diag(1,-1,-1) is +1, so this is still a proper rotation.
    # A reflection is diag(1,-1,1).
    manifest["frames"][0]["T_world_camera"] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert not validate_schema(manifest).ok


def test_captured_scene_requires_hashes_and_records_split_hash():
    manifest = synthetic_manifest()
    manifest["kind"] = "captured_scene"
    blockers = execution_blockers(manifest)
    assert any("split_sha256" in item for item in blockers)
    frames = {frame["frame_id"]: frame["rgb_sha256"] for frame in manifest["frames"]}
    policy = manifest["split_policy"]
    manifest["split_policy"]["split_sha256"] = split_sha256(
        manifest["scene_id"],
        policy["source_frame_ids"],
        policy["validation_frame_ids"],
        policy["test_frame_ids"],
        frames,
    )
    blockers = execution_blockers(manifest, require_files=False)
    assert blockers == []


def test_used_eval_rgb_must_be_declared_for_captured_scenes():
    manifest = synthetic_manifest()
    manifest["kind"] = "captured_scene"
    frames = {frame["frame_id"]: frame["rgb_sha256"] for frame in manifest["frames"]}
    policy = manifest["split_policy"]
    manifest["split_policy"]["split_sha256"] = split_sha256(
        manifest["scene_id"],
        policy["source_frame_ids"],
        policy["validation_frame_ids"],
        policy["test_frame_ids"],
        frames,
    )
    del manifest["pose_provider"]["used_eval_rgb"]
    blockers = execution_blockers(manifest, require_files=False)
    assert any("used_eval_rgb" in item for item in blockers)


def test_context_builder_rejects_heldout_frames():
    manifest = synthetic_manifest()
    with pytest.raises(LeakageError) as caught:
        assert_context_is_source_only(["src_000", "val_000"], manifest)
    assert "val_000" in str(caught.value)
    assert context_frame_ids(["src_001", "src_000"], manifest) == ("src_001", "src_000")


def test_load_manifest_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_mismatched_split_hash_blocks_captured_execution():
    manifest = synthetic_manifest()
    manifest["kind"] = "captured_scene"
    manifest["split_policy"]["split_sha256"] = "0" * 64
    blockers = execution_blockers(manifest, require_files=False)
    assert any("split_sha256" in item for item in blockers)


def test_schema_report_is_independent_of_caller_mutation():
    manifest = synthetic_manifest()
    report = validate_schema(manifest)
    manifest["scene_id"] = ""
    assert report.ok
    assert not validate_schema(manifest).ok
    _ = copy.deepcopy(manifest)
