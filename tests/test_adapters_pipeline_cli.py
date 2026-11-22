import json

import numpy as np
import pytest

from tests.manifest_factory import synthetic_manifest
from vglab.cli import main
from vglab.errors import Dp2IsolationError, LeakageError, ModelPreconditionError, OfficialRunRefused
from vglab.initialization.adapters import (
    BlockedDepthAdapter,
    BlockedSharpAdapter,
    assert_sharp_api_has_no_depth_argument,
    dp2_cache_key,
    environment_blockers,
    sharp_cache_key,
)
from vglab.initialization.sharp_scene import CameraGaussian
from vglab.paths import protocol_path
from vglab.pipeline import initialize_sharp_arm, run_official
from vglab.protocol import arm_spec, load_protocol
from vglab.reporting.status import assert_no_fabricated_measurement, foundation_status


def _gaussian() -> CameraGaussian:
    return CameraGaussian(
        gaussian_id="g0",
        frame_id="src_000",
        layer_id=0,
        mean_camera_m=np.array([1.0, 0.0, 2.0]),
        scale_std_m=np.ones(3),
        rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        opacity=0.4,
        color_linear_rgb=np.ones(3),
        source_grid_uv=(0, 0),
        source_grid_shape=(4, 4),
        original_index=0,
    )


def test_blocked_adapters_check_split_before_weights(repo_root):
    protocol = load_protocol(repo_root / "configs" / "phase1_protocol.json")
    sharp = BlockedSharpAdapter(protocol, gpu=False)
    heldout = {"frame_id": "val_000", "split": "validation"}
    with pytest.raises(LeakageError):
        sharp.predict_source_frame(heldout, np.eye(3))
    with pytest.raises(ModelPreconditionError) as caught:
        sharp.predict_source_frame({"frame_id": "src_000", "split": "source"}, np.eye(3))
    assert "THIRD_PARTY.md" in str(caught.value)
    depth = BlockedDepthAdapter(protocol, gpu=False)
    with pytest.raises(LeakageError):
        depth.predict_window([heldout], 10.0)
    assert_sharp_api_has_no_depth_argument()
    blockers = environment_blockers(protocol, gpu=False)
    assert any("nvidia-smi" in item for item in blockers)
    assert any("format_example" in item for item in blockers)
    assert any("60" in item for item in blockers)


def test_cache_keys_change_with_inputs_and_reject_leakage():
    manifest = synthetic_manifest()
    base = sharp_cache_key(
        weights_sha256="a" * 64,
        frame_id="src_000",
        image_sha256="b" * 64,
        k_matrix=np.eye(3),
        transform_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    changed = sharp_cache_key(
        weights_sha256="a" * 64,
        frame_id="src_000",
        image_sha256="b" * 64,
        k_matrix=np.eye(3) * 2,
        transform_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    assert base != changed
    with pytest.raises(LeakageError):
        sharp_cache_key(
            weights_sha256="a" * 64,
            frame_id="src_000",
            image_sha256="b" * 64,
            k_matrix=np.eye(3),
            transform_sha256="c" * 64,
            config_sha256="d" * 64,
            dp2_sha256="e" * 64,
        )
    with pytest.raises(LeakageError):
        dp2_cache_key(
            weights_sha256="a" * 64,
            context_frame_ids=["src_000", "test_000"],
            image_sha256_by_frame={"src_000": "b" * 64, "test_000": "c" * 64},
            focal_px=10.0,
            config_sha256="d" * 64,
            manifest=manifest,
        )


def test_s2_scales_scene_and_s1_rejects_depth():
    protocol = load_protocol(protocol_path())
    pose = np.eye(4)
    pose[0, 3] = 10.0
    depth = np.ones((4, 4))
    result = initialize_sharp_arm(
        arm_spec(protocol, "S2"),
        [_gaussian()],
        {"src_000": pose},
        z_sharp={"src_000": depth},
        z_dp2={"src_000": depth * 2},
        scale_mask={"src_000": np.ones((4, 4), dtype=bool)},
    )
    assert result.placed[0].mean_world_m == pytest.approx(np.array([12.0, 0.0, 4.0]))
    assert result.optimization.status == "not_run_no_verified_renderer"
    assert result.measurements is None
    with pytest.raises(Dp2IsolationError):
        initialize_sharp_arm(
            arm_spec(protocol, "S1"),
            [_gaussian()],
            {"src_000": pose},
            z_dp2={"src_000": depth},
        )
    diagnostic = initialize_sharp_arm(arm_spec(protocol, "S0"), [_gaussian()], {"src_000": pose})
    assert diagnostic.optimization.status == "diagnostic_only"
    assert diagnostic.placed[0].applied_scale == pytest.approx(1.0)
    with pytest.raises(OfficialRunRefused):
        run_official(protocol)


def test_cli_validates_example_and_refuses_official_run(repo_root, capsys):
    assert main(["validate-manifest", str(repo_root / "examples" / "scene_manifest.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "format_example"
    assert report["schema_ok"] is True
    assert report["executable_capture"] is False
    assert main(["validate-manifest", str(repo_root / "examples" / "scene_manifest.json"), "--require-captured"]) == 2
    assert main(["show-blockers"]) == 0
    blockers = capsys.readouterr().out
    assert "execution_enabled" in blockers
    assert "THIRD_PARTY.md" in blockers
    assert main(["run-official"]) == 2
    status = foundation_status(load_protocol(protocol_path()))
    assert_no_fabricated_measurement(status)
    assert status["official_execution_enabled"] is False
    assert status["measurements"] is None
