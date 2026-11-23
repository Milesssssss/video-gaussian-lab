import numpy as np
import pytest

from tests.manifest_factory import synthetic_manifest
from vglab.depth.aggregate import DepthPrediction
from vglab.errors import ContractError, Dp2IsolationError, LeakageError, ModelPreconditionError
from vglab.initialization.adapters import (
    DepthPro2Adapter,
    ModelPreconditions,
    SharpAdapter,
    SharpImageOutput,
    SharpLayerOutput,
    camera_gaussians_from_sharp_output,
    detect_accelerators,
    precondition_blockers,
    resolve_model_preconditions,
)
from vglab.paths import protocol_path
from vglab.protocol import load_protocol

DIGEST = "a" * 64
K = np.array([[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])


def _preconditions(backend: str = "sharp", accelerators=("mps",), resolved: bool = True) -> ModelPreconditions:
    return ModelPreconditions(
        backend=backend,
        weights_sha256=DIGEST,
        model_use_conditions_resolved=resolved,
        accelerators=accelerators,
    )


def _layer(count: int = 2) -> SharpLayerOutput:
    return SharpLayerOutput(
        mean_camera_m=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])[:count],
        scale_std_m=np.full((count, 3), 0.01),
        rotation_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (count, 1)),
        opacity=np.full(count, 0.5),
        color_linear_rgb=np.full((count, 3), 0.5),
        grid_uv=np.array([[0, 0], [1, 1]])[:count],
        grid_shape=(4, 4),
    )


class _CountingSharpBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.focal = None

    def predict_image(self, image_srgb, focal_px, config):
        self.calls += 1
        self.focal = focal_px
        return SharpImageOutput(layers=(_layer(), _layer(count=2)))


class _CountingDepthBackend:
    def __init__(self, value: float = 2.0) -> None:
        self.calls = 0
        self.value = value

    def predict_window(self, images_srgb, focal_px, config):
        self.calls += 1
        return [np.full(image.shape[:2], self.value) for image in images_srgb]


def test_preconditions_block_before_any_backend_call():
    backend = _CountingSharpBackend()
    missing_license = _preconditions(resolved=False)
    adapter = SharpAdapter(backend, missing_license)
    frame = synthetic_manifest()["frames"][0]
    with pytest.raises(ModelPreconditionError) as caught:
        adapter.predict_source_frame(frame, np.zeros((4, 4, 3)), K)
    assert "THIRD_PARTY.md" in str(caught.value)
    assert backend.calls == 0

    no_accelerator = _preconditions(accelerators=("cpu",))
    adapter = SharpAdapter(backend, no_accelerator)
    with pytest.raises(ModelPreconditionError):
        adapter.predict_source_frame(frame, np.zeros((4, 4, 3)), K)
    assert backend.calls == 0
    assert precondition_blockers(resolve_model_preconditions(load_protocol(protocol_path()), "sharp"))


def test_sharp_adapter_forwards_once_and_converts_two_layers():
    backend = _CountingSharpBackend()
    adapter = SharpAdapter(backend, _preconditions())
    frame = synthetic_manifest()["frames"][0]
    result = adapter.predict_source_frame(frame, np.full((4, 4, 3), 0.25), K)
    assert backend.calls == 1
    assert backend.focal == pytest.approx(2.0)
    assert result.layer_count == 2
    assert len(result.gaussians) == 4
    assert {item.layer_id for item in result.gaussians} == {0, 1}
    assert result.gaussians[0].gaussian_id == "src_000:L0:000000"
    assert result.gaussians[0].source_grid_shape == (4, 4)
    assert all(item.frame_id == "src_000" for item in result.gaussians)


def test_sharp_adapter_rejects_heldout_frame_and_bad_outputs():
    adapter = SharpAdapter(_CountingSharpBackend(), _preconditions())
    heldout = {"frame_id": "val_000", "split": "validation", "width_px": 4, "height_px": 4}
    with pytest.raises(LeakageError):
        adapter.predict_source_frame(heldout, np.zeros((4, 4, 3)), K)
    frame = synthetic_manifest()["frames"][0]
    bad = SharpImageOutput(layers=(_layer(), _layer(), _layer()))
    with pytest.raises(ContractError):
        camera_gaussians_from_sharp_output(bad, frame_id="src_000")
    unnormalized = _layer()
    unnormalized = SharpLayerOutput(
        mean_camera_m=unnormalized.mean_camera_m,
        scale_std_m=unnormalized.scale_std_m,
        rotation_wxyz=unnormalized.rotation_wxyz * 2.0,
        opacity=unnormalized.opacity,
        color_linear_rgb=unnormalized.color_linear_rgb,
        grid_uv=unnormalized.grid_uv,
        grid_shape=unnormalized.grid_shape,
    )
    with pytest.raises(ContractError):
        camera_gaussians_from_sharp_output(SharpImageOutput(layers=(unnormalized, _layer())), frame_id="src_000")


def test_depth_adapter_returns_source_only_predictions():
    backend = _CountingDepthBackend()
    adapter = DepthPro2Adapter(backend, _preconditions("depth"))
    manifest = synthetic_manifest()
    frames = manifest["frames"][:2]
    images = {frame["frame_id"]: np.full((4, 4, 3), 0.5) for frame in frames}
    k_by_frame = {frame["frame_id"]: K for frame in frames}
    predictions = adapter.predict_window(frames, images, k_by_frame)
    assert backend.calls == 1
    assert set(predictions) == {"src_000", "src_001"}
    first = predictions["src_000"]
    assert isinstance(first, DepthPrediction)
    assert first.context_frame_ids == ("src_000", "src_001")
    assert first.representation == "z_depth" and first.unit == "m"
    assert first.k_matrix is not None
    result = adapter.predict_window(frames, images, k_by_frame, pose_by_frame={"src_000": np.eye(4)})
    assert result["src_000"].t_world_camera is not None


def test_depth_adapter_rejects_mixed_focal_heldout_and_negative_depth():
    backend = _CountingDepthBackend()
    adapter = DepthPro2Adapter(backend, _preconditions("depth"))
    frames = synthetic_manifest()["frames"][:2]
    images = {frame["frame_id"]: np.full((4, 4, 3), 0.5) for frame in frames}
    other_k = K.copy()
    other_k[0, 0] = 3.0
    other_k[1, 1] = 3.0
    with pytest.raises(ContractError):
        adapter.predict_window(frames, images, {"src_000": K, "src_001": other_k})
    assert backend.calls == 0
    heldout = synthetic_manifest()["frames"][2]
    mixed = frames + [heldout]
    mixed_images = {frame["frame_id"]: np.full((4, 4, 3), 0.5) for frame in mixed}
    with pytest.raises(LeakageError):
        adapter.predict_window(mixed, mixed_images, {frame["frame_id"]: K for frame in mixed})
    negative = DepthPro2Adapter(_CountingDepthBackend(value=-1.0), _preconditions("depth"))
    with pytest.raises(ContractError):
        negative.predict_window(frames, images, {frame["frame_id"]: K for frame in frames})


def test_adapter_cache_keys_require_recorded_weights():
    adapter = SharpAdapter(_CountingSharpBackend(), _preconditions())
    frame = synthetic_manifest()["frames"][0]
    key = adapter.cache_key(
        frame,
        image_sha256="b" * 64,
        transform_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    assert len(key) == 64
    unknown = SharpAdapter(_CountingSharpBackend(), _preconditions())
    unknown.preconditions = ModelPreconditions("sharp", None, True, ("mps",))
    with pytest.raises(ModelPreconditionError):
        unknown.cache_key(frame, image_sha256="b" * 64, transform_sha256="c" * 64, config_sha256="d" * 64)


def test_accelerator_detection_is_honest_on_this_host():
    names = detect_accelerators()
    assert "cpu" in names
    assert set(names) <= {"cuda", "mps", "cpu"}
