import numpy as np
import pytest

from vglab.errors import ContractError
from vglab.optimize.losses import depth_supervision, depth_supervision_with_gradient
from vglab.render.reference import (
    ReferenceCamera,
    ReferenceScene,
    depth_render_gradients,
    depth_render_step,
    render_reference,
)

K = np.array([[20.0, 0.0, 6.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]])
POSE = np.eye(4)


def _camera(width: int = 12, height: int = 12) -> ReferenceCamera:
    return ReferenceCamera(t_world_camera=POSE, k_matrix=K, width_px=width, height_px=height)


def _single_scene(z: float = 2.0, opacity: float = 0.75) -> ReferenceScene:
    return ReferenceScene(
        mean_world_m=np.array([[0.0, 0.0, z]]),
        scale_std_m=np.array([[0.02, 0.02, 0.02]]),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([opacity]),
        color_linear_rgb=np.array([[0.5, 0.25, 0.125]]),
    )


def test_single_gaussian_renders_piecewise_constant_depth_and_alpha():
    render = render_reference(_single_scene(z=2.0, opacity=0.75), _camera())
    center = render.z_expected_m[6, 6]
    assert center == pytest.approx(2.0)
    assert render.alpha[6, 6] > 0.5
    inside = np.isfinite(render.z_expected_m)
    assert np.allclose(render.z_expected_m[inside], 2.0)
    assert np.all(np.isnan(render.z_expected_m[~inside]))
    assert np.all(render.alpha[~inside] == 0.0)
    assert render.projected_gaussians == 1
    assert np.all((render.rgb_srgb >= 0.0) & (render.rgb_srgb <= 1.0))


def test_two_layers_match_the_expected_depth_formula():
    scene = ReferenceScene(
        mean_world_m=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        scale_std_m=np.full((2, 3), 0.02),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([0.5, 0.5]),
        color_linear_rgb=np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    render = render_reference(scene, _camera())
    alpha = render.alpha[6, 6]
    expected = (0.5 * 2.0 + (1.0 - 0.5) * 0.5 * 3.0) / (0.5 + (1.0 - 0.5) * 0.5)
    assert alpha == pytest.approx(1.0 - 0.5 * 0.5)
    assert render.z_expected_m[6, 6] == pytest.approx(expected)


def test_background_gaussian_never_overrides_nearer_depth():
    scene = ReferenceScene(
        mean_world_m=np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 1.0]]),
        scale_std_m=np.full((2, 3), 0.02),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([0.9, 0.9]),
        color_linear_rgb=np.ones((2, 3)),
    )
    render = render_reference(scene, _camera())
    assert render.z_expected_m[6, 6] < 2.0


def test_scene_validation_rejects_non_finite_and_out_of_range():
    bad_scale = ReferenceScene(
        mean_world_m=np.zeros((1, 3)),
        scale_std_m=np.zeros((1, 3)),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([0.5]),
        color_linear_rgb=np.zeros((1, 3)),
    )
    with pytest.raises(ContractError):
        render_reference(bad_scale, _camera())
    bad_opacity = ReferenceScene(
        mean_world_m=np.zeros((1, 3)),
        scale_std_m=np.ones((1, 3)) * 0.01,
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([1.0]),
        color_linear_rgb=np.zeros((1, 3)),
    )
    with pytest.raises(ContractError):
        render_reference(bad_opacity, _camera())


def _two_gaussian_scene(z1: float, z2: float) -> ReferenceScene:
    return ReferenceScene(
        mean_world_m=np.array([[0.0, 0.0, z1], [0.0, 0.0, z2]]),
        scale_std_m=np.array([[0.03, 0.03, 0.03], [0.03, 0.03, 0.03]]),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([0.4, 0.3]),
        color_linear_rgb=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )


def _sum_log_depth(scene: ReferenceScene) -> float:
    render = render_reference(scene, _camera())
    valid = np.isfinite(render.z_expected_m)
    return float(np.sum(np.log(render.z_expected_m[valid])))


def test_analytic_depth_gradients_match_finite_differences():
    baseline = _two_gaussian_scene(2.0, 3.0)
    render = render_reference(baseline, _camera(), collect_records=True)
    grad_pixels = np.zeros_like(render.z_expected_m)
    valid = np.isfinite(render.z_expected_m)
    grad_pixels[valid] = 1.0 / render.z_expected_m[valid]
    grad_z, _ = depth_render_gradients(
        render.records,
        grad_pixels,
        num_gaussians=2,
        z_render_m=render.z_expected_m,
    )
    epsilon = 1e-5
    for index in range(2):
        plus = _two_gaussian_scene(2.0 + epsilon * (index == 0), 3.0 + epsilon * (index == 1))
        minus = _two_gaussian_scene(2.0 - epsilon * (index == 0), 3.0 - epsilon * (index == 1))
        finite_difference = (_sum_log_depth(plus) - _sum_log_depth(minus)) / (2 * epsilon)
        assert grad_z[index] == pytest.approx(finite_difference, rel=1e-4, abs=1e-6)


def test_relative_step_moves_expected_depth_by_the_requested_ratio():
    scene = _single_scene(z=2.0, opacity=0.75)
    render = render_reference(scene, _camera(), collect_records=True)
    valid = np.isfinite(render.z_expected_m)
    desired = np.zeros_like(render.z_expected_m)
    desired[valid] = 0.1 * render.z_expected_m[valid]
    step = depth_render_step(render.records, desired, num_gaussians=1)
    moved = ReferenceScene(
        mean_world_m=scene.mean_world_m + step[:, None] * np.array([[0.0, 0.0, 1.0]]),
        scale_std_m=scene.scale_std_m,
        rotation_wxyz=scene.rotation_wxyz,
        opacity=scene.opacity,
        color_linear_rgb=scene.color_linear_rgb,
    )
    after = render_reference(moved, _camera())
    assert after.z_expected_m[6, 6] == pytest.approx(2.2, rel=1e-6)


def test_overlapping_gaussians_do_not_inflate_the_step():
    grid = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
    scene = ReferenceScene(
        mean_world_m=np.array([[x * 0.1, y * 0.1, 2.0] for x, y in grid]),
        scale_std_m=np.full((len(grid), 3), 0.1),
        rotation_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (len(grid), 1)),
        opacity=np.full(len(grid), 0.4),
        color_linear_rgb=np.ones((len(grid), 3)),
    )
    render = render_reference(scene, _camera(), collect_records=True)
    valid = np.isfinite(render.z_expected_m)
    desired = np.zeros_like(render.z_expected_m)
    desired[valid] = 0.1 * render.z_expected_m[valid]
    step = depth_render_step(render.records, desired, num_gaussians=len(grid))
    moved = ReferenceScene(
        mean_world_m=scene.mean_world_m + step[:, None] * np.array([[0.0, 0.0, 1.0]]),
        scale_std_m=scene.scale_std_m,
        rotation_wxyz=scene.rotation_wxyz,
        opacity=scene.opacity,
        color_linear_rgb=scene.color_linear_rgb,
    )
    after = render_reference(moved, _camera())
    assert after.z_expected_m[6, 6] == pytest.approx(2.2, rel=5e-2)


def test_analytic_opacity_gradients_match_finite_differences():
    baseline = ReferenceScene(
        mean_world_m=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        scale_std_m=np.array([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]),
        rotation_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        opacity=np.array([0.4, 0.3]),
        color_linear_rgb=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    render = render_reference(baseline, _camera(), collect_records=True)
    grad_pixels = np.zeros_like(render.z_expected_m)
    valid = np.isfinite(render.z_expected_m)
    grad_pixels[valid] = 1.0 / render.z_expected_m[valid]
    _, grad_alpha = depth_render_gradients(
        render.records,
        grad_pixels,
        num_gaussians=2,
        z_render_m=render.z_expected_m,
    )
    epsilon = 1e-6

    def mean_log_depth_with(which: int, delta: float) -> float:
        opacity = baseline.opacity.copy()
        opacity[which] += delta
        scene = ReferenceScene(
            mean_world_m=baseline.mean_world_m,
            scale_std_m=baseline.scale_std_m,
            rotation_wxyz=baseline.rotation_wxyz,
            opacity=opacity,
            color_linear_rgb=baseline.color_linear_rgb,
        )
        return _sum_log_depth(scene)

    for index in range(2):
        finite_difference = (mean_log_depth_with(index, epsilon) - mean_log_depth_with(index, -epsilon)) / (2 * epsilon)
        assert grad_alpha[index] == pytest.approx(finite_difference, rel=1e-4, abs=1e-6)


def test_empty_scene_returns_empty_render_and_missing_records_are_rejected():
    empty = ReferenceScene(
        mean_world_m=np.zeros((0, 3)),
        scale_std_m=np.zeros((0, 3)),
        rotation_wxyz=np.zeros((0, 4)),
        opacity=np.zeros(0),
        color_linear_rgb=np.zeros((0, 3)),
    )
    render = render_reference(empty, _camera())
    assert render.projected_gaussians == 0
    assert render.records is None
    assert np.all(np.isnan(render.z_expected_m))
    assert np.all(render.alpha == 0.0)
    with pytest.raises(ContractError):
        depth_render_step(None, np.zeros((12, 12)), num_gaussians=1)


def test_gradient_version_matches_pixel_rules_of_depth_supervision():
    render = render_reference(_two_gaussian_scene(2.0, 3.0), _camera())
    teacher = np.full_like(render.z_expected_m, 2.5)
    mask = np.isfinite(render.z_expected_m)
    reliability = np.full_like(render.z_expected_m, 0.8)
    expected = depth_supervision(render.z_expected_m, teacher, mask, reliability, mask)
    outcome = depth_supervision_with_gradient(render.z_expected_m, teacher, mask, reliability, mask)
    assert expected.status == outcome.supervision.status == "computed"
    assert expected.loss == pytest.approx(outcome.supervision.loss)
    assert expected.coverage == pytest.approx(outcome.supervision.coverage)
    assert expected.exclusion_counts == outcome.supervision.exclusion_counts
    empty = depth_supervision_with_gradient(
        render.z_expected_m, np.full_like(teacher, np.nan), mask, reliability, mask
    )
    assert empty.supervision.loss is None
    assert np.all(empty.grad_z_render == 0.0)
