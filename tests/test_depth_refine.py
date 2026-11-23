import numpy as np
import pytest

from vglab.depth.teacher import DepthTeacher
from vglab.errors import ContractError, Dp2IsolationError
from vglab.initialization.ordinary import initialize_ordinary_gaussians
from vglab.optimize.depth_refine import DepthConstraintRefiner, DepthRefineConfig
from vglab.render.reference import ReferenceCamera, ReferenceScene, render_reference

K = np.array([[43.2, 0.0, 23.5], [0.0, 43.2, 17.5], [0.0, 0.0, 1.0]])
POSE = np.eye(4)
HEIGHT = 36
WIDTH = 48


def _scene(depth_value: float = 2.4) -> ReferenceScene:
    vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH]
    rgb = np.stack(
        [uu / (WIDTH - 1), vv / (HEIGHT - 1), np.full_like(uu, 0.5, dtype=np.float64)],
        axis=-1,
    )
    depth = np.full((HEIGHT, WIDTH), depth_value)
    init = initialize_ordinary_gaussians(
        depth, rgb, K, POSE, np.ones_like(depth, dtype=bool), "src_000"
    )
    return ReferenceScene(
        mean_world_m=init.mean_world_m,
        scale_std_m=init.scale_std_m,
        rotation_wxyz=init.rotation_wxyz,
        opacity=init.opacity,
        color_linear_rgb=init.color_linear_rgb,
    )


def _teacher(value: float = 2.0, mask: np.ndarray | None = None) -> DepthTeacher:
    values = np.full((HEIGHT, WIDTH), value)
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    return DepthTeacher(
        frame_id="src_000",
        depth_m=values,
        mask=mask,
        reliability=np.where(mask, 1.0, 0.0),
        diagnostics={},
        config_name="test",
    )


def _camera() -> ReferenceCamera:
    return ReferenceCamera(t_world_camera=POSE, k_matrix=K, width_px=WIDTH, height_px=HEIGHT)


def test_refinement_reduces_teacher_error_without_touching_pose_or_mask():
    scene = _scene(2.4)
    camera = _camera()
    teacher = _teacher(2.0)
    mask_before = teacher.mask.copy()
    poses_before = camera.t_world_camera.copy()
    k_before = camera.k_matrix.copy()
    config = DepthRefineConfig(lr_log_step=0.35, log_step_cap=0.5, max_steps=25)
    refiner = DepthConstraintRefiner("D1", scene, camera, teacher, config=config)
    result = refiner.run()
    assert result.status == "ran_reference_depth_pass"
    assert result.steps_run >= 10
    assert result.initial_loss is not None and result.final_loss is not None
    assert result.final_loss < result.initial_loss * 0.2
    assert result.initial_coverage == pytest.approx(1.0)
    assert result.final_coverage == pytest.approx(1.0)
    assert np.array_equal(teacher.mask, mask_before)
    assert np.array_equal(camera.t_world_camera, poses_before)
    assert np.array_equal(camera.k_matrix, k_before)
    assert result.pose_updated is False
    assert result.densification is False
    assert result.measurements is None
    final = render_reference(refiner.current_scene(), camera)
    valid = np.isfinite(final.z_expected_m)
    assert np.mean(np.abs(np.log(final.z_expected_m[valid]) - np.log(2.0))) < 0.02
    assert result.gaussian_count == scene.mean_world_m.shape[0]


def test_depth_arms_are_isolated_from_sharp_only_controls():
    scene = _scene()
    camera = _camera()
    teacher = _teacher()
    for arm in ("S0", "S1"):
        with pytest.raises(Dp2IsolationError):
            DepthConstraintRefiner(arm, scene, camera, teacher)
    for arm in ("S2", "D0"):
        with pytest.raises(ContractError):
            DepthConstraintRefiner(arm, scene, camera, teacher)


def test_empty_teacher_skips_without_moving_gaussians():
    scene = _scene()
    camera = _camera()
    empty = _teacher(2.0, mask=np.zeros((HEIGHT, WIDTH), dtype=bool))
    refiner = DepthConstraintRefiner("S3", scene, camera, empty, config=DepthRefineConfig(max_steps=5))
    before = refiner.current_scene().mean_world_m.copy()
    result = refiner.run()
    assert result.status == "skipped_empty_depth_supervision"
    assert result.steps_run == 0
    assert result.initial_loss is None
    assert result.final_loss is None
    assert result.final_coverage is None
    assert np.array_equal(refiner.current_scene().mean_world_m, before)


def test_sustained_low_coverage_stops_the_pass():
    scene = _scene()
    camera = _camera()
    reliability = np.zeros((HEIGHT, WIDTH))
    reliability[:2, :2] = 1.0
    teacher = DepthTeacher(
        frame_id="src_000",
        depth_m=np.full((HEIGHT, WIDTH), 2.0),
        mask=np.ones((HEIGHT, WIDTH), dtype=bool),
        reliability=reliability,
        diagnostics={},
        config_name="test",
    )
    config = DepthRefineConfig(min_coverage=0.5, patience=2, max_steps=10)
    refiner = DepthConstraintRefiner("D1", scene, camera, teacher, config=config)
    result = refiner.run()
    assert result.status == "stopped_low_coverage"
    assert result.steps_run == 3
    assert result.final_coverage < 0.5


def test_teacher_grid_must_match_camera():
    scene = _scene()
    camera = _camera()
    wrong = DepthTeacher(
        frame_id="src_000",
        depth_m=np.full((10, 10), 2.0),
        mask=np.ones((10, 10), dtype=bool),
        reliability=np.ones((10, 10)),
        diagnostics={},
        config_name="test",
    )
    with pytest.raises(ContractError):
        DepthConstraintRefiner("D1", scene, camera, wrong)


def test_optional_opacity_updates_stay_finite_and_bounded():
    scene = _scene(2.4)
    camera = _camera()
    teacher = _teacher(2.0)
    config = DepthRefineConfig(
        lr_log_step=0.2,
        lr_opacity=0.05,
        update_opacity=True,
        max_steps=6,
    )
    refiner = DepthConstraintRefiner("D1", scene, camera, teacher, config=config)
    result = refiner.run()
    assert result.status == "ran_reference_depth_pass"
    opacity = 1.0 / (1.0 + np.exp(-refiner.opacity_logit))
    assert np.all(np.isfinite(opacity))
    assert np.all((opacity > 0.0) & (opacity < 1.0))
    assert result.final_loss is not None and result.final_loss < result.initial_loss


def test_demo_reports_not_measured_and_improves_depth():
    from vglab.demo import synthetic_depth_constraint_demo

    payload = synthetic_depth_constraint_demo(steps=20, width_px=32, height_px=24)
    assert payload["kind"] == "synthetic_reference_demo"
    assert payload["not_measured"] is True
    assert payload["measurements"] is None
    assert payload["status"] == "ran_reference_depth_pass"
    assert payload["loss"]["final"] < payload["loss"]["initial"] * 0.25
    assert payload["mean_abs_log_error"]["final"] < payload["mean_abs_log_error"]["initial"] * 0.25
