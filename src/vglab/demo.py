"""合成参考演示：DP2 teacher → 参考渲染 → 深度残差步。

这是接线演示，不是 Phase 1 运行，不产生实测指标。输出字段显式标记 not_measured。
"""

from __future__ import annotations

import numpy as np

from vglab.depth.teacher import DepthTeacher
from vglab.initialization.candidates import UNFROZEN_DEPTH_REFINER, UNFROZEN_DEPTH_TEACHER
from vglab.initialization.ordinary import initialize_ordinary_gaussians
from vglab.optimize.depth_refine import DepthConstraintRefiner, DepthRefineConfig
from vglab.render.reference import ReferenceCamera, ReferenceScene, render_reference


def synthetic_depth_constraint_demo(*, steps: int = 24, width_px: int = 48, height_px: int = 36) -> dict:
    if steps <= 0:
        raise ValueError("steps 必须为正")
    if width_px < 8 or height_px < 8:
        raise ValueError("演示图像至少 8×8")
    fx = fy = 0.9 * width_px
    k_matrix = np.array(
        [
            [fx, 0.0, (width_px - 1) / 2.0],
            [0.0, fy, (height_px - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4)
    vv, uu = np.mgrid[0:height_px, 0:width_px]
    rgb = np.stack(
        [uu / max(1, width_px - 1), vv / max(1, height_px - 1), np.full_like(uu, 0.5, dtype=np.float64)],
        axis=-1,
    )
    teacher_depth = np.full((height_px, width_px), 2.0)
    initial_depth = np.full((height_px, width_px), 2.4)
    init = initialize_ordinary_gaussians(
        initial_depth,
        rgb,
        k_matrix,
        pose,
        np.ones((height_px, width_px), dtype=bool),
        "src_000",
    )
    scene = ReferenceScene(
        mean_world_m=init.mean_world_m,
        scale_std_m=init.scale_std_m,
        rotation_wxyz=init.rotation_wxyz,
        opacity=init.opacity,
        color_linear_rgb=init.color_linear_rgb,
    )
    camera = ReferenceCamera(t_world_camera=pose, k_matrix=k_matrix, width_px=width_px, height_px=height_px)
    teacher = DepthTeacher(
        frame_id="src_000",
        depth_m=teacher_depth,
        mask=np.ones((height_px, width_px), dtype=bool),
        reliability=np.ones((height_px, width_px)),
        diagnostics={"source": "synthetic_plane", "teacher_valid_px": int(teacher_depth.size)},
        config_name=str(UNFROZEN_DEPTH_TEACHER["name"]),
        frozen=False,
    )
    config = dict(UNFROZEN_DEPTH_REFINER)
    config["max_steps"] = int(steps)
    refiner = DepthConstraintRefiner(
        "D1",
        scene,
        camera,
        teacher,
        config=DepthRefineConfig.from_mapping(config),
    )
    result = refiner.run()
    before = render_reference(scene, camera)
    before_error = _mean_abs_log_error(before.z_expected_m, teacher_depth)
    final_render = render_reference(refiner.current_scene(), camera)
    after_error = _mean_abs_log_error(final_render.z_expected_m, teacher_depth)
    return {
        "kind": "synthetic_reference_demo",
        "not_measured": True,
        "description": "DP2 teacher 平面 → 普通高斯 → 参考渲染 → 深度残差步；仅验证接线，不是 P1a/P1b 记录。",
        "arm": result.arm,
        "status": result.status,
        "steps_run": result.steps_run,
        "stop_reason": result.stop_reason,
        "gaussian_count": result.gaussian_count,
        "image": {"width_px": width_px, "height_px": height_px},
        "loss": {"initial": result.initial_loss, "final": result.final_loss},
        "coverage": {"initial": result.initial_coverage, "final": result.final_coverage},
        "mean_abs_log_error": {"initial": before_error, "final": after_error},
        "teacher_mask_sha256": result.teacher_mask_sha256,
        "measurements": None,
    }


def _mean_abs_log_error(render_depth_m: np.ndarray, teacher_depth_m: np.ndarray) -> float:
    valid = np.isfinite(render_depth_m) & (render_depth_m > 0) & np.isfinite(teacher_depth_m) & (teacher_depth_m > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(np.log(render_depth_m[valid]) - np.log(teacher_depth_m[valid]))))
