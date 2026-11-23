"""确定性 NumPy 参考渲染器，供合成夹具验证深度约束链路。

限制（必须保留在报告里）：

- 每个高斯按其中心 z 全局排序后依次合成；像素内的真实顺序在中心顺序一致时相同。
  对前后景边界、大尺度倾斜薄片不是正式的可见性解算。
- 协方差按 EWA 近似投影到 2D，每个高斯用轴对齐补丁与二次型衰减绘制。
- 不支持球谐、逐高斯各向异性足迹裁剪、反走样、densification。
- CPU 单线程 Python 循环，复杂度 O(N × 补丁像素)，只用于小规模合成夹具。

它输出设计 §10.3 的 front-to-back 归一化期望深度与 alpha，并提供 `depth_render_step`
与 `depth_render_gradients` 两个反向接口。正式实验必须换成经过单帧对齐验证的共享渲染器。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vglab.calibration.camera import as_intrinsics, as_pose, invert_world_camera
from vglab.errors import ContractError
from vglab.scene.gaussians import covariance


@dataclass(frozen=True)
class ReferenceScene:
    mean_world_m: np.ndarray
    scale_std_m: np.ndarray
    rotation_wxyz: np.ndarray
    opacity: np.ndarray
    color_linear_rgb: np.ndarray


@dataclass(frozen=True)
class ReferenceCamera:
    t_world_camera: np.ndarray
    k_matrix: np.ndarray
    width_px: int
    height_px: int


@dataclass(frozen=True)
class PixelRecords:
    """按绘制顺序记录的逐像素合成输入。只在 collect_records=True 时产生。"""

    pixel_index: np.ndarray
    gaussian_index: np.ndarray
    a: np.ndarray
    d_a_d_opacity: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class RenderOutput:
    rgb_srgb: np.ndarray
    alpha: np.ndarray
    z_expected_m: np.ndarray
    mass: np.ndarray
    records: PixelRecords | None
    projected_gaussians: int


def _validate(scene: ReferenceScene, camera: ReferenceCamera) -> tuple[np.ndarray, np.ndarray]:
    pose = as_pose(camera.t_world_camera)
    intrinsics = as_intrinsics(camera.k_matrix)
    if (
        not isinstance(camera.width_px, int)
        or isinstance(camera.width_px, bool)
        or not isinstance(camera.height_px, int)
        or isinstance(camera.height_px, bool)
        or camera.width_px <= 0
        or camera.height_px <= 0
    ):
        raise ContractError("渲染尺寸必须是正整数")
    means = np.asarray(scene.mean_world_m, dtype=np.float64)
    scales = np.asarray(scene.scale_std_m, dtype=np.float64)
    rotations = np.asarray(scene.rotation_wxyz, dtype=np.float64)
    opacity = np.asarray(scene.opacity, dtype=np.float64).reshape(-1)
    colors = np.asarray(scene.color_linear_rgb, dtype=np.float64)
    count = len(means)
    if means.shape != (count, 3) or scales.shape != (count, 3) or rotations.shape != (count, 4):
        raise ContractError("场景参数必须是 (N,3)/(N,3)/(N,4)")
    if colors.shape != (count, 3) or opacity.shape != (count,):
        raise ContractError("颜色与透明度数量必须与高斯一致")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or not np.isfinite(colors).all():
        raise ContractError("参考渲染只接受有限参数")
    if np.any(scales <= 0):
        raise ContractError("scale_std_m 必须是正数")
    if np.any((opacity <= 0) | (opacity >= 1)):
        raise ContractError("参考渲染要求 opacity 在 (0,1)，避免 alpha=1 后无梯度")
    if np.any((colors < 0) | (colors > 1)):
        raise ContractError("线性 RGB 必须在 [0,1]")
    return pose, intrinsics


def render_reference(
    scene: ReferenceScene,
    camera: ReferenceCamera,
    *,
    near_m: float = 1e-4,
    patch_sigma: float = 3.0,
    max_radius_px: float = 96.0,
    collect_records: bool = False,
) -> RenderOutput:
    pose, intrinsics = _validate(scene, camera)
    height = camera.height_px
    width = camera.width_px
    count = len(scene.mean_world_m)
    empty_rgb = np.zeros((height, width, 3))
    empty_scalar = np.zeros((height, width))
    nan_depth = np.full((height, width), np.nan)
    if count == 0:
        return RenderOutput(empty_rgb, empty_scalar, nan_depth, empty_scalar, None, 0)

    camera_from_world = invert_world_camera(pose)
    rotation_cw = camera_from_world[:3, :3]
    translation_cw = camera_from_world[:3, 3]
    means_cam = scene.mean_world_m @ rotation_cw.T + translation_cw
    z_cam = means_cam[:, 2]
    finite = np.isfinite(means_cam).all(axis=1) & (z_cam > near_m)
    index = np.nonzero(finite)[0]
    if index.size == 0:
        return RenderOutput(empty_rgb, empty_scalar, nan_depth, empty_scalar, None, 0)

    cov_world = covariance(scene.scale_std_m, scene.rotation_wxyz)
    cov_cam = np.einsum("ij,njk,lk->nil", rotation_cw, cov_world, rotation_cw)
    x = means_cam[index, 0]
    y = means_cam[index, 1]
    z = means_cam[index, 2]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = fx * x / z + cx
    v = fy * y / z + cy
    jacobian = np.zeros((index.size, 2, 3))
    jacobian[:, 0, 0] = fx / z
    jacobian[:, 0, 2] = -fx * x / (z * z)
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 1, 2] = -fy * y / (z * z)
    cov_2d = np.einsum("nij,njk,nlk->nil", jacobian, cov_cam[index], jacobian)
    a11 = cov_2d[:, 0, 0]
    a12 = cov_2d[:, 0, 1]
    a22 = cov_2d[:, 1, 1]
    det = a11 * a22 - a12 * a12
    usable = np.isfinite(det) & (det > 1e-12) & (a11 > 0) & (a22 > 0)
    if not np.any(usable):
        return RenderOutput(empty_rgb, empty_scalar, nan_depth, empty_scalar, None, 0)
    index = index[usable]
    u, v, z = u[usable], v[usable], z[usable]
    a11, a12, a22, det = a11[usable], a12[usable], a22[usable], det[usable]
    lambda_max = 0.5 * (a11 + a22) + np.sqrt(0.25 * (a11 - a22) ** 2 + a12 * a12)
    radius = np.clip(patch_sigma * np.sqrt(np.maximum(lambda_max, 0.0)), 0.0, max_radius_px)
    opacity = scene.opacity[index]
    colors = scene.color_linear_rgb[index]

    order = np.argsort(z, kind="stable")
    transmittance = np.ones((height, width), dtype=np.float64)
    acc_z = np.zeros((height, width), dtype=np.float64)
    acc_rgb = np.zeros((height, width, 3), dtype=np.float64)
    mass = np.zeros((height, width), dtype=np.float64)
    record_pixel: list[np.ndarray] = []
    record_gaussian: list[np.ndarray] = []
    record_a: list[np.ndarray] = []
    record_da: list[np.ndarray] = []
    record_z: list[np.ndarray] = []
    t_flat = transmittance.reshape(-1)
    acc_z_flat = acc_z.reshape(-1)
    mass_flat = mass.reshape(-1)
    acc_rgb_flat = acc_rgb.reshape(-1, 3)
    projected = 0

    for position in order:
        radius_px = float(radius[position])
        u_center = float(u[position])
        v_center = float(v[position])
        u0 = max(0, int(np.ceil(u_center - radius_px)))
        u1 = min(width - 1, int(np.floor(u_center + radius_px)))
        v0 = max(0, int(np.ceil(v_center - radius_px)))
        v1 = min(height - 1, int(np.floor(v_center + radius_px)))
        if u0 > u1 or v0 > v1:
            continue
        projected += 1
        du = np.arange(u0, u1 + 1, dtype=np.float64)[None, :] - u_center
        dv = np.arange(v0, v1 + 1, dtype=np.float64)[:, None] - v_center
        maha2 = (
            a22[position] * du * du
            - 2.0 * a12[position] * du * dv
            + a11[position] * dv * dv
        ) / det[position]
        falloff = np.exp(-0.5 * maha2)
        per_pixel_alpha = opacity[position] * falloff
        unclamped = per_pixel_alpha < 1.0
        a_values = np.minimum(per_pixel_alpha, 1.0)
        a_values = np.where(falloff > 1e-8, a_values, 0.0)
        derivative = np.where(unclamped & (falloff > 1e-8), falloff, 0.0)
        pixel = ((np.arange(v0, v1 + 1)[:, None] * width + np.arange(u0, u1 + 1)[None, :])).reshape(-1)
        a_flat = a_values.reshape(-1)
        weight = t_flat[pixel] * a_flat
        acc_z_flat[pixel] += weight * z[position]
        mass_flat[pixel] += weight
        acc_rgb_flat[pixel] += weight[:, None] * colors[position]
        t_flat[pixel] *= 1.0 - a_flat
        if collect_records:
            used = a_flat > 0.0
            if np.any(used):
                record_pixel.append(pixel[used])
                record_gaussian.append(np.full(int(used.sum()), int(index[position]), dtype=np.int64))
                record_a.append(a_flat[used])
                record_da.append(derivative.reshape(-1)[used])
                record_z.append(np.full(int(used.sum()), float(z[position])))

    safe_mass = mass > 1e-12
    z_expected = np.full((height, width), np.nan)
    z_expected[safe_mass] = acc_z[safe_mass] / mass[safe_mass]
    mean_linear = np.zeros_like(acc_rgb)
    mean_linear[safe_mass] = acc_rgb[safe_mass] / mass[safe_mass][:, None]
    rgb_srgb = _linear_to_srgb(mean_linear)
    rgb_srgb[~safe_mass] = 0.0
    alpha = 1.0 - transmittance
    records = None
    if collect_records:
        if record_pixel:
            records = PixelRecords(
                pixel_index=np.concatenate(record_pixel),
                gaussian_index=np.concatenate(record_gaussian),
                a=np.concatenate(record_a),
                d_a_d_opacity=np.concatenate(record_da),
                z=np.concatenate(record_z),
            )
        else:
            records = PixelRecords(
                pixel_index=np.zeros(0, dtype=np.int64),
                gaussian_index=np.zeros(0, dtype=np.int64),
                a=np.zeros(0),
                d_a_d_opacity=np.zeros(0),
                z=np.zeros(0),
            )
    return RenderOutput(rgb_srgb, alpha, z_expected, mass, records, projected)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def _pixel_groups(records: PixelRecords) -> list[tuple[int, np.ndarray]]:
    if len(records.pixel_index) == 0:
        return []
    order = np.argsort(records.pixel_index, kind="stable")
    pixels = records.pixel_index[order]
    boundaries = np.flatnonzero(np.r_[True, pixels[1:] != pixels[:-1]])
    groups = []
    for start, end in zip(boundaries, np.r_[boundaries[1:], pixels.size], strict=True):
        groups.append((int(pixels[start]), order[start:end]))
    return groups


def depth_render_step(
    records: PixelRecords,
    desired_delta_z_m: np.ndarray,
    *,
    num_gaussians: int,
) -> np.ndarray:
    """把逐像素期望的 z̄ 位移分摊到各高斯。

    Δz_i = Σ_p (w_i / Σw) · Δz̄_p / Σ_p (w_i / Σw)。

    分母是 Jacobi 预条件：同一表面上重叠的高斯会收到多个像素的同一目标，
    按覆盖权重求平均，而不是把每个像素的目标都累加。
    """
    delta = np.asarray(desired_delta_z_m, dtype=np.float64).reshape(-1)
    if records is None:
        raise ContractError("缺少像素记录，不能用相对步更新")
    step = np.zeros(num_gaussians, dtype=np.float64)
    coverage = np.zeros(num_gaussians, dtype=np.float64)
    for pixel, selection in _pixel_groups(records):
        a = records.a[selection]
        transmittance = np.ones_like(a)
        if a.size > 1:
            transmittance[1:] = np.cumprod(1.0 - a[:-1])
        weight = transmittance * a
        total = float(weight.sum())
        if total <= 1e-12:
            continue
        gaussians = records.gaussian_index[selection]
        share = weight / total
        step[gaussians] += share * float(delta[pixel])
        coverage[gaussians] += share
    active = coverage > 1e-12
    step[active] /= coverage[active]
    return step


def depth_render_gradients(
    records: PixelRecords,
    grad_z_render: np.ndarray,
    *,
    num_gaussians: int,
    z_render_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """解析计算 ∂L/∂z_g 与 ∂L/∂α_g。

    L 只通过每像素期望深度 z̄_p 依赖场景。梯度按 front-to-back 合成公式推导，
    在这里顺序视为分段常数（与渲染器一致）。
    """
    if records is None:
        raise ContractError("缺少像素记录，不能计算解析梯度")
    gradient_z = np.zeros(num_gaussians, dtype=np.float64)
    gradient_alpha = np.zeros(num_gaussians, dtype=np.float64)
    grad_flat = np.asarray(grad_z_render, dtype=np.float64).reshape(-1)
    z_flat = np.asarray(z_render_m, dtype=np.float64).reshape(-1)
    for pixel, selection in _pixel_groups(records):
        gradient_value = float(grad_flat[pixel])
        if gradient_value == 0.0 or not np.isfinite(gradient_value):
            continue
        a = records.a[selection]
        z = records.z[selection]
        factor = records.d_a_d_opacity[selection]
        transmittance = np.ones_like(a)
        if a.size > 1:
            transmittance[1:] = np.cumprod(1.0 - a[:-1])
        weight = transmittance * a
        total = float(weight.sum())
        if total <= 1e-12:
            continue
        suffix_s = np.zeros(a.size + 1)
        suffix_a = np.zeros(a.size + 1)
        for position in range(a.size - 1, -1, -1):
            suffix_s[position] = a[position] + (1.0 - a[position]) * suffix_s[position + 1]
            suffix_a[position] = a[position] * z[position] + (1.0 - a[position]) * suffix_a[position + 1]
        z_bar = float(z_flat[pixel])
        if not np.isfinite(z_bar):
            z_bar = float((weight * z).sum() / total)
        d_total = transmittance * (1.0 - suffix_s[1:])
        d_weighted = transmittance * z - transmittance * suffix_a[1:]
        d_zbar_d_alpha = (d_weighted - z_bar * d_total) / total
        gaussians = records.gaussian_index[selection]
        gradient_z[gaussians] += gradient_value * weight / total
        gradient_alpha[gaussians] += gradient_value * d_zbar_d_alpha * factor
    return gradient_z, gradient_alpha
