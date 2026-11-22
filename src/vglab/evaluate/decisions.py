"""实验协议第 9 节的候选决策规则。

输入表是调用方提供的数字。本模块不读取测试结果，也不把假设输入写成实测。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    status: str
    geometry: str
    mean_lpips_improvement: float | None
    per_scene_lpips_improvement: tuple[float, ...]
    note: str


def pair_status(completed_pairs: int, required_pairs: int = 12) -> str:
    if completed_pairs < required_pairs:
        return "inconclusive"
    if completed_pairs == required_pairs:
        return "complete"
    return "inconclusive"


def g1_coverage_ok(baseline: float, candidate: float, max_drop_percentage_points: float) -> bool:
    drop = (baseline - candidate) * 100.0
    return drop <= max_drop_percentage_points + 1e-12


def classify_h2(
    scenes: list[dict],
    *,
    g0: str,
    max_coverage_drop_pp: float = 2.0,
    seed_std: float | None = None,
) -> Decision:
    """S2 相对 S1。LPIPS 下降记为改善。"""
    parsed = _parse_scenes(scenes, max_coverage_drop_pp)
    if parsed is None:
        return Decision("inconclusive", "not_evaluated", None, (), "需要 4 个场景的完整配对")
    improvements, psnr_drop, g1_flags, f_drop, geometry_complete = parsed
    mean_lpips = sum(improvements) / len(improvements)
    if seed_std is not None and seed_std > mean_lpips:
        return Decision("inconclusive", "not_evaluated", mean_lpips, improvements, "seed 波动大于候选增益")
    if g0 == "available" and not geometry_complete:
        return Decision("inconclusive", "not_evaluated", mean_lpips, improvements, "G0 可用时四个场景都要有 F-score")
    geometry = _geometry_status(g0, f_drop, max_drop=0.02)
    if not all(g1_flags):
        return Decision("g1_failed", geometry, mean_lpips, improvements, "覆盖下降超过候选门槛")
    positive = sum(item > 0 for item in improvements)
    psnr_ok = psnr_drop <= 0.3 + 1e-12
    quality = positive >= 3 and mean_lpips >= 0.01 - 1e-12 and psnr_ok and geometry != "degraded"
    status = "advance_candidate" if quality else "no_clear_gain"
    return Decision(status, geometry, mean_lpips, improvements, "候选规则，不是实测结论")


def classify_h4(
    scenes: list[dict],
    *,
    g0: str,
    cost_supports: bool | None,
    max_coverage_drop_pp: float = 2.0,
) -> Decision:
    parsed = _parse_scenes(scenes, max_coverage_drop_pp)
    if parsed is None or cost_supports is None:
        return Decision("inconclusive", "not_evaluated", None, (), "成本或完整场景配对尚未测得")
    improvements, _, g1_flags, f_drop, geometry_complete = parsed
    mean_lpips = sum(improvements) / len(improvements)
    if g0 == "available" and not geometry_complete:
        return Decision("inconclusive", "not_evaluated", mean_lpips, improvements, "G0 可用时四个场景都要有 F-score")
    geometry = _geometry_status(g0, f_drop, max_drop=0.02)
    if not all(g1_flags):
        return Decision("g1_failed", geometry, mean_lpips, improvements, "覆盖下降超过候选门槛")
    if not cost_supports:
        return Decision("no_clear_gain", geometry, mean_lpips, improvements, "等时间成本未支持额外复杂度")
    if mean_lpips >= 0.01 - 1e-12 and any(item > 0 for item in improvements) and geometry != "degraded":
        return Decision("advance_candidate", geometry, mean_lpips, improvements, "候选规则，不是实测结论")
    return Decision("no_clear_gain", geometry, mean_lpips, improvements, "感知增益未达到候选门槛")


def _geometry_status(g0: str, f_drop: float | None, *, max_drop: float) -> str:
    if g0 == "na":
        return "not_evaluated"
    if g0 != "available" or f_drop is None:
        return "not_evaluated"
    if f_drop > max_drop + 1e-12:
        return "degraded"
    return "no_degradation"


def _parse_scenes(
    scenes: list[dict],
    max_coverage_drop_pp: float,
) -> tuple[tuple[float, ...], float, list[bool], float | None, bool] | None:
    if len(scenes) != 4:
        return None
    improvements = []
    psnr_drops = []
    coverage_flags = []
    f_drops = []
    for scene in scenes:
        improvements.append(float(scene["lpips_baseline"] - scene["lpips_candidate"]))
        psnr_drops.append(float(scene["psnr_baseline"] - scene["psnr_candidate"]))
        coverage_flags.append(
            g1_coverage_ok(scene["coverage_baseline"], scene["coverage_candidate"], max_coverage_drop_pp)
        )
        has_baseline = "f_score_baseline" in scene
        has_candidate = "f_score_candidate" in scene
        if has_baseline and has_candidate:
            f_drops.append(float(scene["f_score_baseline"] - scene["f_score_candidate"]))
        else:
            f_drops.append(None)
    geometry_complete = all(item is not None for item in f_drops)
    numeric = [float(item) for item in f_drops if item is not None]
    f_drop = sum(numeric) / len(numeric) if geometry_complete else None
    return tuple(improvements), sum(psnr_drops) / len(psnr_drops), coverage_flags, f_drop, geometry_complete
