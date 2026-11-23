"""开发期候选数值。frozen=False，不能当作协议冻结值或实测结果。"""

UNFROZEN_ORDINARY_GAUSSIAN = {
    "name": "pixel_footprint_v0",
    "frozen": False,
    "opacity": 0.8,
    "footprint_px": 1.0,
    "min_scale_m": 1.0e-4,
    "normal_rule": "camera_axes",
}

UNFROZEN_SCALE_FIT = {
    "name": "log_median_v0",
    "frozen": False,
    "min_samples": 32,
    "s_min": 0.25,
    "s_max": 4.0,
    "outlier_log_abs": 0.4054651081081644,
}

UNFROZEN_VISIBILITY = {
    "name": "center_tolerance_v0",
    "frozen": False,
    "tau_abs_m": 0.02,
    "tau_rel": 0.01,
    "alpha_min": 0.1,
}

UNFROZEN_DEPTH_TEACHER = {
    "name": "heuristic_reliability_v0",
    "frozen": False,
    "edge_k": 8.0,
    "range_near_m": 0.5,
    "range_far_m": 10.0,
    "range_rolloff_m": 2.0,
    "range_floor": 0.2,
    "support_min_views": 1,
    "q_unsupported": 0.6,
    "temporal_tau": 0.15,
    "weights": {"edge": 0.40, "range": 0.20, "support": 0.25, "temporal": 0.15},
    "visibility": {"tau_abs_m": 0.02, "tau_rel": 0.01},
    "conflict_excludes": True,
    "q_floor": 0.05,
}

UNFROZEN_DEPTH_REFINER = {
    "name": "log_step_optical_axis_v0",
    "frozen": False,
    "lr_log_step": 0.35,
    "log_step_cap": 0.5,
    "alpha_min": 0.1,
    "max_steps": 25,
    "min_coverage": 0.5,
    "patience": 5,
    "update_opacity": False,
    "lr_opacity": 0.02,
    "opacity_logit_clip": 12.0,
}
