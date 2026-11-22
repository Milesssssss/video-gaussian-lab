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
