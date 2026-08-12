from __future__ import annotations


TEX_PT_PER_IN = 72.27
PDF_PT_PER_IN = 72.0

TEXTWIDTH_IN = 516.0 / TEX_PT_PER_IN
COLUMNWIDTH_IN = 252.0 / TEX_PT_PER_IN

PAPER_FIGURE_SIZES_IN = {
    "transient_prediction_task": (COLUMNWIDTH_IN, 2.000),
    "model_overview": (COLUMNWIDTH_IN, 3.220),
    "data_processing_pipeline": (2.335, 2.855),
    "three_stage_training": (COLUMNWIDTH_IN, 2.320),
    "update_regimes": (COLUMNWIDTH_IN, 1.803),
    "rategru_tanh_update_schematic": (COLUMNWIDTH_IN, 1.633),
    "limiter_function_comparison": (COLUMNWIDTH_IN, 0.55 * COLUMNWIDTH_IN),
    "all_lr12_lr3_heatmaps": (TEXTWIDTH_IN, 3.115),
    "no_h_initial_sync_comparison": (COLUMNWIDTH_IN, 1.950),
    "n87_sampling_upsampling_overview": (TEXTWIDTH_IN, 1.704),
    "n87_sampling_downsampling_low2": (COLUMNWIDTH_IN, 2.548),
    "shortest_upsample_three_model_waveforms_1_2_4_8": (TEXTWIDTH_IN, 2.568),
    "uniform_waveforms_16_32": (COLUMNWIDTH_IN, 2.250),
    "downsample_32x_input_distortion_zoom": (COLUMNWIDTH_IN, 2.050),
    "random_interval_representative_zoom_points_1to10_window750_meanlike": (TEXTWIDTH_IN, 1.869),
    "update_coeff_sampling_diagnostics_combined": (TEXTWIDTH_IN, 3.850),
}

ORIGINAL_FIGURE_WIDTHS_IN = {
    # Laid out directly at the printed size, so no geometry rescaling applies.
    "transient_prediction_task": COLUMNWIDTH_IN,
    "model_overview": COLUMNWIDTH_IN,
    "data_processing_pipeline": 2.335,
    "three_stage_training": COLUMNWIDTH_IN,
    "update_regimes": COLUMNWIDTH_IN,
    "rategru_tanh_update_schematic": 5.45,
    "limiter_function_comparison": 3.45,
    "all_lr12_lr3_heatmaps": 29.639,
    "no_h_initial_sync_comparison": 7.2,
    "n87_sampling_upsampling_overview": 15.6,
    "n87_sampling_downsampling_low2": 7.2,
    "shortest_upsample_three_model_waveforms_1_2_4_8": 22.0,
    "uniform_waveforms_16_32": 12.0,
    "downsample_32x_input_distortion_zoom": 14.4,
    "random_interval_representative_zoom_points_1to10_window750_meanlike": 12.8,
    "update_coeff_sampling_diagnostics_combined": 14.4,
}


def figure_size(stem: str) -> tuple[float, float]:
    return PAPER_FIGURE_SIZES_IN[stem]


def figure_size_points(stem: str) -> tuple[int, int]:
    width, height = figure_size(stem)
    return round(width * PDF_PT_PER_IN), round(height * PDF_PT_PER_IN)


def figure_scale(stem: str) -> float:
    width, _ = figure_size(stem)
    return width / ORIGINAL_FIGURE_WIDTHS_IN[stem]


def line_width(stem: str, value: float) -> float:
    return value * figure_scale(stem)


def marker_size(stem: str, value: float) -> float:
    return value * figure_scale(stem)


def marker_area(stem: str, value: float) -> float:
    scale = figure_scale(stem)
    return value * scale * scale
