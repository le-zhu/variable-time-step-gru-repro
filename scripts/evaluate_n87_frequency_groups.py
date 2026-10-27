from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_style import paper_matplotlib_rc

from .evaluate_n87_longest_downsample import (
    evaluate_batch as evaluate_downsample_batch,
    make_downsampled_batch,
)
from .evaluate_n87_longest_upsample import (
    ROOT,
    evaluate_batch_streaming,
    evaluate_equal_length_batch_streaming,
    load_model,
    load_origin_test,
    make_upsampled_batch,
    resolve_device,
)


PAPER_FIGURE_DIR = ROOT / "figures" / "output"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "n87_frequency_groups"
# The released N87 checkpoints, and the processed split the reader supplies.
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"
SAMPLE_RATE_HZ = 16_000_000.0
SUBCYCLES_PER_SEQUENCE = 100.0
TRAIN_RELATIVE_DT_MIN = 1.0
TRAIN_RELATIVE_DT_MAX = 9.0
MAGNETX_NOMINAL_FREQUENCY_KHZ = {
    2016: 800.0,
    3216: 500.0,
    5008: 320.0,
    8015: 200.0,
    12816: 125.0,
    20016: 80.0,
    32016: 50.0,
}

MODEL_SPECS = {
    "dtscaledgru": {
        "title": "DtScaledGRU on N87 frequency-group diagnostics",
        "paper_stem": "dtscaledgru_longest_sampling_interval_error",
        "run_dir": ROOT / "checkpoints" / "n87_paper_models" / "dtscaledgru",
        "data_dir": DEFAULT_DATA_DIR,
        "ymax": 50.0,
    },
    "rategru": {
        "title": "RateGRU on N87 frequency-group diagnostics",
        "paper_stem": "rategru_longest_sampling_interval_error",
        "run_dir": ROOT / "checkpoints" / "n87_paper_models" / "rategru",
        "data_dir": DEFAULT_DATA_DIR,
        "ymax": 50.0,
    },
    # DtInputGRU enters only the revised low-frequency downsampling figure, so
    # only those cases are evaluated for it.
    "dtinputgru": {
        "title": "DtInputGRU on N87 frequency-group diagnostics",
        "paper_stem": "dtinputgru_longest_sampling_interval_error",
        "run_dir": ROOT / "checkpoints" / "n87_paper_models" / "dtinputgru",
        "data_dir": DEFAULT_DATA_DIR,
        "ymax": 50.0,
        "low2_downsampling_only": True,
    },
}

MODEL_TITLES = {
    "dtscaledgru": "DtScaledGRU",
    "rategru": "RateGRU",
    "dtinputgru": "DtInputGRU",
}
PAPER_MODEL_LABELS = {
    "SoftplusRateExpCapGRU": "RateGRU",
    "DtInputFixedGRU": "DtInputGRU",
}
DEFAULT_MODELS = ["dtscaledgru", "rategru", "dtinputgru"]
# Panels of the upsampling overview; DtInputGRU is not evaluated for it.
UPSAMPLING_OVERVIEW_MODELS = ["dtscaledgru", "rategru"]
LOW2_STEM = "n87_sampling_downsampling_low2_with_dtinput"
LOW2_SIZE_KEY = "n87_sampling_downsampling_low2"
LOW2_YMAX = 40.0
# Factors drawn in the manuscript figure; every evaluated factor stays in the CSV.
LOW2_PLOT_FACTORS = [1, 2, 4, 8, 14, 16, 18, 22, 28, 32]
# Draw order and legend order of the revised figure.
LOW2_MODEL_ORDER = ["dtscaledgru", "rategru", "dtinputgru"]
LOW2_LEGEND_ORDER = ["dtscaledgru", "dtinputgru", "rategru"]


def paper_model_label(model_key: str) -> str:
    return PAPER_MODEL_LABELS.get(str(model_key), str(model_key))


def combined_figure_size(models: list[str]) -> tuple[float, float]:
    if len(models) == 3:
        return figure_size("n87_sampling_upsampling_overview")
    return (round(figure_size("n87_sampling_upsampling_overview")[0] * max(len(models), 1) / 3.0, 10), 1.704)


def low2_downsampling_figure_size() -> tuple[float, float]:
    return figure_size("n87_sampling_downsampling_low2")


TRAIN_DOWNSAMPLE_FACTOR_BY_LENGTH = {
    2016: 1,
    3216: 1,
    5008: 2,
    8015: 3,
    12816: 4,
    20016: 6,
    32016: 9,
}
LOG_TICKS = [
    (1.0 / 32.0, "1/32"),
    (1.0 / 16.0, "1/16"),
    (1.0 / 8.0, "1/8"),
    (1.0 / 4.0, "1/4"),
    (1.0 / 2.0, "1/2"),
    (1.0, "1"),
    (2.0, "2"),
    (4.0, "4"),
    (8.0, "8"),
    (16.0, "16"),
    (32.0, "32"),
]
HIGH_UPSAMPLE_FACTORS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32]
LOW_UPSAMPLE_FACTORS = [2, 4, 6, 8]
DOWNSAMPLE_PLOT_FACTORS = [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32]
DOWNSAMPLE_FACTORS = DOWNSAMPLE_PLOT_FACTORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frequency-group sampling diagnostics and draw "
            "n87_sampling_downsampling_low2_with_dtinput (DtScaledGRU, RateGRU, DtInputGRU)."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Processed N87 split (std_db_dt) shared by all models; defaults to data/N87_processed.",
    )
    parser.add_argument(
        "--low2-only",
        action="store_true",
        help="Evaluate only the 80/50 kHz downsampling cases that the manuscript figure draws.",
    )
    parser.add_argument("--low2-plot-factors", type=int, nargs="+", default=LOW2_PLOT_FACTORS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--high-upsample-factors", type=int, nargs="+", default=HIGH_UPSAMPLE_FACTORS)
    parser.add_argument("--low-upsample-factors", type=int, nargs="+", default=LOW_UPSAMPLE_FACTORS)
    parser.add_argument("--downsample-factors", type=int, nargs="+", default=DOWNSAMPLE_FACTORS)
    parser.add_argument("--plot-downsample-factors", type=int, nargs="+", default=DOWNSAMPLE_PLOT_FACTORS)
    parser.add_argument("--dtscaledgru-run-dir", type=Path, default=None)
    parser.add_argument("--dtscaledgru-data-dir", type=Path, default=None)
    parser.add_argument("--rategru-run-dir", type=Path, default=None)
    parser.add_argument("--rategru-data-dir", type=Path, default=None)
    parser.add_argument("--dtinputgru-run-dir", type=Path, default=None)
    parser.add_argument("--dtinputgru-data-dir", type=Path, default=None)
    parser.add_argument("--no-low-upsampling", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--copy-to-paper", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_source_paths(model_name: str, args: argparse.Namespace) -> tuple[dict[str, object], Path, Path]:
    """Resolve the checkpoint and processed-data directories for one model.

    The released checkpoints are the default. Point ``--<model>-run-dir`` at a
    locally retrained run directory to evaluate that instead.
    """
    spec = MODEL_SPECS[model_name]
    run_dir = Path(str(spec["run_dir"]))
    data_dir = Path(str(spec["data_dir"]))
    override_run = getattr(args, f"{model_name}_run_dir", None)
    override_data = getattr(args, f"{model_name}_data_dir", None)
    if override_data is None:
        override_data = getattr(args, "data_dir", None)
    if override_run is not None:
        run_dir = resolve(override_run)
    if override_data is not None:
        data_dir = resolve(override_data)
    return {}, run_dir, data_dir


def frequency_khz(length: int) -> float:
    return MAGNETX_NOMINAL_FREQUENCY_KHZ.get(
        int(length),
        SUBCYCLES_PER_SEQUENCE * SAMPLE_RATE_HZ / float(length) / 1000.0,
    )


def frequency_label(length: int) -> str:
    value = frequency_khz(length)
    return f"{value:g} kHz"


def load_existing_rows(path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str, int, str, int]]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (
            row["model"],
            row["frequency_group"],
            int(row["selected_length"]),
            row["sampling"],
            int(row["factor"]),
        )
        for row in rows
    }
    return rows, keys


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "model_label",
        "frequency_group",
        "selected_length",
        "nominal_frequency_khz",
        "sampling",
        "factor",
        "x_value",
        "selected_row_count",
        "effective_length_min",
        "effective_length_max",
        "warmup_steps",
        "dt_eff_min",
        "dt_eff_max",
        "mean_err",
        "p95_err",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fieldnames}
            for field in ["nominal_frequency_khz", "x_value", "dt_eff_min", "dt_eff_max", "mean_err", "p95_err"]:
                if out[field] != "":
                    out[field] = f"{float(out[field]):.10g}"
            writer.writerow(out)


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    int_fields = ["selected_length", "factor", "selected_row_count", "effective_length_min", "effective_length_max", "warmup_steps"]
    float_fields = ["nominal_frequency_khz", "x_value", "dt_eff_min", "dt_eff_max", "mean_err", "p95_err"]
    out_rows = []
    for row in rows:
        item = dict(row)
        for field in int_fields:
            item[field] = int(item[field])
        for field in float_fields:
            item[field] = float(item[field])
        item["nominal_frequency_khz"] = frequency_khz(int(item["selected_length"]))
        item["frequency_group"] = frequency_label(int(item["selected_length"]))
        out_rows.append(item)
    return out_rows


def select_lengths(lengths: np.ndarray) -> tuple[list[int], list[int]]:
    unique_lengths = sorted(int(value) for value in np.unique(lengths))
    return unique_lengths[:3], unique_lengths[-4:]


def evaluate_case(
    *,
    model,
    model_name: str,
    model_label: str,
    data: np.ndarray,
    lengths: np.ndarray,
    deltas: np.ndarray,
    selected_length: int,
    sampling: str,
    factor: int,
    base_warmup: int,
    device: torch.device,
) -> dict[str, object]:
    selected_indices = np.flatnonzero(lengths == selected_length).astype(int).tolist()
    if sampling == "upsample":
        batch_data, batch_lengths, batch_deltas = make_upsampled_batch(data, lengths, deltas, selected_indices, factor)
        warmup_steps = int(math.ceil(base_warmup * factor))
        x_value = 1.0 / float(factor)
    elif sampling == "downsample":
        batch_data, batch_lengths, batch_deltas = make_downsampled_batch(
            data, lengths, deltas, selected_indices, factor
        )
        warmup_steps = int(math.ceil(base_warmup / factor))
        x_value = float(factor)
    else:
        raise ValueError(f"Unsupported sampling mode: {sampling}")

    if sampling == "downsample":
        row_err, _pred, _target = evaluate_downsample_batch(
            model,
            batch_data,
            batch_lengths,
            batch_deltas,
            base_warmup,
            factor,
            device,
        )
    else:
        evaluator = (
            evaluate_equal_length_batch_streaming
            if int(batch_lengths.min()) == int(batch_lengths.max())
            else evaluate_batch_streaming
        )
        if evaluator is evaluate_equal_length_batch_streaming:
            row_err, _ = evaluator(model, batch_data, batch_deltas, warmup_steps, device)
        else:
            row_err, _ = evaluator(
                model,
                batch_data,
                batch_lengths,
                batch_deltas,
                warmup_steps,
                device,
            )
    valid = row_err[np.isfinite(row_err)]
    mean_err = float(np.mean(valid)) if valid.size else float("nan")
    p95_err = float(np.percentile(valid, 95)) if valid.size else float("nan")
    row = {
        "model": model_name,
        "model_label": model_label,
        "frequency_group": frequency_label(selected_length),
        "selected_length": int(selected_length),
        "nominal_frequency_khz": frequency_khz(selected_length),
        "sampling": sampling,
        "factor": int(factor),
        "x_value": x_value,
        "selected_row_count": len(selected_indices),
        "effective_length_min": int(batch_lengths.min()),
        "effective_length_max": int(batch_lengths.max()),
        "warmup_steps": warmup_steps,
        "dt_eff_min": float(batch_deltas.min()),
        "dt_eff_max": float(batch_deltas.max()),
        "mean_err": mean_err,
        "p95_err": p95_err,
    }
    del batch_data, batch_lengths, batch_deltas
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def write_case_metadata(
    *,
    output_dir: Path,
    model_name: str,
    run_dir: Path,
    model_path: Path,
    data_dir: Path,
    run_config: dict[str, object],
    selected_length: int,
    sampling: str,
    factors: list[int],
    selected_indices: list[int],
    base_warmup: int,
    device: torch.device,
) -> Path:
    metadata_path = output_dir / f"metadata_{model_name}_{int(selected_length)}_{sampling}.json"
    payload = {
        "model": model_name,
        "model_label": paper_model_label(run_config["model_key"]),
        "run_dir": str(run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "model_key": run_config["model_key"],
        "run_name": run_config["run_name"],
        "data_name": run_config["data_name"],
        "dt_norm_mode": run_config["dt_norm_mode"],
        "batch_size": run_config["train"]["batch_size"],
        "lr_12": run_config["train"]["lr_12"],
        "lr_3": run_config["train"]["lr_3"],
        "base_eval_warmup_len": int(base_warmup),
        "selected_length": int(selected_length),
        "frequency_group": frequency_label(selected_length),
        "sampling": sampling,
        "selected_row_count": len(selected_indices),
        "selected_row_indices": [int(idx) for idx in selected_indices],
        "factors": [int(factor) for factor in factors],
        "device": str(device),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return metadata_path


def row_sort_key(row: dict[str, object]) -> tuple[str, float, float, str]:
    return (str(row["model"]), -float(row["nominal_frequency_khz"]), float(row["x_value"]), str(row["sampling"]))


def line_style_for_frequency(freq_khz: float) -> tuple[str, str]:
    high_freqs = [800.0, 500.0, 320.0]
    low_freqs = [200.0, 125.0, 80.0, 50.0]
    palette = {
        800.0: ("#1f77b4", "-"),
        500.0: ("#ff7f0e", "-"),
        320.0: ("#8c564b", "-"),
        200.0: ("#FFD400", "-"),
        125.0: ("#9467bd", "-"),
        80.0: ("#2ca02c", "-"),
        50.0: ("#d62728", "-"),
    }
    nearest = min(high_freqs + low_freqs, key=lambda value: abs(value - freq_khz))
    return palette[nearest]


def line_zorder_for_frequency(freq_khz: float) -> float:
    if freq_khz >= 300.0:
        return 3.0 + freq_khz / 1000.0
    low_frequency_order = {
        50.0: 0.4,
        80.0: 0.3,
        125.0: 0.2,
        200.0: 0.1,
    }
    nearest = min(low_frequency_order, key=lambda freq: abs(freq - freq_khz))
    return 2.0 + low_frequency_order[nearest]


def legend_sort_key(label: str) -> tuple[int, float]:
    if label == "Training dt range":
        return (0, 0.0)
    try:
        value = float(label.split()[0])
    except (ValueError, IndexError):
        return (2, 0.0)
    if value >= 300.0:
        return (1, -value)
    low_frequency_order = {
        125.0: 0.0,
        200.0: 1.0,
        80.0: 2.0,
        50.0: 3.0,
    }
    nearest = min(low_frequency_order, key=lambda freq: abs(freq - value))
    return (2, low_frequency_order[nearest])


def configured_downsample_factors(max_factor: int, candidate_factors: list[int]) -> list[int]:
    return [int(factor) for factor in candidate_factors if int(factor) <= int(max_factor)]


def is_upsampling_train_range_row(row: dict[str, object], plot_downsample_factors: set[int]) -> bool:
    length = int(row["selected_length"])
    sampling = str(row["sampling"])
    factor = int(row["factor"])
    if sampling == "upsample":
        return True
    if sampling != "downsample":
        return False
    train_factor = TRAIN_DOWNSAMPLE_FACTOR_BY_LENGTH.get(length, 1)
    return factor in plot_downsample_factors and factor <= train_factor


def is_low2_downsampling_row(row: dict[str, object], plot_downsample_factors: set[int]) -> bool:
    length = int(row["selected_length"])
    return length in {20016, 32016} and str(row["sampling"]) == "downsample" and int(row["factor"]) in plot_downsample_factors


def plot_group_rows(
    ax,
    model_rows: list[dict[str, object]],
    row_filter,
    ymax: float,
    lengths: list[int],
    stem: str,
) -> None:
    for length in sorted(lengths, key=frequency_khz):
        group_rows = sorted(
            [row for row in model_rows if int(row["selected_length"]) == length and row_filter(row)],
            key=lambda row: float(row["x_value"]),
        )
        if not group_rows:
            continue
        freq = frequency_khz(length)
        color, linestyle = line_style_for_frequency(freq)
        marker = "o" if freq >= 300.0 else "s"
        ax.plot(
            [float(row["x_value"]) for row in group_rows],
            [
                float(row["mean_err"]) * 100.0 if float(row["mean_err"]) * 100.0 <= float(ymax) else float("nan")
                for row in group_rows
            ],
            marker=marker,
            linewidth=line_width(stem, 1.05),
            markersize=marker_size(stem, 2.2),
            linestyle=linestyle,
            color=color,
            label=frequency_label(length),
            zorder=line_zorder_for_frequency(freq),
        )


def style_sampling_axis(
    ax,
    *,
    title: str,
    ymax: float,
    ytick_step: int,
    xlim: tuple[float, float],
    xticks: list[tuple[float, str]],
    show_ylabel: bool,
    show_xlabel: bool,
    stem: str,
    show_unit_line: bool = True,
) -> None:
    if show_unit_line:
        ax.axvline(1.0, color="#999999", linestyle=":", linewidth=line_width(stem, 1.0))
    ax.set_xscale("log", base=2)
    ax.set_xlim(*xlim)
    ax.set_xticks([item[0] for item in xticks])
    ax.set_xticklabels([item[1] for item in xticks])
    ax.minorticks_off()
    ax.set_ylim(0.0, float(ymax))
    ax.set_yticks(range(0, int(float(ymax)) + 1, ytick_step))
    if show_xlabel:
        ax.set_xlabel("Relative time-step interval", fontsize=8)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Mean sequence error (%)", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=line_width(stem, 0.8), length=marker_size(stem, 3.5))
    ax.grid(True, color="#d9d9d9", linewidth=line_width(stem, 0.8))
    for spine in ax.spines.values():
        spine.set_linewidth(line_width(stem, 0.8))
    ax.margins(x=0.04)


def save_combined_figure(
    output_dir: Path,
    rows: list[dict[str, object]],
    models: list[str],
    copy_to_paper: bool,
    *,
    stem: str,
    row_filter,
    ymax: float,
    ytick_step: int,
    xlim: tuple[float, float],
    xticks: list[tuple[float, str]],
    lengths: list[int],
) -> None:
    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        fig, axes = plt.subplots(1, len(models), figsize=combined_figure_size(models), dpi=180, sharey=True)
        axes = np.atleast_1d(axes)
        for idx, (ax, model_name) in enumerate(zip(axes, models)):
            model_rows = [row for row in rows if row["model"] == model_name]
            style_sampling_axis(
                ax,
                title=MODEL_TITLES.get(model_name, model_name),
                ymax=ymax,
                ytick_step=ytick_step,
                xlim=xlim,
                xticks=xticks,
                show_ylabel=idx == 0,
                show_xlabel=idx == len(models) // 2,
                stem=stem,
            )
            plot_group_rows(ax, model_rows, row_filter, ymax, lengths, stem)
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                order = sorted(range(len(labels)), key=lambda label_idx: legend_sort_key(labels[label_idx]))
                ordered_handles = [handles[label_idx] for label_idx in order]
                ordered_labels = [labels[label_idx] for label_idx in order]
                if stem == "n87_sampling_upsampling_overview":
                    by_label = dict(zip(ordered_labels, ordered_handles))
                    legend_labels = ["800 kHz", "500 kHz", "320 kHz", " ", "200 kHz", "125 kHz", "80 kHz", "50 kHz"]
                    ordered_handles = [
                        by_label[label] if label in by_label else plt.Line2D([0], [0], color="none", linewidth=0)
                        for label in legend_labels
                    ]
                    ordered_labels = legend_labels
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    frameon=False,
                    fontsize=8.0,
                    ncol=2,
                    loc="upper right",
                    handlelength=0.75,
                    handletextpad=0.25,
                    columnspacing=0.45,
                    labelspacing=0.15,
                    borderpad=0.0,
                    borderaxespad=0.15,
                )
        fig.tight_layout(pad=0.15, w_pad=0.25)
        for suffix in ("png", "pdf", "svg"):
            out_path = output_dir / f"{stem}.{suffix}"
            fig.savefig(out_path)
            if copy_to_paper and suffix == "pdf":
                PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_path, PAPER_FIGURE_DIR / out_path.name)
        plt.close(fig)


def save_low2_downsampling_overlay(
    output_dir: Path,
    rows: list[dict[str, object]],
    models: list[str],
    copy_to_paper: bool,
    *,
    plot_downsample_factors: set[int],
) -> None:
    """Draw the revised Fig. 8: 80/50 kHz downsampling for all three models.

    Values above the y-limit are kept and clipped by the axis, so a curve that
    leaves the plot still shows where it crosses the upper edge.
    """
    model_colors = {
        ("dtscaledgru", 80.0): "#2ca02c",
        ("dtscaledgru", 50.0): "#d62728",
        ("rategru", 80.0): "#1f77b4",
        ("rategru", 50.0): "#FFD400",
        ("dtinputgru", 80.0): "#9467bd",
        ("dtinputgru", 50.0): "#ff7f0e",
    }
    model_markers = {"dtscaledgru": "s", "rategru": "o", "dtinputgru": "^"}
    model_zorder = {"dtscaledgru": 0.0, "rategru": 0.2, "dtinputgru": 0.4}
    downsample_ticks = [item for item in LOG_TICKS if 1.0 <= item[0] <= 36.0]
    stem = LOW2_SIZE_KEY
    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        fig, ax = plt.subplots(figsize=low2_downsampling_figure_size(), dpi=180)
        style_sampling_axis(
            ax,
            title="N87 low-frequency downsampling diagnostics",
            ymax=LOW2_YMAX,
            ytick_step=10,
            xlim=(0.9, 36.0),
            xticks=downsample_ticks,
            show_ylabel=True,
            show_xlabel=True,
            stem=stem,
            show_unit_line=False,
        )
        handles_by_key = {}
        draw_models = [name for name in LOW2_MODEL_ORDER if name in models]
        draw_models += [name for name in models if name not in draw_models]
        for model_name in draw_models:
            for length in [20016, 32016]:
                freq = frequency_khz(length)
                group_rows = sorted(
                    [
                        row
                        for row in rows
                        if row["model"] == model_name
                        and int(row["selected_length"]) == length
                        and is_low2_downsampling_row(row, plot_downsample_factors)
                    ],
                    key=lambda row: float(row["x_value"]),
                )
                if not group_rows:
                    continue
                (line,) = ax.plot(
                    [float(row["x_value"]) for row in group_rows],
                    [float(row["mean_err"]) * 100.0 for row in group_rows],
                    marker=model_markers.get(model_name, "o"),
                    linewidth=line_width(stem, 1.05),
                    markersize=marker_size(stem, 2.2),
                    linestyle="-",
                    color=model_colors.get((model_name, freq)),
                    label=f"{MODEL_TITLES.get(model_name, model_name)} {frequency_label(length)}",
                    zorder=2.0 + model_zorder.get(model_name, 0.0) + (0.1 if length == 32016 else 0.0),
                )
                handles_by_key[(model_name, length)] = line
        legend_models = [name for name in LOW2_LEGEND_ORDER if name in draw_models]
        legend_models += [name for name in draw_models if name not in legend_models]
        legend_handles = [
            handles_by_key[(name, length)]
            for name in legend_models
            for length in [20016, 32016]
            if (name, length) in handles_by_key
        ]
        ax.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            frameon=False,
            fontsize=7.2,
            ncol=1,
            loc="upper left",
            handlelength=0.7,
            handletextpad=0.22,
            labelspacing=0.1,
            columnspacing=0.6,
            borderpad=0.0,
            borderaxespad=0.12,
        )
        fig.tight_layout(pad=0.18)
        for suffix in ("png", "pdf", "svg"):
            out_path = output_dir / f"{LOW2_STEM}.{suffix}"
            fig.savefig(out_path)
            if copy_to_paper and suffix == "pdf":
                PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_path, PAPER_FIGURE_DIR / out_path.name)
        plt.close(fig)


def save_plots(
    output_dir: Path,
    rows: list[dict[str, object]],
    models: list[str],
    copy_to_paper: bool,
    *,
    plot_downsample_factors: set[int],
    low2_plot_factors: set[int],
) -> None:
    upsample_ticks = [item for item in LOG_TICKS if 1.0 / 36.0 <= item[0] <= 10.0]
    downsample_ticks = [item for item in LOG_TICKS if 1.0 <= item[0] <= 36.0]
    all_lengths = sorted({int(row["selected_length"]) for row in rows}, key=frequency_khz)
    # The upsampling overview is a working figure that the revised manuscript
    # does not show; draw it only for models that have upsampling rows.
    upsampled_models = {str(row["model"]) for row in rows if str(row["sampling"]) == "upsample"}
    overview_models = [name for name in models if name in UPSAMPLING_OVERVIEW_MODELS and name in upsampled_models]
    if overview_models:
        save_combined_figure(
            output_dir,
            rows,
            overview_models,
            False,
            stem="n87_sampling_upsampling_overview",
            row_filter=lambda row: is_upsampling_train_range_row(row, plot_downsample_factors),
            ymax=20.0,
            ytick_step=5,
            xlim=(1.0 / 36.0, 10.0),
            xticks=upsample_ticks,
            lengths=all_lengths,
        )
    save_low2_downsampling_overlay(
        output_dir,
        rows,
        models,
        copy_to_paper,
        plot_downsample_factors=low2_plot_factors,
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and args.overwrite:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads is not None:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(int(args.torch_threads))
        except RuntimeError:
            pass

    summary_path = output_dir / "sampling_error_summary.csv"
    rows, done_keys = load_existing_rows(summary_path)
    rows = normalize_rows(rows) if rows else []

    if not args.plot_only:
        device = resolve_device(args.device)
        model_source_metadata: dict[str, dict[str, object]] = {}
        case_metadata_paths: list[str] = []
        for model_name in args.models:
            source_metadata, run_dir, data_dir = model_source_paths(model_name, args)
            model_source_metadata[model_name] = {
                "effective_run_dir": str(run_dir.resolve()),
                "effective_data_dir": str(data_dir.resolve()),
            }
            model, run_config, model_path = load_model(run_dir, device)
            data, lengths, deltas, _norm_stats = load_origin_test(data_dir)
            high_lengths, low_lengths = select_lengths(lengths)
            unique_lengths = sorted(int(value) for value in np.unique(lengths))
            base_warmup = int(run_config["train"]["eval_warmup_len"])
            model_label = paper_model_label(run_config["model_key"])
            low2_only = bool(args.low2_only or MODEL_SPECS[model_name].get("low2_downsampling_only", False))

            cases: list[tuple[int, str, int]] = []
            downsample_factors = sorted(dict.fromkeys(int(factor) for factor in args.downsample_factors))
            if not low2_only:
                for length in high_lengths:
                    cases.extend((length, "upsample", int(factor)) for factor in sorted(dict.fromkeys(args.high_upsample_factors)))
                if not args.no_low_upsampling:
                    for length in low_lengths:
                        cases.extend((length, "upsample", int(factor)) for factor in sorted(dict.fromkeys(args.low_upsample_factors)))
                for length in unique_lengths:
                    train_factor = TRAIN_DOWNSAMPLE_FACTOR_BY_LENGTH.get(int(length), 1)
                    cases.extend(
                        (length, "downsample", int(factor))
                        for factor in configured_downsample_factors(train_factor, downsample_factors)
                    )
            for length in sorted(unique_lengths[-2:]):
                cases.extend((length, "downsample", int(factor)) for factor in downsample_factors)
            cases = sorted(set(cases), key=lambda item: (item[0], item[1], item[2]))

            for selected_length, sampling in sorted({(item[0], item[1]) for item in cases}):
                group_factors = sorted({int(item[2]) for item in cases if item[0] == selected_length and item[1] == sampling})
                selected_indices = np.flatnonzero(lengths == selected_length).astype(int).tolist()
                case_metadata_paths.append(
                    str(
                        write_case_metadata(
                            output_dir=output_dir,
                            model_name=model_name,
                            run_dir=run_dir,
                            model_path=model_path,
                            data_dir=data_dir,
                            run_config=run_config,
                            selected_length=int(selected_length),
                            sampling=sampling,
                            factors=group_factors,
                            selected_indices=selected_indices,
                            base_warmup=base_warmup,
                            device=device,
                        ).resolve()
                    )
                )

            for selected_length, sampling, factor in cases:
                key = (model_name, frequency_label(selected_length), int(selected_length), sampling, int(factor))
                if key in done_keys:
                    print(f"[sampling] skip {model_name} {frequency_label(selected_length)} {sampling} factor={factor}", flush=True)
                    continue
                print(f"[sampling] eval {model_name} {frequency_label(selected_length)} {sampling} factor={factor}", flush=True)
                row = evaluate_case(
                    model=model,
                    model_name=model_name,
                    model_label=model_label,
                    data=data,
                    lengths=lengths,
                    deltas=deltas,
                    selected_length=int(selected_length),
                    sampling=sampling,
                    factor=int(factor),
                    base_warmup=base_warmup,
                    device=device,
                )
                rows.append(row)
                done_keys.add(key)
                rows = sorted(rows, key=row_sort_key)
                write_summary(summary_path, rows)
                print(
                    f"[sampling] {model_name} {frequency_label(selected_length)} {sampling} factor={factor} "
                    f"mean={float(row['mean_err']):.5f}",
                    flush=True,
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        model_source_metadata = {}
        case_metadata_paths = []

    plot_downsample_factors = {int(factor) for factor in args.plot_downsample_factors}
    low2_plot_factors = {int(factor) for factor in args.low2_plot_factors}
    rows = sorted(rows, key=row_sort_key)
    write_summary(summary_path, rows)
    save_plots(
        output_dir,
        rows,
        args.models,
        args.copy_to_paper,
        plot_downsample_factors=plot_downsample_factors,
        low2_plot_factors=low2_plot_factors,
    )
    metadata = {
        "output_dir": str(output_dir.resolve()),
        "paper_figure_dir": str(PAPER_FIGURE_DIR.resolve()) if args.copy_to_paper else None,
        "models": args.models,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "subcycles_per_sequence": SUBCYCLES_PER_SEQUENCE,
        "frequency_policy": "MagNetX nominal frequency labels are used when the sequence length is known; otherwise the fallback is 100 * 16 MHz / sequence_length / 1000.",
        "magnetx_nominal_frequency_khz": MAGNETX_NOMINAL_FREQUENCY_KHZ,
        "upsampling_figure_policy": "all available upsampling points plus each length group's training-time downsampling factor; y-axis capped at 20%",
        "downsampling_figure": f"{LOW2_STEM}.pdf",
        "downsampling_figure_policy": (
            "two lowest nominal-frequency groups only, original sampling to 32x downsampling; "
            f"y-axis limited to {LOW2_YMAX:g}% (higher values are clipped by the axis)"
        ),
        "low2_plot_factors": sorted(low2_plot_factors),
        "low2_only": bool(args.low2_only),
        "train_downsample_factor_by_length": TRAIN_DOWNSAMPLE_FACTOR_BY_LENGTH,
        "high_upsample_factors": sorted(dict.fromkeys(int(factor) for factor in args.high_upsample_factors)),
        "low_upsample_factors": [] if args.no_low_upsampling else sorted(dict.fromkeys(int(factor) for factor in args.low_upsample_factors)),
        "downsample_factors": sorted(dict.fromkeys(int(factor) for factor in args.downsample_factors)),
        "plot_downsample_factors": sorted(plot_downsample_factors),
        "plotted_metric": "mean relative RMS error only; P95 is retained in CSV but not plotted",
        "summary_csv": str(summary_path.resolve()),
        "model_source_paths": model_source_metadata,
        "case_metadata": case_metadata_paths,
        "copy_to_paper": bool(args.copy_to_paper),
        "device": str(args.device),
        "torch_threads": torch.get_num_threads(),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[sampling] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
