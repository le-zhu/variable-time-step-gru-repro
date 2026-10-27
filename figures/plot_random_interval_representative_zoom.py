import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures.paper_figure_sizes import PAPER_FIGURE_SIZES_IN, figure_size, line_width, marker_area, marker_size
from figures.paper_plot_cache import load_plot_cache, save_plot_cache
from figures.paper_plot_style import apply_paper_plot_style

from scripts.evaluate_n87_longest_random_interval import (
    DEFAULT_DATA_DIR,
    DEFAULT_RUN_DIR,
    evaluate_batch,
    load_model,
    load_origin_test,
    make_random_interval_batch,
    relative_rms,
    resolve_device,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "figures" / "output"
PAPER_FIGURE_DIR = PAPER_DIR
PAPER_PLOT_DATA_DIR = PAPER_DIR / "plot_data"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "n87_random_interval_zoom"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw the representative random-interval waveform (Fig. 11 of the revised manuscript)."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-figure-dir", type=Path, default=PAPER_FIGURE_DIR)
    parser.add_argument("--copy-to-paper", action="store_true")
    parser.add_argument("--stem", default="random_interval_representative_zoom_points_1to10")
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--row-idx", type=int, default=668)
    parser.add_argument("--window-start", type=int, default=16700)
    parser.add_argument("--window-end", type=int, default=18200)
    parser.add_argument(
        "--auto-window-span",
        type=int,
        default=None,
        help="If set, search all selected rows for a window of this width whose local error is closest to the seed mean error.",
    )
    parser.add_argument(
        "--meanlike-row",
        action="store_true",
        help="With --auto-window-span, first pick the row whose error is closest to the seed mean, then search windows only in that row.",
    )
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--min-window-points", type=int, default=50)
    parser.add_argument("--warmup-original-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-only-cache", type=Path, default=None, help="Replot from a saved NPZ cache without model inference.")
    return parser.parse_args()


def selected_longest_rows(lengths):
    max_len = int(lengths.max())
    rows = np.flatnonzero(lengths == max_len).astype(int).tolist()
    if len(rows) != 23 or max_len != 32016:
        raise RuntimeError(f"Expected 23 longest rows of length 32016, got {len(rows)} rows length {max_len}.")
    return rows, max_len


def setup_dirs(output_dir, paper_figure_dir, copy_to_paper):
    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_to_paper:
        paper_figure_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, paper_figure_dir


def padded_limits(values, *, pad_fraction=0.08, min_pad=0.01):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    if span <= 0:
        pad = max(abs(high) * pad_fraction, min_pad)
    else:
        pad = max(span * pad_fraction, min_pad)
    return low - pad, high + pad


def window_relative_rms(pred_norm, true_norm, original_index, warmup_original_steps, start, end):
    evaluation_point = np.zeros(len(original_index), dtype=bool)
    if len(original_index) > 1:
        evaluation_point[1:] = original_index[:-1] >= warmup_original_steps
    mask = (original_index >= start) & (original_index <= end) & evaluation_point
    if not np.any(mask):
        return float("nan"), 0
    pred_slice = pred_norm[mask]
    target_slice = true_norm[mask]
    rms_err = np.sqrt(np.mean((pred_slice - target_slice) ** 2))
    rms_target = np.sqrt(np.mean(target_slice**2))
    if rms_target <= 0:
        return float("nan"), int(np.count_nonzero(mask))
    return float(rms_err / rms_target), int(np.count_nonzero(mask))


def choose_representative_window(
    *,
    pred_norm,
    h_true_norm,
    original_indices,
    batch_lengths,
    selected_rows,
    target_error,
    span,
    stride,
    min_points,
    warmup_original_steps,
):
    if span <= 0:
        raise ValueError("--auto-window-span must be positive")
    if stride <= 0:
        raise ValueError("--window-stride must be positive")
    if min_points <= 0:
        raise ValueError("--min-window-points must be positive")

    best = None
    for pos, row_idx in enumerate(selected_rows):
        length = int(batch_lengths[pos])
        sample_index = original_indices[pos, :length]
        first_start = max(int(sample_index[0]), int(warmup_original_steps) + 1)
        last_start = int(sample_index[-1]) - span
        if last_start < first_start:
            continue
        for start in range(first_start, last_start + 1, stride):
            end = start + span
            local_error, point_count = window_relative_rms(
                pred_norm[pos, :length],
                h_true_norm[pos, :length],
                sample_index,
                warmup_original_steps,
                start,
                end,
            )
            if not np.isfinite(local_error) or point_count < min_points:
                continue
            score = abs(local_error - target_error)
            candidate = {
                "score": float(score),
                "row_idx": int(row_idx),
                "position": int(pos),
                "window_start": int(start),
                "window_end": int(end),
                "window_error": float(local_error),
                "window_points": int(point_count),
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "Could not find a representative window. Try reducing --min-window-points or --window-stride."
        )
    return best


def draw_representative_plot(
    *,
    output_dir,
    paper_figure_dir,
    stem,
    original_index,
    h_true,
    h_pred,
    b_t,
    std_h,
    std_b,
    row_err,
    seed,
    row_idx,
    max_step,
    window_start,
    window_end,
    copy_to_paper,
):
    window_mask = (original_index >= window_start) & (original_index <= window_end)
    if not np.any(window_mask):
        raise RuntimeError("The requested plotting window does not contain sampled points.")

    x = original_index[window_mask]
    true_window = h_true[window_mask]
    pred_window = h_pred[window_mask]
    b_window = b_t[window_mask]
    normalized_values = np.concatenate(
        [
            true_window / float(std_h),
            pred_window / float(std_h),
            b_window / float(std_b),
        ]
    )
    z_low, z_high = padded_limits(normalized_values, pad_fraction=0.08, min_pad=0.05)

    gap_x = original_index[1:]
    gaps = np.diff(original_index)
    gap_mask = (gap_x >= window_start) & (gap_x <= window_end)

    apply_paper_plot_style(font_size=8)
    size_key = stem if stem in PAPER_FIGURE_SIZES_IN else "random_interval_representative_zoom_points_1to10_window750_meanlike"
    fig, (ax_h, ax_gap) = plt.subplots(
        2,
        1,
        figsize=figure_size(size_key),
        dpi=180,
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [3.15, 0.85], "hspace": 0.13},
    )
    ax_b = ax_h.twinx()
    h_true_width = line_width(size_key, 1.05)
    h_pred_width = line_width(size_key, 0.95)
    b_width = line_width(size_key, 0.9)
    grid_width = line_width(size_key, 0.7)
    gap_width = line_width(size_key, 0.7)
    marker_edge = line_width(size_key, 0.6)
    spine_width = line_width(size_key, 0.8)
    tick_width = line_width(size_key, 0.8)
    tick_length = marker_size(size_key, 3.5)

    h_true_line = ax_h.plot(x, true_window, color="#111111", linewidth=h_true_width, label="True H", zorder=3)[0]
    h_pred_line = ax_h.plot(
        x,
        pred_window,
        color="#D62728",
        linewidth=h_pred_width,
        label="Pred H",
        zorder=4,
    )[0]
    b_line = ax_b.plot(
        x,
        1000.0 * b_window,
        color="#9AA4B2",
        linewidth=b_width,
        alpha=0.82,
        label="B input",
        zorder=2,
    )[0]
    sampled = ax_h.scatter(
        x,
        true_window,
        s=marker_area(size_key, 7.0),
        facecolors="white",
        edgecolors="#303030",
        linewidths=marker_edge,
        label="Sampled points",
        zorder=5,
    )

    ax_h.text(
        0.5,
        0.96,
        f"row err = {100.0 * row_err:.2f}%",
        transform=ax_h.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax_h.set_xlabel("")
    ax_h.set_ylabel("")
    ax_b.set_ylabel("")
    ax_h.set_ylim(z_low * float(std_h), z_high * float(std_h))
    ax_b.set_ylim(z_low * float(std_b) * 1000.0, z_high * float(std_b) * 1000.0)
    ax_h.grid(True, color="#E0E0E0", linewidth=grid_width, zorder=0)
    ax_h.tick_params(axis="both", labelsize=8, pad=1, width=tick_width, length=tick_length)
    ax_b.tick_params(axis="y", colors="#6B7280", labelsize=8, pad=1, width=tick_width, length=tick_length)
    for spine in ax_h.spines.values():
        spine.set_linewidth(spine_width)
    for spine in ax_b.spines.values():
        spine.set_linewidth(spine_width)
    ax_h.set_axisbelow(True)

    fig.subplots_adjust(left=0.065, right=0.935, bottom=0.170, top=0.840, hspace=0.18)
    top_box = ax_h.get_position()
    fig.legend(
        handles=[h_true_line, h_pred_line, b_line, sampled],
        loc="lower center",
        bbox_to_anchor=(0.5, top_box.y1 - 0.0287),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    ax_gap.vlines(gap_x[gap_mask], 1, gaps[gap_mask], color="#808A96", linewidth=gap_width, zorder=2)
    ax_gap.scatter(gap_x[gap_mask], gaps[gap_mask], color="#4B5563", s=marker_area(size_key, 7.0), zorder=3)
    ax_gap.set_ylabel("")
    ax_gap.set_xlabel("")
    ax_gap.set_ylim(0.8, max_step + 0.8)
    gap_ticks = [1, max_step]
    if max_step > 3:
        gap_ticks.insert(1, max(2, max_step // 2))
    ax_gap.set_yticks(gap_ticks)
    ax_gap.grid(True, color="#E0E0E0", linewidth=grid_width, zorder=0)
    ax_gap.tick_params(axis="both", labelsize=8, pad=1, width=tick_width, length=tick_length)
    for spine in ax_gap.spines.values():
        spine.set_linewidth(spine_width)
    ax_gap.set_axisbelow(True)
    ax_gap.set_xlim(window_start, window_end)

    bottom_box = ax_gap.get_position()
    fig.text(0.030, 0.51, "H (A/m)", ha="center", va="center", rotation=90, fontsize=8)
    fig.text(0.982, 0.51, "B (mT)", ha="center", va="center", rotation=90, color="#6B7280", fontsize=8)
    fig.text((bottom_box.x0 + bottom_box.x1) / 2, 0.055, "Original sample index", ha="center", va="center", fontsize=8)
    fig.text(bottom_box.x0, 0.055, f"row {row_idx}", ha="left", va="center", fontsize=8)
    output_paths = {}
    for suffix in ["png", "pdf", "svg"]:
        out_path = output_dir / f"{stem}.{suffix}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
        output_paths[suffix] = {"results": str(out_path.resolve())}
        if copy_to_paper and suffix == "pdf":
            paper_path = paper_figure_dir / f"{stem}.{suffix}"
            fig.savefig(paper_path, bbox_inches="tight", pad_inches=0.0)
            output_paths[suffix]["paper_figures"] = str(paper_path.resolve())
    plt.close(fig)
    return output_paths


def save_random_plot_cache(
    path: Path,
    *,
    stem: str,
    original_index,
    h_true,
    h_pred,
    b_t,
    std_h: float,
    std_b: float,
    row_err: float,
    seed: int,
    row_idx: int,
    max_step: int,
    window_start: int,
    window_end: int,
) -> None:
    save_plot_cache(
        path,
        arrays={
            "original_index": np.asarray(original_index, dtype=np.int64),
            "h_true": np.asarray(h_true, dtype=np.float32),
            "h_pred": np.asarray(h_pred, dtype=np.float32),
            "b_t": np.asarray(b_t, dtype=np.float32),
        },
        metadata={
            "stem": stem,
            "std_h": float(std_h),
            "std_b": float(std_b),
            "row_err": float(row_err),
            "seed": int(seed),
            "row_idx": int(row_idx),
            "max_step": int(max_step),
            "window_start": int(window_start),
            "window_end": int(window_end),
        },
    )


def replot_random_from_cache(cache_path: Path, output_dir: Path, paper_figure_dir: Path, copy_to_paper: bool) -> dict[str, dict[str, str]]:
    arrays, metadata = load_plot_cache(cache_path)
    return draw_representative_plot(
        output_dir=output_dir,
        paper_figure_dir=paper_figure_dir,
        stem=str(metadata["stem"]),
        original_index=arrays["original_index"],
        h_true=arrays["h_true"],
        h_pred=arrays["h_pred"],
        b_t=arrays["b_t"],
        std_h=float(metadata["std_h"]),
        std_b=float(metadata["std_b"]),
        row_err=float(metadata["row_err"]),
        seed=int(metadata["seed"]),
        row_idx=int(metadata["row_idx"]),
        max_step=int(metadata["max_step"]),
        window_start=int(metadata["window_start"]),
        window_end=int(metadata["window_end"]),
        copy_to_paper=copy_to_paper,
    )


def main():
    args = parse_args()
    if args.max_step < 1:
        raise ValueError("--max-step must be >= 1")
    if args.window_end <= args.window_start:
        raise ValueError("--window-end must be greater than --window-start")

    output_dir, paper_figure_dir = setup_dirs(args.output_dir, args.paper_figure_dir, args.copy_to_paper)
    if args.plot_only_cache is not None:
        cache_path = args.plot_only_cache if args.plot_only_cache.is_absolute() else ROOT / args.plot_only_cache
        output_paths = replot_random_from_cache(cache_path, output_dir, paper_figure_dir, args.copy_to_paper)
        print(json.dumps({"plot_only_cache": str(cache_path.resolve()), "output_paths": output_paths}, indent=2))
        return

    device = resolve_device(args.device)
    model, run_config, model_path = load_model(args.run_dir, device)
    data, lengths, deltas, norm_stats = load_origin_test(args.data_dir)
    selected_rows, max_len = selected_longest_rows(lengths)
    if args.row_idx not in selected_rows:
        raise RuntimeError(f"row_idx={args.row_idx} is not one of the selected longest rows: {selected_rows}")

    warmup_original_steps = (
        int(run_config["train"]["eval_warmup_len"])
        if args.warmup_original_steps is None
        else int(args.warmup_original_steps)
    )
    batch_data, batch_lengths, dt_steps, original_indices, sample_stats = make_random_interval_batch(
        data, lengths, deltas, selected_rows, args.seed, args.max_step
    )
    row_err, pred_norm, h_true_norm = evaluate_batch(
        model,
        batch_data,
        batch_lengths,
        dt_steps,
        original_indices,
        warmup_original_steps,
        device,
    )

    valid = row_err[~np.isnan(row_err)]
    seed_mean_err = float(np.mean(valid)) if valid.size else float("nan")
    seed_p95_err = float(np.percentile(valid, 95)) if valid.size else float("nan")

    selected_window = None
    if args.auto_window_span is not None:
        search_positions = list(range(len(selected_rows)))
        if args.meanlike_row:
            finite = [p for p in search_positions if np.isfinite(row_err[p])]
            search_positions = [min(finite, key=lambda p: abs(float(row_err[p]) - seed_mean_err))]
        selected_window = choose_representative_window(
            pred_norm=pred_norm[search_positions],
            h_true_norm=h_true_norm[search_positions],
            original_indices=original_indices[search_positions],
            batch_lengths=batch_lengths[search_positions],
            selected_rows=[selected_rows[p] for p in search_positions],
            target_error=seed_mean_err,
            span=int(args.auto_window_span),
            stride=int(args.window_stride),
            min_points=int(args.min_window_points),
            warmup_original_steps=warmup_original_steps,
        )
        args.row_idx = int(selected_window["row_idx"])
        args.window_start = int(selected_window["window_start"])
        args.window_end = int(selected_window["window_end"])

    pos = selected_rows.index(args.row_idx)
    length = int(batch_lengths[pos])
    sample_index = original_indices[pos, :length]
    std_h = float(norm_stats["std_h"])
    std_b = float(norm_stats["std_b"])
    h_true = h_true_norm[pos, :length] * std_h
    h_pred = pred_norm[pos, :length] * std_h
    b_t = batch_data[pos, :length, 0] * std_b
    representative_err = relative_rms(
        pred_norm[pos],
        h_true_norm[pos],
        original_indices[pos],
        warmup_original_steps,
        length,
    )
    window_err, window_points = window_relative_rms(
        pred_norm[pos, :length],
        h_true_norm[pos, :length],
        sample_index,
        warmup_original_steps,
        args.window_start,
        args.window_end,
    )

    output_paths = draw_representative_plot(
        output_dir=output_dir,
        paper_figure_dir=paper_figure_dir,
        stem=args.stem,
        original_index=sample_index,
        h_true=h_true,
        h_pred=h_pred,
        b_t=b_t,
        std_h=std_h,
        std_b=std_b,
        row_err=representative_err,
        seed=args.seed,
        row_idx=args.row_idx,
        max_step=args.max_step,
        window_start=args.window_start,
        window_end=args.window_end,
        copy_to_paper=args.copy_to_paper,
    )
    PAPER_PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_cache_path = output_dir / f"{args.stem}_plot_data.npz"
    paper_cache_path = PAPER_PLOT_DATA_DIR / f"{args.stem}.npz"
    for cache_path in (output_cache_path, paper_cache_path):
        save_random_plot_cache(
            cache_path,
            stem=args.stem,
            original_index=sample_index,
            h_true=h_true,
            h_pred=h_pred,
            b_t=b_t,
            std_h=std_h,
            std_b=std_b,
            row_err=representative_err,
            seed=args.seed,
            row_idx=args.row_idx,
            max_step=args.max_step,
            window_start=args.window_start,
            window_end=args.window_end,
        )

    metadata = {
        "run_dir": str(args.run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "paper_figure_dir": str(paper_figure_dir.resolve()),
        "stem": args.stem,
        "model_key": run_config["model_key"],
        "run_name": run_config["run_name"],
        "data_name": run_config["data_name"],
        "dt_norm_mode": run_config["dt_norm_mode"],
        "seed": int(args.seed),
        "max_step": int(args.max_step),
        "row_idx": int(args.row_idx),
        "row_error": float(representative_err),
        "window_error": float(window_err),
        "window_points": int(window_points),
        "window_error_target": seed_mean_err,
        "auto_window_selection": selected_window,
        "seed_mean_error": seed_mean_err,
        "seed_p95_error": seed_p95_err,
        "window_start": int(args.window_start),
        "window_end": int(args.window_end),
        "axis_scale_policy": "shared normalized limits: H axes use z*std_h and B axes use z*std_b; tick labels remain physical units",
        "selected_length": max_len,
        "selected_row_count": len(selected_rows),
        "selected_row_indices": selected_rows,
        "sample_stats": sample_stats,
        "warmup_original_steps": int(warmup_original_steps),
        "error_start_policy": (
            "include outputs starting with the output produced by the first "
            "autoregressive H input"
        ),
        "random_sampling_policy": "start at 0, repeatedly jump a random integer in [1, max_step], stop before overflow",
        "db_dt_policy": "recomputed after random sampling using dB_norm / per-step dt",
        "dt_policy": "variable per-step dt_t = origin_dt * sampled_index_gap",
        "device": str(device),
        "output_paths": output_paths,
        "output_plot_cache": str(output_cache_path.resolve()),
        "paper_plot_cache": str(paper_cache_path.resolve()),
    }
    json_paths = [output_dir / f"{args.stem}.json"]
    if args.copy_to_paper:
        json_paths.append(paper_figure_dir / f"{args.stem}.json")
    for json_path in json_paths:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    print(
        f"seed={args.seed} max_step={args.max_step} row={args.row_idx} "
        f"row_err={representative_err:.6f} window_err={window_err:.6f}"
    )
    print(f"window={args.window_start}..{args.window_end} points={window_points}")
    print(f"seed mean={seed_mean_err:.6f} p95={seed_p95_err:.6f}")
    print(f"Saved figure to {output_dir}")
    if args.copy_to_paper:
        print(f"Copied figure to {paper_figure_dir}")


if __name__ == "__main__":
    main()
