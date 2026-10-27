import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .no_h_waveform_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_H_FEEDBACK_RUN_DIR,
    DEFAULT_RUN_DIR,
    evaluate_rows,
    load_longest_origin_split,
    load_model,
    resolve_device,
)
from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_style import apply_paper_plot_style
from figures.paper_plot_cache import load_plot_cache, save_plot_cache


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "figures" / "output"
PAPER_FIGURE_DIR = PAPER_DIR
PAPER_PLOT_DATA_DIR = PAPER_DIR / "plot_data"
OUTPUT_BASENAME = "no_h_initial_sync_comparison"
CANDIDATE_COUNT = 8
# The revised manuscript (Fig. 7) shows validation-set rank 6 (origin_val row 324).
PAPER_CANDIDATE_RANKS = (6,)
INITIAL_WINDOW = (1, 1000)
LATER_WINDOW = (2500, 4500)
PLOT_WINDOW = (0, 2000)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or replot the RateGRU-NF (No-H) initial-sync paper figure (Fig. 7).")
    parser.add_argument("--plot-only-cache", type=Path, default=None, help="Replot from a saved NPZ cache without model inference.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to a timestamped directory under results/.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Processed N87 split (std_db_dt); defaults to data/N87_processed.")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Original data split used for candidate selection; the manuscript uses val.")
    parser.add_argument("--candidate-count", type=int, default=CANDIDATE_COUNT, help="Number of candidate sequences to render.")
    parser.add_argument(
        "--skip-paper-copy",
        action="store_true",
        help="Keep all generated figures and caches in the output directory without modifying paper artifacts.",
    )
    return parser.parse_args()


def candidate_row_index(candidate):
    """Read new split-neutral caches and legacy test-only caches."""
    if "origin_row" in candidate:
        return int(candidate["origin_row"])
    return int(candidate["origin_test_row"])


def window_mean_abs_error(pred_h, true_h, start, end):
    start = max(0, int(start))
    end = min(pred_h.shape[-1], int(end))
    if end <= start:
        return float("nan")
    return float(np.mean(np.abs(pred_h[start:end] - true_h[start:end])))


def choose_initial_sync_candidate(no_h_pred_h, main_pred_h, true_h, lengths, row_indices):
    candidates = []
    for pos, row_idx in enumerate(row_indices):
        length = int(lengths[pos])
        initial_start, initial_end = INITIAL_WINDOW
        later_start, later_end = LATER_WINDOW
        if length <= max(initial_end, later_end):
            continue

        no_h_initial = window_mean_abs_error(no_h_pred_h[pos], true_h[pos], initial_start, initial_end)
        no_h_later = window_mean_abs_error(no_h_pred_h[pos], true_h[pos], later_start, later_end)
        main_initial = window_mean_abs_error(main_pred_h[pos], true_h[pos], initial_start, initial_end)
        main_later = window_mean_abs_error(main_pred_h[pos], true_h[pos], later_start, later_end)

        ratio = no_h_initial / max(no_h_later, 1e-9)
        score = (no_h_initial - no_h_later) * np.log1p(no_h_initial) * min(ratio, 5.0)
        candidates.append(
            {
                "score": float(score),
                "origin_row": int(row_idx),
                "position": int(pos),
                "no_h_initial_mean_abs_error": no_h_initial,
                "no_h_later_mean_abs_error": no_h_later,
                "no_h_initial_later_ratio": float(ratio),
                "main_initial_mean_abs_error": main_initial,
                "main_later_mean_abs_error": main_later,
            }
        )
    if not candidates:
        raise RuntimeError("No candidate has sufficient length for the initial-sync comparison.")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], candidates


def plot_initial_sync_comparison(
    output_dir,
    output_basename,
    row_idx,
    true_h,
    main_pred_h,
    no_h_pred_h,
    main_row_rms,
    no_h_row_rms,
    main_warmup_steps,
    candidate,
):
    start, end = PLOT_WINDOW
    x = np.arange(start, end)

    apply_paper_plot_style(font_size=9, extra={"axes.linewidth": 0.8})

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.4, 4.8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )
    warmup_width = line_width(OUTPUT_BASENAME, 0.7)
    true_width = line_width(OUTPUT_BASENAME, 0.72)
    pred_width = line_width(OUTPUT_BASENAME, 0.6)
    grid_width = line_width(OUTPUT_BASENAME, 0.5)
    spine_width = line_width(OUTPUT_BASENAME, 0.8)
    tick_width = line_width(OUTPUT_BASENAME, 0.8)
    tick_length = marker_size(OUTPUT_BASENAME, 3.5)

    rows = [
        {
            "ax": axes[0],
            "pred": main_pred_h,
            "ylabel": "RateGRU\nH",
            "rms": main_row_rms,
            "note": f"teacher forcing warmup: 0-{main_warmup_steps}",
        },
        {
            "ax": axes[1],
            "pred": no_h_pred_h,
            "ylabel": "RateGRU-NF\nH",
            "rms": no_h_row_rms,
            "note": "RateGRU-NF: no autoregressive H input",
        },
    ]

    for row in rows:
        ax = row["ax"]
        ax.plot(x, true_h[start:end], color="#1f77b4", linewidth=1.6, label="True H", zorder=3)
        ax.plot(
            x,
            row["pred"][start:end],
            color="#ff7f0e",
            linewidth=1.2,
            linestyle="--",
            label="Prediction",
            zorder=3,
        )
        ax.set_ylabel(row["ylabel"])
        ax.grid(True, color="#e2e2e2", linewidth=0.6, zorder=1)
        ax.text(
            0.01,
            0.93,
            f"row={row_idx}, RMS={row['rms']:.5f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.text(
            0.99,
            0.08,
            row["note"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes:
        ax.axvline(main_warmup_steps, color="#d62728", linewidth=1.0, linestyle="--", zorder=2)
    axes[0].text(
        main_warmup_steps + 35,
        0.78,
        "teacher forcing",
        transform=axes[0].get_xaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color="#4a4a4a",
    )

    axes[1].set_xlabel("Time step")
    axes[1].set_xlim(start, end)

    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.6, label="True H"),
        Line2D([0], [0], color="#ff7f0e", linewidth=1.2, linestyle="--", label="Prediction"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        frameon=False,
        ncol=2,
        handlelength=2.2,
    )

    fig.text(
        0.075,
        0.02,
        (
            f"No-H initial/later mean |error|: "
            f"{candidate['no_h_initial_mean_abs_error']:.2f} -> {candidate['no_h_later_mean_abs_error']:.2f}"
        ),
        ha="left",
        va="bottom",
        fontsize=8,
    )

    fig.subplots_adjust(left=0.075, right=0.975, top=0.94, bottom=0.13)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{output_basename}.{suffix}")
    plt.close(fig)


def plot_candidates_overview(
    output_dir,
    candidates,
    true_h,
    main_pred_h,
    no_h_pred_h,
    main_row_rms,
    no_h_row_rms,
    main_warmup_steps,
):
    start, end = PLOT_WINDOW
    x = np.arange(start, end)
    count = len(candidates)

    fig, axes = plt.subplots(
        count,
        2,
        figsize=(14.4, 2.05 * count),
        dpi=180,
        sharex=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.22},
    )
    if count == 1:
        axes = np.array([axes])

    for rank, candidate in enumerate(candidates, start=1):
        pos = candidate["position"]
        row_idx = candidate_row_index(candidate)
        title = (
            f"#{rank} row={row_idx}, "
            f"No-H {candidate['no_h_initial_mean_abs_error']:.2f}->{candidate['no_h_later_mean_abs_error']:.2f}"
        )
        row_axes = axes[rank - 1]
        series = [
            ("RateGRU", main_pred_h[pos], float(main_row_rms[pos]), row_axes[0]),
            ("RateGRU-NF", no_h_pred_h[pos], float(no_h_row_rms[pos]), row_axes[1]),
        ]
        for model_label, pred, rms, ax in series:
            ax.plot(x, true_h[pos, start:end], color="#1f77b4", linewidth=1.0, zorder=3)
            ax.plot(x, pred[start:end], color="#ff7f0e", linewidth=0.9, linestyle="--", zorder=3)
            ax.axvline(main_warmup_steps, color="#d62728", linewidth=0.8, linestyle="--", zorder=2)
            ax.grid(True, color="#e5e5e5", linewidth=0.5, zorder=1)
            ax.text(0.01, 0.9, f"{model_label}, RMS={rms:.5f}", transform=ax.transAxes, fontsize=7, va="top")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        row_axes[0].set_ylabel(title, fontsize=7)

    axes[0, 0].set_title("RateGRU with H feedback", fontsize=9)
    axes[0, 1].set_title("RateGRU-NF", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Time step")
        ax.set_xlim(start, end)

    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.2, label="True H"),
        Line2D([0], [0], color="#ff7f0e", linewidth=1.0, linestyle="--", label="Prediction"),
        Line2D([0], [0], color="#d62728", linewidth=0.9, linestyle="--", label="Warmup boundary"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.985, 0.995))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.955, bottom=0.045)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"no_h_initial_sync_candidates_overview.{suffix}")
    plt.close(fig)


def plot_paper_candidate_grid(
    output_dir,
    candidates,
    true_h,
    main_pred_h,
    no_h_pred_h,
    main_row_rms,
    no_h_row_rms,
    main_warmup_steps,
    selected_ranks=PAPER_CANDIDATE_RANKS,
    output_basename=OUTPUT_BASENAME,
):
    apply_paper_plot_style(font_size=8, extra={"axes.linewidth": 0.8})
    start, end = PLOT_WINDOW
    x = np.arange(start, end)
    selected = [candidates[rank - 1] for rank in selected_ranks]

    fig, axes = plt.subplots(
        2,
        len(selected),
        figsize=figure_size(OUTPUT_BASENAME),
        dpi=180,
        sharex=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.18},
    )
    axes = np.asarray(axes).reshape(2, len(selected))
    warmup_width = line_width(OUTPUT_BASENAME, 0.7)
    true_width = line_width(OUTPUT_BASENAME, 0.72)
    pred_width = line_width(OUTPUT_BASENAME, 0.6)
    grid_width = line_width(OUTPUT_BASENAME, 0.5)
    spine_width = line_width(OUTPUT_BASENAME, 0.8)
    tick_width = line_width(OUTPUT_BASENAME, 0.8)
    tick_length = marker_size(OUTPUT_BASENAME, 3.5)

    for col, candidate in enumerate(selected):
        pos = candidate["position"]
        row_idx = candidate_row_index(candidate)
        title = f"row {row_idx}"

        panel_specs = [
            ("RateGRU", main_pred_h[pos], float(main_row_rms[pos]), axes[0, col]),
            ("RateGRU-NF", no_h_pred_h[pos], float(no_h_row_rms[pos]), axes[1, col]),
        ]
        for model_label, pred, rms, ax in panel_specs:
            if ax is axes[0, col]:
                ax.axvspan(start, main_warmup_steps, color="#9CA3AF", alpha=0.12, linewidth=0, zorder=0)
                ax.axvline(main_warmup_steps, color="#6B7280", linewidth=warmup_width, linestyle="--", zorder=2)
            ax.plot(x, true_h[pos, start:end], color="#111111", linewidth=true_width, zorder=4)
            ax.plot(x, pred[start:end], color="#d62728", linewidth=pred_width, linestyle="-", zorder=4)
            ax.set_axisbelow(False)
            ax.grid(True, color="#e5e5e5", linewidth=grid_width, zorder=1)
            ax.set_title("")
            ax.text(
                0.5,
                0.02,
                f"row err = {100.0 * rms:.2f}%",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(spine_width)
            ax.spines["bottom"].set_linewidth(spine_width)
            ax.set_xlim(start, end + 40)
            ax.tick_params(axis="both", labelsize=8, pad=1, width=tick_width, length=tick_length)
            for tick_label in ax.get_xticklabels():
                if tick_label.get_text() == str(end):
                    tick_label.set_ha("right")

    fig.subplots_adjust(left=0.145, right=0.985, top=0.82, bottom=0.18, hspace=0.48)
    top_box = axes[0, 0].get_position()
    bottom_box = axes[-1, 0].get_position()
    axis_center_x = (bottom_box.x0 + bottom_box.x1) / 2
    model_label_x = 0.070

    handles = [
        Line2D([0], [0], color="#111111", linewidth=true_width, label="True H"),
        Line2D([0], [0], color="#d62728", linewidth=pred_width, linestyle="-", label="Pred H"),
    ]
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(axis_center_x, top_box.y1 - 0.0325),
        fontsize=8,
        handlelength=2.0,
        handletextpad=0.35,
        labelspacing=0.18,
        columnspacing=0.7,
        borderpad=0.0,
    )
    fig.text(axis_center_x, 0.065, "Time step", ha="center", va="center", fontsize=8)
    fig.text(bottom_box.x0, 0.065, title, ha="left", va="center", fontsize=8)
    fig.text(0.023, 0.50, "H (A/m)", ha="center", va="center", rotation=90, fontsize=8)
    fig.text(model_label_x, (top_box.y0 + top_box.y1) / 2, "RateGRU", ha="center", va="center", rotation=90, fontsize=8)
    fig.text(model_label_x, (bottom_box.y0 + bottom_box.y1) / 2, "RateGRU-NF", ha="center", va="center", rotation=90, fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{output_basename}.{suffix}", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_paper_plot_cache(
    cache_path: Path,
    candidates,
    true_h,
    main_pred_h,
    no_h_pred_h,
    main_row_rms,
    no_h_row_rms,
    main_warmup_steps,
) -> None:
    top_candidates = candidates[:CANDIDATE_COUNT]
    positions = [int(item["position"]) for item in top_candidates]
    cached_candidates = []
    for local_pos, item in enumerate(top_candidates):
        cached = dict(item)
        cached["position"] = local_pos
        cached_candidates.append(cached)
    save_plot_cache(
        cache_path,
        arrays={
            "true_h": np.asarray(true_h[positions], dtype=np.float32),
            "main_pred_h": np.asarray(main_pred_h[positions], dtype=np.float32),
            "no_h_pred_h": np.asarray(no_h_pred_h[positions], dtype=np.float32),
            "main_row_rms": np.asarray(main_row_rms[positions], dtype=np.float64),
            "no_h_row_rms": np.asarray(no_h_row_rms[positions], dtype=np.float64),
        },
        metadata={
            "stem": OUTPUT_BASENAME,
            "top_candidates": cached_candidates,
            "main_warmup_steps": int(main_warmup_steps),
            "no_h_row_rms_policy": "first main_warmup_steps excluded from the error, matching the paper protocol",
            "paper_candidate_ranks": list(PAPER_CANDIDATE_RANKS),
        },
    )


def replot_paper_from_cache(cache_path: Path, output_dir: Path, copy_to_paper: bool = True) -> None:
    arrays, metadata = load_plot_cache(cache_path)
    plot_paper_candidate_grid(
        output_dir,
        metadata["top_candidates"],
        arrays["true_h"],
        arrays["main_pred_h"],
        arrays["no_h_pred_h"],
        arrays["main_row_rms"],
        arrays["no_h_row_rms"],
        int(metadata["main_warmup_steps"]),
        selected_ranks=tuple(metadata.get("paper_candidate_ranks", PAPER_CANDIDATE_RANKS)),
    )
    if copy_to_paper:
        PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        source = output_dir / f"{OUTPUT_BASENAME}.pdf"
        target = PAPER_FIGURE_DIR / f"{OUTPUT_BASENAME}.pdf"
        target.write_bytes(source.read_bytes())


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results" / f"local_no_h_initial_sync_{stamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only_cache is not None:
        cache_path = args.plot_only_cache if args.plot_only_cache.is_absolute() else ROOT / args.plot_only_cache
        replot_paper_from_cache(cache_path, output_dir, copy_to_paper=not args.skip_paper_copy)
        print(
            json.dumps(
                {
                    "plot_only_cache": str(cache_path.resolve()),
                    "figure_copied_to": None
                    if args.skip_paper_copy
                    else str(PAPER_FIGURE_DIR / f"{OUTPUT_BASENAME}.pdf"),
                },
                indent=2,
            )
        )
        return

    if args.candidate_count < max(PAPER_CANDIDATE_RANKS):
        raise ValueError(f"--candidate-count must be at least {max(PAPER_CANDIDATE_RANKS)} (the paper candidate rank).")

    device = resolve_device()
    no_h_model, no_h_config, no_h_model_path = load_model(DEFAULT_RUN_DIR, device)
    main_model, main_config, main_model_path = load_model(DEFAULT_H_FEEDBACK_RUN_DIR, device)
    data, lengths, deltas, norm_stats, row_indices = load_longest_origin_split(args.data_dir, split=args.split)

    main_warmup_steps = int(main_config["train"]["eval_warmup_len"])
    # RateGRU-NF has no H feedback, so it is rolled out without teacher forcing
    # but scored over the same post-warmup steps as RateGRU.
    no_h_row_rms, no_h_pred_norm, true_norm = evaluate_rows(
        no_h_model, data, lengths, deltas, 0, device, eval_warmup_steps=main_warmup_steps
    )
    main_row_rms, main_pred_norm, _ = evaluate_rows(main_model, data, lengths, deltas, main_warmup_steps, device)

    std_h = float(norm_stats["std_h"])
    true_h = true_norm * std_h
    no_h_pred_h = no_h_pred_norm * std_h
    main_pred_h = main_pred_norm * std_h

    candidate, candidates = choose_initial_sync_candidate(no_h_pred_h, main_pred_h, true_h, lengths, row_indices)
    pos = candidate["position"]
    row_idx = candidate_row_index(candidate)

    plot_initial_sync_comparison(
        output_dir,
        OUTPUT_BASENAME,
        row_idx,
        true_h[pos],
        main_pred_h[pos],
        no_h_pred_h[pos],
        float(main_row_rms[pos]),
        float(no_h_row_rms[pos]),
        main_warmup_steps,
        candidate,
    )

    candidate_outputs = []
    top_candidates = candidates[: args.candidate_count]
    for rank, item in enumerate(top_candidates, start=1):
        item_pos = item["position"]
        item_row = candidate_row_index(item)
        basename = f"candidate_{rank:02d}_row_{item_row}_initial_sync"
        plot_initial_sync_comparison(
            output_dir,
            basename,
            item_row,
            true_h[item_pos],
            main_pred_h[item_pos],
            no_h_pred_h[item_pos],
            float(main_row_rms[item_pos]),
            float(no_h_row_rms[item_pos]),
            main_warmup_steps,
            item,
        )
        candidate_outputs.append(
            {
                "rank": rank,
                "origin_row": int(item_row),
                "basename": basename,
                "png": str(output_dir / f"{basename}.png"),
                "pdf": str(output_dir / f"{basename}.pdf"),
                "svg": str(output_dir / f"{basename}.svg"),
            }
        )

    plot_candidates_overview(
        output_dir,
        top_candidates,
        true_h,
        main_pred_h,
        no_h_pred_h,
        main_row_rms,
        no_h_row_rms,
        main_warmup_steps,
    )

    plot_paper_candidate_grid(
        output_dir,
        top_candidates,
        true_h,
        main_pred_h,
        no_h_pred_h,
        main_row_rms,
        no_h_row_rms,
        main_warmup_steps,
    )

    paper_candidate_outputs = []
    for rank, item in enumerate(top_candidates, start=1):
        item_row = candidate_row_index(item)
        basename = f"paper_candidate_{rank:02d}_row_{item_row}_initial_sync"
        plot_paper_candidate_grid(
            output_dir,
            top_candidates,
            true_h,
            main_pred_h,
            no_h_pred_h,
            main_row_rms,
            no_h_row_rms,
            main_warmup_steps,
            selected_ranks=(rank,),
            output_basename=basename,
        )
        paper_candidate_outputs.append(
            {
                "rank": rank,
                "origin_row": int(item_row),
                "basename": basename,
                "png": str(output_dir / f"{basename}.png"),
                "pdf": str(output_dir / f"{basename}.pdf"),
                "svg": str(output_dir / f"{basename}.svg"),
            }
        )

    output_cache_path = output_dir / f"{OUTPUT_BASENAME}_plot_data.npz"
    save_paper_plot_cache(
        output_cache_path,
        top_candidates,
        true_h,
        main_pred_h,
        no_h_pred_h,
        main_row_rms,
        no_h_row_rms,
        main_warmup_steps,
    )
    paper_cache_path = None
    figure_copied_to = None
    if not args.skip_paper_copy:
        PAPER_PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        paper_cache_path = PAPER_PLOT_DATA_DIR / f"{OUTPUT_BASENAME}.npz"
        save_paper_plot_cache(
            paper_cache_path,
            top_candidates,
            true_h,
            main_pred_h,
            no_h_pred_h,
            main_row_rms,
            no_h_row_rms,
            main_warmup_steps,
        )

        PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        source = output_dir / f"{OUTPUT_BASENAME}.pdf"
        target = PAPER_FIGURE_DIR / f"{OUTPUT_BASENAME}.pdf"
        target.write_bytes(source.read_bytes())
        figure_copied_to = str(target)

    metadata = {
        "no_h_run_dir": str(DEFAULT_RUN_DIR),
        "no_h_model_path": str(no_h_model_path),
        "main_run_dir": str(DEFAULT_H_FEEDBACK_RUN_DIR),
        "main_model_path": str(main_model_path),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "device": str(device),
        "longest_group_size": int(len(row_indices)),
        "longest_length": int(lengths.max()),
        "selected_origin_row": int(row_idx),
        "selected_position_in_longest_group": int(pos),
        "selection_rule": (
            f"Rank longest-{args.split} rows by (initial_error - later_error) * log1p(initial_error) "
            "* min(initial_error / later_error, 5), where initial is steps 1-1000 "
            "and later is steps 2500-4500."
        ),
        "initial_window": list(INITIAL_WINDOW),
        "later_window": list(LATER_WINDOW),
        "plot_window": list(PLOT_WINDOW),
        "main_warmup_steps": int(main_warmup_steps),
        "no_h_warmup_steps": 0,
        "no_h_eval_warmup_steps": int(main_warmup_steps),
        "main_selected_row_relative_rms": float(main_row_rms[pos]),
        "no_h_selected_row_relative_rms": float(no_h_row_rms[pos]),
        "selected_candidate": candidate,
        "top_candidates": candidates[:10],
        "candidate_count_rendered": len(top_candidates),
        "paper_candidate_ranks": list(PAPER_CANDIDATE_RANKS),
        "paper_candidate_rows": [candidate_row_index(top_candidates[rank - 1]) for rank in PAPER_CANDIDATE_RANKS],
        "candidate_outputs": candidate_outputs,
        "paper_candidate_outputs": paper_candidate_outputs,
        "overview_png": str(output_dir / "no_h_initial_sync_candidates_overview.png"),
        "overview_pdf": str(output_dir / "no_h_initial_sync_candidates_overview.pdf"),
        "overview_svg": str(output_dir / "no_h_initial_sync_candidates_overview.svg"),
        "figure_copied_to": figure_copied_to,
        "paper_plot_cache": None if paper_cache_path is None else str(paper_cache_path.resolve()),
        "output_plot_cache": str(output_cache_path.resolve()),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
