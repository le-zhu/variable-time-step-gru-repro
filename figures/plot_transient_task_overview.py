"""Introduction figure: input-output arrangement of the prediction task.

The excitation B(t) is known over the whole window, whereas the true H is
supplied only over the warmup segment and the remainder has to be predicted
autoregressively.  Only the H panel marks the warmup segment, with a grey
background, a boundary line, and a solid trace; the autoregressive segment is
left white and its trace is dashed.  The B panel carries no boundary, because
nothing changes there at that instant.

The layout is specified in inches at the final printed size, so the 8 pt text
is 8 pt on the page and no scaling happens at \\includegraphics time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_cache import load_plot_cache, save_plot_cache
from figures.paper_plot_style import apply_paper_plot_style


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "output"
PAPER_PLOT_DATA_DIR = DEFAULT_OUTPUT_DIR / "plot_data"

STEM = "transient_prediction_task"
W, H = figure_size(STEM)

SAMPLING_RATE_HZ = 16.0e6          # original MagNetX sampling rate
US_PER_STEP = 1.0e6 / SAMPLING_RATE_HZ

# axes box, in inches, measured from the lower-left corner of the canvas
PAD_L = 0.387                      # rotated y label + tick labels
PAD_R = 0.035
PAD_B = 0.315                      # x tick labels + x label
PAD_T = 0.185                      # legend strip
GAP = 0.075                        # between the two panels
PANEL_W = W - PAD_L - PAD_R
PANEL_H = (H - PAD_B - PAD_T - GAP) / 2.0
YLABEL_X = 0.068                   # keeps the rotated label clear of the tick labels

C_B = "#111111"
C_H = "#d62728"
C_BOUNDARY = "#9CA3AF"
C_WARMUP_SPAN = "#9CA3AF"
C_GRID = "#E0E0E0"

FS = 8.0
LW_B = line_width(STEM, 0.9)
LW_H = line_width(STEM, 0.9)
LW_BOUNDARY = line_width(STEM, 0.7)
LW_GRID = line_width(STEM, 0.7)
LW_CHROME = line_width(STEM, 0.8)
TICK_LEN = marker_size(STEM, 3.5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--row", type=int, default=656, help="Row index in origin_test.npy.")
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-end", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=100, help="Evaluation warmup length in original steps.")
    parser.add_argument("--plot-only-cache", type=Path, default=None, help="Replot from a saved NPZ cache without reading the dataset.")
    return parser.parse_args()


def load_window(data_dir: Path, row: int, start: int, end: int):
    data = np.load(data_dir / "origin_test.npy", mmap_mode="r")
    lengths = np.load(data_dir / "origin_test_lengths.npy").astype(np.int64)
    with (data_dir / "norm_stats.json").open(encoding="utf-8") as handle:
        norm_stats = json.load(handle)
    length = int(lengths[row])
    if end > length:
        raise ValueError(f"--window-end {end} exceeds the length {length} of row {row}")
    window = np.asarray(data[row, start:end, :], dtype=np.float64)
    b_mt = window[:, 0] * float(norm_stats["std_b"]) * 1000.0
    h_am = window[:, 3] * float(norm_stats["std_h"])
    temperature = float(window[0, 2])
    return b_mt, h_am, temperature, length


def padded_limits(values, pad_low=0.12, pad_high=0.12):
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-6)
    return low - span * pad_low, high + span * pad_high


def style_axes(ax):
    ax.grid(True, color=C_GRID, linewidth=LW_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", labelsize=FS, pad=1, width=LW_CHROME, length=TICK_LEN)
    for spine in ax.spines.values():
        spine.set_linewidth(LW_CHROME)


def draw_figure(
    *,
    output_dir: Path,
    b_mt,
    h_am,
    window_start: int,
    warmup_steps: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_plot_style(font_size=FS, extra={"svg.hashsalt": STEM})

    steps = window_start + np.arange(len(b_mt))
    t_us = steps * US_PER_STEP
    t_split = warmup_steps * US_PER_STEP
    t_first, t_last = float(t_us[0]), float(t_us[-1])

    warmup_mask = steps <= warmup_steps
    predict_mask = steps >= warmup_steps

    fig = plt.figure(figsize=(W, H), dpi=300)
    ax_b = fig.add_axes([PAD_L / W, (PAD_B + PANEL_H + GAP) / H, PANEL_W / W, PANEL_H / H])
    ax_h = fig.add_axes([PAD_L / W, PAD_B / H, PANEL_W / W, PANEL_H / H])

    for ax in (ax_b, ax_h):
        ax.set_xlim(t_first, t_last)
    # B is known throughout, so only the H panel marks the warmup segment
    ax_h.axvline(t_split, color=C_BOUNDARY, linewidth=LW_BOUNDARY, linestyle="--", zorder=2)
    ax_h.axvspan(t_first, t_split, color=C_WARMUP_SPAN, alpha=0.16, linewidth=0, zorder=0)

    # ---------------- top: the excitation, known throughout ----------------
    b_line = ax_b.plot(t_us, b_mt, color=C_B, linewidth=LW_B, zorder=3, label="Known $B$")[0]
    ax_b.set_ylim(*padded_limits(b_mt))
    ax_b.set_xticklabels([])
    style_axes(ax_b)

    # ---------------- bottom: the response, known only over the warmup ----------------
    h_true_line = ax_h.plot(
        t_us[warmup_mask],
        h_am[warmup_mask],
        color=C_H,
        linewidth=LW_H,
        zorder=4,
        label="True $H$",
    )[0]
    h_unknown_line = ax_h.plot(
        t_us[predict_mask],
        h_am[predict_mask],
        color=C_H,
        linewidth=LW_H,
        linestyle=(0, (3, 2)),
        zorder=3,
        label="$H$ to predict",
    )[0]
    ax_h.set_ylim(*padded_limits(h_am))
    style_axes(ax_h)

    # ---------------- chrome ----------------
    fig.legend(
        handles=[b_line, h_true_line, h_unknown_line],
        loc="lower center",
        bbox_to_anchor=(PAD_L / W + 0.5 * PANEL_W / W, 1.0 - PAD_T / H),
        ncol=3,
        frameon=False,
        fontsize=FS,
        handlelength=1.6,
        handletextpad=0.35,
        labelspacing=0.15,
        columnspacing=1.0,
        borderpad=0.0,
    )
    fig.text(YLABEL_X / W, (PAD_B + PANEL_H + GAP + 0.5 * PANEL_H) / H, "$B$ (mT)",
             ha="center", va="center", rotation=90, fontsize=FS)
    fig.text(YLABEL_X / W, (PAD_B + 0.5 * PANEL_H) / H, "$H$ (A/m)",
             ha="center", va="center", rotation=90, fontsize=FS)
    fig.text((PAD_L + 0.5 * PANEL_W) / W, 0.30 * PAD_B / H, r"Time ($\mu$s)",
             ha="center", va="center", fontsize=FS)

    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{STEM}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    if args.window_end <= args.window_start:
        raise ValueError("--window-end must be greater than --window-start")

    if args.plot_only_cache is not None:
        cache_path = args.plot_only_cache if args.plot_only_cache.is_absolute() else ROOT / args.plot_only_cache
        arrays, metadata = load_plot_cache(cache_path)
        b_mt, h_am = arrays["b_mt"], arrays["h_am"]
        window_start = int(metadata["window_start"])
        warmup_steps = int(metadata["warmup_steps"])
        temperature = float(metadata["temperature_c"])
        row = int(metadata["row"])
    else:
        b_mt, h_am, temperature, row_length = load_window(
            args.data_dir, args.row, args.window_start, args.window_end
        )
        window_start = args.window_start
        warmup_steps = args.warmup_steps
        row = args.row
        print(f"row {row} length {row_length} T={temperature:.0f} C")

    paths = draw_figure(
        output_dir=args.output_dir,
        b_mt=b_mt,
        h_am=h_am,
        window_start=window_start,
        warmup_steps=warmup_steps,
    )

    if args.plot_only_cache is None:
        metadata = {
            "stem": STEM,
            "row": row,
            "window_start": window_start,
            "window_end": args.window_end,
            "warmup_steps": warmup_steps,
            "temperature_c": temperature,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "data_dir": str(args.data_dir.resolve()),
        }
        save_plot_cache(
            PAPER_PLOT_DATA_DIR / f"{STEM}.npz",
            arrays={
                "b_mt": np.asarray(b_mt, dtype=np.float32),
                "h_am": np.asarray(h_am, dtype=np.float32),
            },
            metadata=metadata,
        )
        with (args.output_dir / f"{STEM}.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
