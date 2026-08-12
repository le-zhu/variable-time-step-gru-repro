from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.paper_figure_sizes import COLUMNWIDTH_IN
from figures.paper_plot_style import paper_matplotlib_rc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "n87_update_coefficients" / "sampling_update_coefficient_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "n87_update_coeff_downsampling"
COMPACT_FIGURE_DIR = ROOT / "figures" / "output"
STEM = "update_coeff_downsampling_diagnostics"
GEOMETRY_SCALE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the compact-paper single-column update-coefficient figure.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser.parse_args()


def read_downsampling_rows(path: Path, model_label: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model_label"] != model_label or row["sampling"] != "downsample":
                continue
            rows.append(
                {
                    "factor": float(row["factor"]),
                    "effective_mean": float(row["effective_mean"]),
                    "effective_median": float(row["effective_median"]),
                    "ratio": float(
                        row["effective_gt_1_ratio"]
                        if model_label == "DtScaledGRU"
                        else row["sdt_gt_0p8_ratio"]
                    ),
                }
            )
    rows.sort(key=lambda item: item["factor"])
    if not rows:
        raise ValueError(f"No downsampling rows found for {model_label} in {path}")
    return rows


def configure_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.2, 50.8)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.04, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, color="#d9d9d9", linewidth=0.38)
    ax.set_box_aspect(1.0)
    ax.tick_params(
        axis="both",
        labelsize=8,
        width=0.8 * GEOMETRY_SCALE,
        length=3.5 * GEOMETRY_SCALE,
        pad=1.0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8 * GEOMETRY_SCALE)


def draw_panel(ax: plt.Axes, rows: list[dict[str, float]], title: str) -> None:
    factors = [row["factor"] for row in rows]

    mean_line = ax.plot(
        factors,
        [row["effective_mean"] for row in rows],
        color="#1f77b4",
        marker="^",
        linewidth=1.5 * GEOMETRY_SCALE,
        markersize=4.8 * GEOMETRY_SCALE,
        markeredgewidth=0.8 * GEOMETRY_SCALE,
        label="Mean",
    )[0]
    median_line = ax.plot(
        factors,
        [row["effective_median"] for row in rows],
        color="#FFD400",
        marker="D",
        linewidth=1.4 * GEOMETRY_SCALE,
        markersize=4.4 * GEOMETRY_SCALE,
        markeredgewidth=0.8 * GEOMETRY_SCALE,
        label="Median",
    )[0]
    ratio_line = ax.plot(
        factors,
        [row["ratio"] for row in rows],
        color="#d62728",
        marker="v",
        linestyle=":",
        linewidth=1.35 * GEOMETRY_SCALE,
        markersize=4.6 * GEOMETRY_SCALE,
        markeredgewidth=0.8 * GEOMETRY_SCALE,
        label="Fraction",
    )[0]

    configure_axis(ax)
    ax.set_title(title, fontsize=8, pad=1.2)
    ax.set_xlabel("Downsampling factor", fontsize=8, labelpad=1.0)
    ax.set_ylabel("Value", fontsize=8, labelpad=1.0)
    ax.legend(
        [mean_line, median_line, ratio_line],
        ["Mean", "Median", "Ratio"],
        loc="upper left",
        frameon=False,
        fontsize=8,
        handlelength=0.9,
        handletextpad=0.25,
        labelspacing=0.02,
        borderaxespad=0.08,
    )


def main() -> None:
    args = parse_args()
    summary = args.summary if args.summary.is_absolute() else ROOT / args.summary
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dtscaled_rows = read_downsampling_rows(summary, "DtScaledGRU")
    rategru_rows = read_downsampling_rows(summary, "RateGRU")

    width = COLUMNWIDTH_IN
    height = 1.72
    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        fig, axes = plt.subplots(1, 2, figsize=(width, height), dpi=300)
        draw_panel(axes[0], dtscaled_rows, "DtScaledGRU")
        draw_panel(axes[1], rategru_rows, "RateGRU")
        # The preceding diagnostic figure used square coordinate boxes. Keep
        # that 1:1 axes ratio while retaining 8 pt text on a single-column page.
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.95, wspace=0.34)
        fig.canvas.draw()
        axes_aspects = []
        for ax in axes:
            box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            axes_aspects.append(box.width / box.height)

        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.0}
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(output_dir / f"{STEM}.{suffix}", **save_kwargs)
        if not args.no_paper_copy:
            COMPACT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(COMPACT_FIGURE_DIR / f"{STEM}.pdf", **save_kwargs)
        plt.close(fig)

    print(f"[figure] source={summary}")
    print(f"[figure] size={width:.6f} x {height:.6f} in")
    print(f"[figure] axes_width_to_height={[round(value, 6) for value in axes_aspects]}")
    print(f"[figure] output={output_dir / f'{STEM}.pdf'}")


if __name__ == "__main__":
    main()
