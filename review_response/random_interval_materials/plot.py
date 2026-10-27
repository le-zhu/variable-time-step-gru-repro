"""Response-letter figure: random-grid over original-grid error per cell.

Each point is one material x frequency-group cell of the ten-material summary;
the y value is the cell's mean error on the random nonuniform grid divided by
its mean error on the original measured grid (stride 1), the baseline of the
manuscript's nonuniform-time-step subsection. The dashed line marks equal error.

    python -m review_response.random_interval_materials.plot `
      --summary results\\random_interval_materials\\random_interval_materials_summary.csv
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from figures.paper_figure_sizes import COLUMNWIDTH_IN, figure_scale, marker_area  # noqa: E402
from figures.paper_plot_style import paper_matplotlib_rc, save_paper_figure  # noqa: E402
from review_response.random_interval_materials.aggregate import MATERIALS, load_cells  # noqa: E402


FREQUENCIES = (320, 200, 125, 80, 50)
COLORS = dict(zip(MATERIALS, plt.get_cmap("tab10").colors))
STEM = "named_materials_random_interval_vs_original"
FIGURE_SIZE = (COLUMNWIDTH_IN, 2.05)
# Marker sizes follow the column-width sampling figure of the manuscript.
SIZE_REFERENCE = "n87_sampling_downsampling_low2"
# A small horizontal offset per material keeps overlapping points readable.
JITTER = {material: (index - 4.5) * 0.022 for index, material in enumerate(MATERIALS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "random_interval_materials")
    args = parser.parse_args()

    points = [cell for cell in load_cells(args.summary) if cell["frequency_khz"] in FREQUENCIES]
    unexpected = sorted({cell["material"] for cell in points} - set(MATERIALS))
    if unexpected:
        raise ValueError(f"Unexpected materials in summary: {unexpected}")

    plt.rcParams.update(paper_matplotlib_rc(font_size=8))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    positions = {frequency: index for index, frequency in enumerate(FREQUENCIES)}
    for cell in points:
        ax.scatter(
            positions[cell["frequency_khz"]] + JITTER[cell["material"]],
            cell["random"] / cell["original"],
            s=marker_area(SIZE_REFERENCE, 18.0), marker="o",
            facecolors=COLORS[cell["material"]], edgecolors="none", alpha=0.9, zorder=3,
        )
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(FREQUENCIES)))
    ax.set_xticklabels([str(frequency) for frequency in FREQUENCIES])
    ax.set_xlabel("Excitation frequency (kHz)")
    ax.set_ylabel("Random-grid error / original-grid error")
    ax.set_ylim(0.6, 1.5)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticklabels(["0.6", "0.8", "1.0", "1.2", "1.4"])
    ax.set_xlim(-0.55, len(FREQUENCIES) - 0.45)
    ax.grid(True, axis="y", color="#dcdcdc", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=COLORS[material],
                   markersize=4 * figure_scale(SIZE_REFERENCE), label=material)
        for material in MATERIALS
    ]
    ax.legend(handles=handles, frameon=False, ncol=5, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), columnspacing=1.1, handletextpad=0.3)
    fig.tight_layout()
    paths = save_paper_figure(fig, args.output_dir, STEM, creator="response-letter-figure")
    plt.close(fig)
    for path in paths:
        print(f"saved {path}")
    print(f"points={len(points)} materials={len({cell['material'] for cell in points})}")


if __name__ == "__main__":
    main()
