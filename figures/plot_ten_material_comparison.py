"""Ten-material sequence error of this work beside two recent transient models.

Every value is the figure its own authors report for their own model; none is a
re-implementation.  The three works do not share a protocol, so the figure is a
side-by-side reading of published numbers rather than a ranking, and the caption
in the manuscript carries that qualification.

Materials run down a shared vertical axis so the two metrics can keep separate
horizontal scales: the P95 spread of PI-HNO is three times ours, and a common
scale would compress the whole mean panel into the left margin.  Row order
follows the manuscript's cross-material table, ascending in our own test mean.

Sources
  This work    manuscript cross-material table, test mean and test P95
  LSTM-DAO     Y. Wang et al., IEEE Trans. Power Electron., 2026,
               doi:10.1109/TPEL.2026.3715566, Table III, 4-40-16 model
  PI-HNO       Y. Zhu et al., arXiv:2608.02965, Table II, sequence-error columns
"""

from pathlib import Path

import matplotlib.pyplot as plt

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import apply_paper_plot_style, save_paper_figure


STEM = "ten_material_error_comparison"
W, H = figure_size(STEM)

FS = 8.0

# material: (ours mean, ours P95, LSTM-DAO mean, LSTM-DAO P95, PI-HNO mean, PI-HNO P95)
MATERIALS = (
    ("3F4",  3.215,  6.057, 1.82,  4.94,  9.84, 24.39),
    ("N87",  3.389,  5.457, 2.46,  5.32, 17.45, 44.24),
    ("N27",  4.381,  7.461, 3.05,  6.62, 14.94, 32.35),
    ("N49",  4.526,  8.256, 2.62,  5.82, 12.61, 27.70),
    ("78",   5.006,  8.473, 3.38,  8.34, 13.54, 30.32),
    ("77",   4.769,  7.980, 3.68,  8.61, 14.01, 31.79),
    ("3C94", 5.373,  9.544, 3.07,  7.30, 13.51, 39.07),
    ("3C90", 5.485, 10.034, 3.29,  7.90, 10.72, 27.84),
    ("N30",  5.738, 10.566, 3.54,  7.61, 20.41, 47.97),
    ("3E6",  8.122, 14.748, 5.30, 11.04, 17.37, 36.45),
)

SERIES = (
    ("This work (RateGRU), 15 873 par.", "#1f77b4", "o", 22.0),
    ("LSTM-DAO, 8600 par.",                   "#d62728", "s", 19.0),
    ("PI-HNO, 4777 par.",                     "#7f7f7f", "^", 22.0),
)

PANELS = (("(a) Mean sequence error", (0, 2, 4)), ("(b) P95 sequence error", (1, 3, 5)))

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"


def make_figure(output_dir=None):
    apply_paper_plot_style(font_size=FS, extra={"svg.hashsalt": STEM})

    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    # The right margin has to clear half of the last tick label: the P95 panel
    # autoscales to a tick at 50 and at right=0.995 the "50" fell off the page.
    fig.subplots_adjust(left=0.062, right=0.978, bottom=0.235, top=0.925, wspace=0.16)

    rows = list(range(len(MATERIALS)))[::-1]  # first table row at the top

    for ax, (title, columns) in zip(axes, PANELS):
        for row, entry in zip(rows, MATERIALS):
            # Spans the three points rather than a fixed pair: LSTM-DAO falls to
            # the left of this work on the mean panel and to the right of it on
            # parts of the P95 panel, so no fixed pair brackets every row.
            values = [entry[1 + column] for column in columns]
            ax.plot([min(values), max(values)], [row, row], color="#c8c8c8",
                    linewidth=0.7, zorder=1, solid_capstyle="round")
        for column, (label, color, marker, area) in zip(columns, SERIES):
            ax.scatter(
                [entry[1 + column] for entry in MATERIALS], rows,
                s=area, marker=marker, facecolor=color, edgecolor="white",
                linewidth=0.5, zorder=3, label=label,
            )

        ax.set_yticks(rows)
        ax.set_yticklabels([entry[0] for entry in MATERIALS])
        ax.set_ylim(-0.7, len(MATERIALS) - 0.3)
        ax.set_xlim(left=0)
        ax.set_xlabel("Sequence error (%)", fontsize=FS)
        ax.set_title(title, fontsize=FS, pad=4)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, -0.012), handletextpad=0.35, columnspacing=1.6)

    outputs = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_ten_material_comparison.py",
    )
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for path in make_figure():
        print(f"[figure] output={path}")
