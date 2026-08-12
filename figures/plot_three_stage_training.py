"""Three-stage training schedule: teacher forcing, curriculum, autoregressive.

The figure is a map rather than a block chart.  The horizontal axis is the
training epoch and the vertical axis is the position inside a training segment,
so the shaded area is exactly the set of steps that receive the measured
$H^{\\mathrm{true}}$ as input.  The teacher-forcing boundary $L_{\\mathrm{TF}}(e)$
is the upper edge of that area: constant at $L_{12}$ through the first stage,
falling linearly to the warmup length $L_{\\mathrm{w}}$ through the second, and
held there while the segment length steps up to $L_3$ for the third.

The layout is specified in inches at the final printed size, so the 8 pt text is
8 pt on the page and \\includegraphics[width=\\columnwidth] does not scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import (
    PAPER_FONT_FAMILY,
    apply_paper_plot_style,
    save_paper_figure,
)


STEM = "three_stage_training"
W, H = figure_size(STEM)

# Table I of the manuscript.
L12 = 100.0       # stage-1/2 segment length
L3 = 1000.0       # stage-3 segment length
LW = 10.0         # training warmup length
E1 = 50           # stage-1 epochs
E2 = 50           # stage-2 epochs
E3 = 150          # maximum stage-3 epochs

TF_FACE = "#cfe0f2"        # steps driven by the measured H
TF_EDGE = "#1f77b4"
SEG_FACE = "#f1f1f1"       # steps driven by the model's own prediction
SEG_EDGE = "#3f3f3f"
GUIDE = "#8c8c8c"

LW_EDGE = 0.7
FS = 8.0
FS_SMALL = 7.0

Y_BOTTOM = 1.0
Y_TOP = 2200.0

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"


def teacher_forcing_boundary(epoch):
    """L_TF as a function of the global epoch, following (2)."""
    epoch = np.asarray(epoch, dtype=float)
    stage2 = LW + (L12 - LW) * (1.0 - (epoch - E1) / E2)
    return np.where(epoch <= E1, L12, np.where(epoch <= E1 + E2, stage2, LW))


def make_figure(output_dir=None):
    apply_paper_plot_style(
        font_size=FS,
        extra={"svg.hashsalt": STEM, "mathtext.bf": f"{PAPER_FONT_FAMILY}:italic:bold"},
    )

    total = E1 + E2 + E3
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes(
        [0.44 / W, 0.33 / H, (W - 0.44 - 0.10) / W, (H - 0.33 - 0.30) / H]
    )
    ax.set_xlim(0, total)
    ax.set_yscale("log")
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.set_xticks([0, E1, E1 + E2, 150, 200, total])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("position in training segment")
    ax.tick_params(direction="in", length=2.2, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # The segment itself: length L12 while the stages are short-segment, L3 once
    # the long-sequence fine-tuning starts.
    ax.add_patch(
        Polygon(
            [(0, Y_BOTTOM), (0, L12), (E1 + E2, L12), (E1 + E2, L3),
             (total, L3), (total, Y_BOTTOM)],
            closed=True,
            facecolor=SEG_FACE,
            edgecolor=SEG_EDGE,
            linewidth=LW_EDGE,
            zorder=2,
        )
    )

    # The teacher-forced part of that segment.
    epochs = np.linspace(0, total, 600)
    boundary = teacher_forcing_boundary(epochs)
    vertices = [(0, Y_BOTTOM)]
    vertices += list(zip(epochs, boundary))
    vertices += [(total, Y_BOTTOM)]
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=TF_FACE,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.plot(epochs, boundary, color=TF_EDGE, linewidth=1.0, zorder=5,
            solid_capstyle="butt")

    for boundary_epoch in (E1, E1 + E2):
        ax.axvline(boundary_epoch, color=GUIDE, linewidth=0.6, linestyle=(0, (2.4, 1.8)),
                   zorder=4)

    transform = ax.get_xaxis_transform()
    for centre, label in ((0.5 * E1, "Stage 1"), (E1 + 0.5 * E2, "Stage 2"),
                          (E1 + E2 + 0.5 * E3, "Stage 3")):
        ax.text(centre, 1.035, label, transform=transform, ha="center", va="bottom",
                fontsize=FS)

    ax.text(0.5 * total, 3.0, r"teacher-forced input $H^{\mathrm{true}}$",
            ha="center", va="center", fontsize=FS_SMALL, color="#0f4c81", zorder=6)
    # The manuscript writes the measured field as H^true and the model's own
    # prediction as plain H, so the two regions are labelled the same way.
    ax.text(E1 + E2 + 0.52 * E3, 110.0, r"autoregressive input $H$",
            ha="center", va="center", fontsize=FS_SMALL, zorder=6)

    # Both learning-rate notes sit in the blank strip above the segment, so the
    # two shaded regions carry only the teacher-forcing story.
    ax.text(0.5 * (E1 + E2), 450.0, "lr12, fixed", ha="center", va="center",
            fontsize=FS_SMALL, zorder=6)
    ax.text(E1 + E2 + 0.5 * E3, 1480.0, "lr3, decayed on plateau", ha="center",
            va="center", fontsize=FS_SMALL, zorder=6)

    ax.annotate(
        r"$L_{\mathrm{TF}}(e)$",
        xy=(E1 + 0.5 * E2, 55.0),
        xytext=(E1 + E2 + 5.0, 560.0),
        fontsize=FS_SMALL,
        color=TF_EDGE,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-|>", "mutation_scale": 5.0, "color": TF_EDGE,
                    "linewidth": 0.6, "shrinkA": 1.0, "shrinkB": 1.0},
        zorder=7,
    )

    ax.text(total - 3.0, 11.5, r"$L_{\mathrm{w}}=10$", ha="right", va="bottom",
            fontsize=FS_SMALL, color=TF_EDGE, zorder=6)
    ax.text(3.0, L12 * 1.13, r"$L_{12}=100$", ha="left", va="bottom",
            fontsize=FS_SMALL, zorder=6)
    ax.text(total - 3.0, L3 / 1.13, r"$L_{3}=1000$", ha="right", va="top",
            fontsize=FS_SMALL, zorder=6)

    outputs = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_three_stage_training.py",
    )
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for path in make_figure():
        print(f"[figure] output={path}")
