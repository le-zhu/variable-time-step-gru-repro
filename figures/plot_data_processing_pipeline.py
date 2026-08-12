"""Data processing pipeline from the raw MagNetX sequences to the model input.

The chain is drawn top to bottom in the order the preprocessor executes it: the
sequence-level split comes first so that no later statistic can see the held-out
data, resampling follows, and the normalization statistics are taken from the
training split alone.  Reading order down the spine carries the sequence, so the
figure needs no arrowheads; the parenthetical lines carry the two facts that the
order alone does not, namely which sequences each stage touches.

The layout is specified in inches at the final printed size, so the 8 pt text is
8 pt on the page and \\includegraphics[width=\\columnwidth] does not scale.
"""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import (
    PAPER_FONT_FAMILY,
    apply_paper_plot_style,
    save_paper_figure,
)


STEM = "data_processing_pipeline"
W, H = figure_size(STEM)

ACCENT = "#1f77b4"             # spine and stage nodes
TITLE_COLOR = "#111111"
DETAIL_COLOR = "#4a4a4a"

FS = 8.0
FS_DETAIL = 7.5

X_SPINE = 0.145                # vertical rail through the stage nodes
NODE_R = 0.042
X_L, X_R = 0.31, W - 0.06      # stage panels, clear of the spine
X_TEXT = X_L + 0.115           # left-aligned text column

TITLE_H = 0.140                # baseline pitch of the stage name
DETAIL_H = 0.125               # baseline pitch of one detail line
PAD_H = 0.065
GAP = 0.075
MIN_H = 0.270
Y_TOP = H - 0.055              # top edge of the first panel


class Style(NamedTuple):
    face: str
    edge: str
    node: str


# Data entering or leaving the pipeline is drawn as an outlined white panel with
# an open node; an operation is drawn as a filled panel with a filled node.
DATA = Style(face="#ffffff", edge=TITLE_COLOR, node="#ffffff")
STEP = Style(face="#eef2f6", edge=ACCENT, node=ACCENT)


class Stage(NamedTuple):
    style: Style
    title: str
    details: tuple[str, ...] = ()


STAGES = (
    Stage(DATA, "MagNetX sequences", ("sampled at 16 MHz",)),
    Stage(STEP, "Sequence-level split", ("80% / 10% / 10%",)),
    Stage(STEP, "Resampling", ("(training set and resampled test set only)",)),
    Stage(STEP, "Forward differences"),
    Stage(STEP, "Normalization", (
        "(statistics from the training set only)",
        r"$B/\mathrm{R}(B)$, $H/\mathrm{R}(H)$; Max-step or RMS-slope",
    )),
    Stage(DATA, "Model input", (
        r"$\mathbf{x}_t=[\,B_t,\ (\mathrm{d}B/\mathrm{d}t)_t,\ H_t,\ T\,]^{\top}$ and $\mathrm{d}t_t$",
    )),
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"


def stage_boxes():
    """Bottom and top edge of every stage panel, in figure inches."""
    boxes = []
    top = Y_TOP
    for stage in STAGES:
        height = max(MIN_H, 2 * PAD_H + TITLE_H + DETAIL_H * len(stage.details))
        boxes.append((top - height, top))
        top -= height + GAP
    return boxes


def make_figure(output_dir=None):
    apply_paper_plot_style(
        font_size=FS,
        extra={"svg.hashsalt": STEM, "mathtext.bf": f"{PAPER_FONT_FAMILY}:italic:bold"},
    )

    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    boxes = stage_boxes()
    centres = [0.5 * (bottom + top) for bottom, top in boxes]
    ax.plot([X_SPINE, X_SPINE], [centres[0], centres[-1]], color=ACCENT,
            linewidth=0.9, solid_capstyle="round", zorder=1)

    for stage, (bottom, top), centre in zip(STAGES, boxes, centres):
        ax.add_patch(
            FancyBboxPatch(
                (X_L, bottom),
                X_R - X_L,
                top - bottom,
                boxstyle="round,pad=0,rounding_size=0.045",
                facecolor=stage.style.face,
                edgecolor=stage.style.edge,
                linewidth=0.6,
                zorder=2,
            )
        )
        ax.add_patch(
            Circle((X_SPINE, centre), NODE_R, facecolor=stage.style.node,
                   edgecolor=ACCENT, linewidth=0.9, zorder=4)
        )

        y_title = top - PAD_H - 0.5 * TITLE_H
        ax.text(X_TEXT, y_title, stage.title, ha="left", va="center", fontsize=FS,
                color=TITLE_COLOR, fontweight="bold")
        for line, detail in enumerate(stage.details):
            y = y_title - 0.5 * TITLE_H - DETAIL_H * (line + 0.5)
            ax.text(X_TEXT, y, detail, ha="left", va="center", fontsize=FS_DETAIL,
                    color=DETAIL_COLOR)

    outputs = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_data_processing_pipeline.py",
    )
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for path in make_figure():
        print(f"[figure] output={path}")
