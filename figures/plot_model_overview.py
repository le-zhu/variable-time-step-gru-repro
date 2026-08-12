"""Single-step recurrent structure of the variable-time-step GRU framework.

The figure has two parts.  The upper part is the generic single-step data flow
shared by all three models: the interval-independent input vector, the GRU
cell, the state update, the MLP prediction head, and the two feedback paths
that close the loop.  The hidden state and the predicted field strength are
carried to the next step separately: h_{t+dt} produced by the state update
returns as h_t, while H_{t+dt} produced by the head returns inside x_t.  The
lower part isolates the only place where the three models differ, namely how
the update coefficient is formed and where dt enters it.

The layout is specified in inches at the final printed size, so the 8 pt text
is 8 pt on the page and \\includegraphics[width=\\columnwidth] does not scale.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import (
    PAPER_FONT_FAMILY,
    apply_paper_plot_style,
    save_paper_figure,
)


STEM = "model_overview"
W, H = figure_size(STEM)

DT_COLOR = "#1f77b4"       # everything that carries the explicit time step
OFF_COLOR = "#8a8a8a"      # the fixed-update baseline, which has no dt
BOX_FACE = "#f2f2f2"
CELL_FACE = "#fbfbfb"

BOX_LW = 0.7
ARROW_LW = 0.7
FS = 8.0
DASH = (0, (3.5, 2.2))

# --- horizontal layout ------------------------------------------------------
X_L, X_R = 0.05, 3.44          # content margins
X_IN = 1.025                   # input-vector / head / output column
X_H = 2.54                     # hidden-state column
X_FB_H = 2.98                  # inner feedback channel, h_{t+dt} -> h_t
X_FB_BIG = 3.43                # outer feedback channel, H_{t+dt} -> x_t
CELL_R = 2.86                  # GRU cell right edge
X_MID = 1.44                   # candidate column; the state update, the head
                               # and the output box all centre on it
UPD_L, UPD_R = 0.665, 2.215    # state-update box, centred on X_MID

# --- vertical layout --------------------------------------------------------
Y_FB_BIG = 3.18                # outer feedback rail
Y_FB_H = 3.06                  # inner feedback rail
Y_IN_B, Y_IN_T = 2.74, 2.98    # input row
Y_CELL_B, Y_CELL_T = 1.38, 2.62
Y_BUS = 2.46                   # distribution bus inside the cell
Y_BLK_B, Y_BLK_T = 2.04, 2.36  # gate / candidate / rate blocks
Y_UPD_B, Y_UPD_T = 1.50, 1.84  # state-update box
Y_HEAD_B, Y_HEAD_T = 1.04, 1.26
Y_OUT_B, Y_OUT_T = 0.68, 0.90  # H_{t+dt} box
Y_VAR_B, Y_VAR_T = 0.07, 0.59  # update-coefficient panel
Y_A_IN = 1.72                  # coefficient enters the state update
Y_H_OUT = 1.60                 # h_{t+dt} leaves for the next step


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"

def _box(ax, x0, y0, x1, y1, *, face=BOX_FACE, edge="black", lw=BOX_LW, ls="-", r=0.035, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
    )


def _arrow(ax, p0, p1, *, color="black", lw=ARROW_LW, ls="-", head=5.5):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=head,
            color=color,
            linewidth=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def _line(ax, xs, ys, *, color="black", lw=ARROW_LW, ls="-", z=4):
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="butt", zorder=z)


def make_figure(output_dir=None):
    # mathtext has no \boldsymbol; pointing the bold math font at Times bold
    # italic makes \mathbf render exactly like \boldsymbol does in the text
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

    y_in_c = 0.5 * (Y_IN_B + Y_IN_T)

    # ---------------- input row ----------------
    _box(ax, X_L, Y_IN_B, 2.00, Y_IN_T)
    ax.text(X_IN, y_in_c,
            r"$\mathbf{x}_t=[\,B_t,\ (\mathrm{d}B/\mathrm{d}t)_t,\ H_t,\ T\,]^{\top}$",
            ha="center", va="center", fontsize=FS)

    _box(ax, 2.30, Y_IN_B, 2.78, Y_IN_T)
    ax.text(X_H, y_in_c, r"$\mathbf{h}_t$", ha="center", va="center", fontsize=FS)

    # ---------------- GRU cell ----------------
    _box(ax, X_L, Y_CELL_B, CELL_R, Y_CELL_T, face=CELL_FACE, r=0.05)
    ax.text(0.11, 2.55, "GRU cell", ha="left", va="center", fontsize=FS)

    # both inputs feed a common distribution bus inside the cell
    _line(ax, [X_IN, X_IN], [Y_IN_B, Y_BUS])
    _line(ax, [X_H, X_H], [Y_IN_B, Y_BUS])
    _line(ax, [0.525, X_H], [Y_BUS, Y_BUS])

    blocks = (
        (0.15, 0.90, "reset gate", r"$\mathbf{r}_t=\sigma(\cdot)$", "black"),
        (1.00, 1.88, "candidate", r"$\tilde{\mathbf{h}}_t=\tanh(\cdot)$", "black"),
        (1.98, 2.78, "gate / rate", r"$\mathbf{u}_t$  or  $\mathbf{s}_t$", DT_COLOR),
    )
    for x0, x1, title, expr, edge in blocks:
        xc = 0.5 * (x0 + x1)
        _box(ax, x0, Y_BLK_B, x1, Y_BLK_T, edge=edge)
        ax.text(xc, Y_BLK_T - 0.08, title, ha="center", va="center", fontsize=FS)
        ax.text(xc, Y_BLK_B + 0.08, expr, ha="center", va="center", fontsize=FS)
        _arrow(ax, (xc, Y_BUS), (xc, Y_BLK_T))

    _arrow(ax, (0.90, 0.5 * (Y_BLK_B + Y_BLK_T)), (1.00, 0.5 * (Y_BLK_B + Y_BLK_T)))

    # ---------------- state update ----------------
    _box(ax, UPD_L, Y_UPD_B, UPD_R, Y_UPD_T, r=0.04)
    ax.text(0.5 * (UPD_L + UPD_R), 0.5 * (Y_UPD_B + Y_UPD_T),
            r"$\mathbf{h}_{t+\mathrm{d}t}=\mathbf{h}_t"
            r"+\mathbf{\alpha}_t\odot(\tilde{\mathbf{h}}_t-\mathbf{h}_t)$",
            ha="center", va="center", fontsize=FS)

    # candidate direction and update coefficient both enter from above, which
    # keeps the coefficient path clear of the h_{t+dt} feedback on the right
    _arrow(ax, (X_MID, Y_BLK_B), (X_MID, Y_UPD_T))
    _line(ax, [2.38, 2.38], [Y_BLK_B, Y_A_IN], color=DT_COLOR)
    _arrow(ax, (2.38, Y_A_IN), (UPD_R, Y_A_IN), color=DT_COLOR)

    # ---------------- head and output ----------------
    # the head reads h_{t+dt} straight from the state update, not from the cell
    _arrow(ax, (X_MID, Y_UPD_B), (X_MID, Y_HEAD_T))

    _box(ax, 0.94, Y_HEAD_B, 1.94, Y_HEAD_T)
    ax.text(X_MID, 0.5 * (Y_HEAD_B + Y_HEAD_T), "MLP head",
            ha="center", va="center", fontsize=FS)

    _arrow(ax, (X_MID, Y_HEAD_B), (X_MID, Y_OUT_T))
    _box(ax, 1.065, Y_OUT_B, 1.815, Y_OUT_T)
    ax.text(X_MID, 0.5 * (Y_OUT_B + Y_OUT_T), r"$H_{t+\mathrm{d}t}$",
            ha="center", va="center", fontsize=FS)

    # ---------------- feedback: hidden state and field strength separately ----
    fb = dict(color=DT_COLOR, ls=DASH)
    y_out_c = 0.5 * (Y_OUT_B + Y_OUT_T)

    _line(ax, [UPD_R, X_FB_H], [Y_H_OUT, Y_H_OUT], **fb)
    _line(ax, [X_FB_H, X_FB_H], [Y_H_OUT, Y_FB_H], **fb)
    _line(ax, [X_H, X_FB_H], [Y_FB_H, Y_FB_H], **fb)
    _arrow(ax, (X_H, Y_FB_H), (X_H, Y_IN_T), color=DT_COLOR)

    _line(ax, [1.815, X_FB_BIG], [y_out_c, y_out_c], **fb)
    _line(ax, [X_FB_BIG, X_FB_BIG], [y_out_c, Y_FB_BIG], **fb)
    _line(ax, [X_IN, X_FB_BIG], [Y_FB_BIG, Y_FB_BIG], **fb)
    _arrow(ax, (X_IN, Y_FB_BIG), (X_IN, Y_IN_T), color=DT_COLOR)
    ax.text(0.5 * (X_FB_H + X_FB_BIG), 2.30, "next time step", rotation=90,
            ha="center", va="center", fontsize=FS, color=DT_COLOR)

    # ---------------- update-coefficient variants ----------------
    _box(ax, X_L, Y_VAR_B, X_R, Y_VAR_T, face="white", ls=(0, (2.5, 1.8)), lw=0.6, r=0.04, z=1)
    ax.text(0.15, 0.50,
            r"update coefficient $\mathbf{\alpha}_t$   "
            r"($\mathbf{u}_t=\sigma(\cdot)$,  $\mathbf{s}_t=\mathrm{softplus}(\cdot)$)",
            ha="left", va="center", fontsize=FS)

    rows = (
        ("FixedGRU", r"$\mathbf{\alpha}_t=\mathbf{u}_t$", OFF_COLOR),
        ("DtScaledGRU", r"$\mathbf{\alpha}_t=\mathrm{d}t\cdot\mathbf{u}_t$", "black"),
        ("RateGRU", r"$\mathbf{\alpha}_t=\mathrm{expcap}(\mathrm{d}t\cdot\mathbf{s}_t)$", DT_COLOR),
    )
    for i, (name, expr, color) in enumerate(rows):
        y = 0.38 - 0.12 * i
        ax.text(0.15, y, name, ha="left", va="center", fontsize=FS, color=color)
        ax.text(1.20, y, expr, ha="left", va="center", fontsize=FS, color=color)

    paths = []
    paths = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_model_overview.py",
    )
    plt.close(fig)

    for p in paths:
        print(p)
    return paths


if __name__ == "__main__":
    make_figure(sys.argv[1] if len(sys.argv) > 1 else None)
