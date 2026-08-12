"""Three regimes of the hidden-state update coefficient.

Section II-C holds u_t and h_tilde_t fixed and rearranges the state update into
h_{t+dt} - h_tilde = (1 - dt*u_t) * (h_t - h_tilde), a scalar linear recursion
whose behaviour is set entirely by the contraction factor |1 - dt*u_t|.  The
left panel iterates that recursion for one value of dt*u_t from each regime;
the right panel plots the contraction factor itself, so the thresholds at 1 and
2 that separate the regimes are read off directly.

The layout is specified in inches at the final printed size: each panel
occupies exactly half of the column and its axes box is square, so the 8 pt
text is 8 pt on the page and no scaling happens at \\includegraphics time.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import apply_paper_plot_style, save_paper_figure


STEM = "update_regimes"
W, H = figure_size(STEM)
HALF = W / 2.0             # one panel slot

PAD_L = 0.280              # y label + tick labels, per panel
PAD_B = 0.375              # x tick labels + x label
SLACK_R = 0.050            # keeps the axes box off the slot boundary
SIDE = HALF - PAD_L - SLACK_R          # square axes box

C_MONO = "#2ca02c"     # 0 < dt*u < 1
C_OSC = "#ff7f0e"      # 1 < dt*u < 2
C_DIV = "#d62728"      # dt*u > 2

H0 = 0.2               # initial hidden-state component
HT = 0.8               # candidate state, held fixed by the local approximation
K = 6                  # update steps shown

CHROME_LW = 0.6
CURVE_LW = 0.9
FS = 8.0


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"

def trajectory(alpha, k):
    steps = np.arange(k + 1)
    return HT + (1.0 - alpha) ** steps * (H0 - HT)


def make_figure(output_dir=None):
    apply_paper_plot_style(font_size=FS, extra={"svg.hashsalt": STEM})

    fig = plt.figure(figsize=(W, H), dpi=300)
    ax_a = fig.add_axes([PAD_L / W, PAD_B / H, SIDE / W, SIDE / H])
    ax_b = fig.add_axes([(HALF + PAD_L) / W, PAD_B / H, SIDE / W, SIDE / H])

    steps = np.arange(K + 1)

    # ---------------- left: iterated state trajectories ----------------
    ax_a.axhline(HT, color="black", linestyle=(0, (4, 3)), linewidth=CHROME_LW, zorder=1)
    # sits left of where the rising curves cross this height
    ax_a.text(-0.12, HT + 0.04, r"$\tilde{h}_t$", ha="left", va="bottom", fontsize=FS)

    for alpha, color, marker in (
        (0.5, C_MONO, "o"),
        (1.5, C_OSC, "s"),
        (2.1, C_DIV, "^"),
    ):
        ax_a.plot(
            steps,
            trajectory(alpha, K),
            color=color,
            linewidth=CURVE_LW,
            marker=marker,
            markersize=2.6,
            markeredgewidth=0.0,
            label=f"{alpha}",
            zorder=3,
        )

    ax_a.set_xlim(-0.25, K + 0.25)
    ax_a.set_ylim(-0.55, 2.62)
    ax_a.set_xticks([0, 2, 4, 6])
    ax_a.set_yticks([0, 1, 2])
    ax_a.set_xlabel(r"update step $k$", fontsize=FS, labelpad=1.5)
    ax_a.set_ylabel(r"$h_k$", fontsize=FS, labelpad=1.5)
    leg = ax_a.legend(
        title=r"$\mathrm{d}t\cdot u_t$",
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=FS,
        title_fontsize=FS,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.55,
        borderpad=0.0,
        labelspacing=0.15,
        borderaxespad=0.08,
    )
    leg._legend_box.align = "center"

    # ---------------- right: contraction factor ----------------
    z = np.linspace(0.0, 2.5, 400)
    ax_b.axvspan(0.0, 1.0, color=C_MONO, alpha=0.16, linewidth=0, zorder=0)
    ax_b.axvspan(1.0, 2.0, color=C_OSC, alpha=0.16, linewidth=0, zorder=0)
    ax_b.axvspan(2.0, 2.5, color=C_DIV, alpha=0.16, linewidth=0, zorder=0)
    ax_b.axhline(1.0, color="black", linestyle=(0, (4, 3)), linewidth=CHROME_LW, zorder=1)
    ax_b.plot(z, np.abs(1.0 - z), color="black", linewidth=CURVE_LW, zorder=3)

    ax_b.set_xlim(0.0, 2.5)
    ax_b.set_ylim(0.0, 1.62)
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_yticks([0, 1])
    for xr, yr, text, color in (
        (0.13, 0.40, "monotone", C_MONO),
        (1.13, 0.80, "oscillatory", C_OSC),
        (2.14, 0.52, "divergent", C_DIV),
    ):
        ax_b.text(xr, yr, text, rotation=90, ha="left", va="center",
                  fontsize=FS, color=color, zorder=4)

    ax_b.set_xlabel(r"$\mathrm{d}t\cdot u_t$", fontsize=FS, labelpad=1.5)
    ax_b.set_ylabel(r"$|1-\mathrm{d}t\cdot u_t|$", fontsize=FS, labelpad=1.5)

    for ax in (ax_a, ax_b):
        ax.tick_params(axis="both", which="major", labelsize=FS,
                       width=CHROME_LW, length=2.2, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(CHROME_LW)

    paths = []
    paths = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_update_regimes.py",
    )
    plt.close(fig)

    for p in paths:
        print(p)
    return paths


if __name__ == "__main__":
    make_figure(sys.argv[1] if len(sys.argv) > 1 else None)
