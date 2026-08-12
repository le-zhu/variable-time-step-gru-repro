from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_style import apply_paper_plot_style, save_paper_figure


# The sampling-diagnostics figure is drawn at roughly half this figure's
# figure_scale, so reusing its nominal stroke arguments would render the axes
# frame about twice as thick.  Take its *rendered* widths instead.
CHROME_REFERENCE_STEM = "n87_sampling_downsampling_low2"

STEM = "limiter_function_comparison"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"

def hardcut(z: np.ndarray) -> np.ndarray:
    return np.clip(z, 0.0, 1.0)


def expcap(z: np.ndarray) -> np.ndarray:
    return np.where(z <= 0.8, z, 1.0 - 0.2 * np.exp(-(z - 0.8) / 0.2))


def globalexp(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-z)


def make_figure(output_dir: str | Path | None = None) -> tuple[Path, ...]:
    z = np.linspace(0.0, 2.0, 600)
    expcap_y = expcap(z)
    hardcut_y = hardcut(z)
    globalexp_y = globalexp(z)

    apply_paper_plot_style(extra={"svg.hashsalt": STEM})

    fig, ax = plt.subplots(figsize=figure_size(STEM), dpi=300)
    # Same red/green/blue as the sampling-diagnostics figure.
    expcap_color = "#1f77b4"
    hardcut_color = "#d62728"
    globalexp_color = "#2ca02c"
    curve_width = line_width(STEM, 1.15)
    chrome_width = line_width(CHROME_REFERENCE_STEM, 0.8)
    chrome_length = marker_size(CHROME_REFERENCE_STEM, 3.5)
    guide_width = line_width(CHROME_REFERENCE_STEM, 1.0)

    ax.plot(z, expcap_y, color=expcap_color, linewidth=curve_width, label="expcap")
    ax.plot(z, hardcut_y, color=hardcut_color, linewidth=curve_width, linestyle="--", label="hardcut")
    ax.plot(
        z,
        globalexp_y,
        color=globalexp_color,
        linewidth=curve_width,
        linestyle=(0, (5, 1.6)),
        label="globalexp",
    )

    ax.axvline(0.8, color=expcap_color, linestyle=(0, (3, 2)), linewidth=guide_width, alpha=0.75)
    ax.axvline(1.0, color=hardcut_color, linestyle=(0, (3, 2)), linewidth=guide_width, alpha=0.75)

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.8, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.8", "1.0", "1.5", "2.0"])
    ax.set_yticks([0.0, 0.5, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.5", "0.8", "1.0"])
    ax.set_xlabel("Input", fontsize=8)
    ax.set_ylabel("Output", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8, width=chrome_width, length=chrome_length)
    ax.grid(True, color="#d9d9d9", linewidth=chrome_width)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    for spine in ax.spines.values():
        spine.set_linewidth(chrome_width)

    fig.tight_layout(pad=0.25)

    paths = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_limiter_function_comparison.py",
    )
    plt.close(fig)

    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    make_figure()
