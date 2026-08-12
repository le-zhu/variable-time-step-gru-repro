from __future__ import annotations

from pathlib import Path
from typing import Any


# Matplotlib falls back to its default serif face where this font is absent,
# which changes the lettering but not the geometry of any figure.
PAPER_FONT_FAMILY = "Times New Roman"


def paper_matplotlib_rc(font_size: float = 8, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rc: dict[str, Any] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": [PAPER_FONT_FAMILY],
        "font.size": font_size,
        "mathtext.fontset": "custom",
        "mathtext.rm": PAPER_FONT_FAMILY,
        "mathtext.it": f"{PAPER_FONT_FAMILY}:italic",
        "mathtext.bf": f"{PAPER_FONT_FAMILY}:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
    if extra:
        rc.update(extra)
    return rc


def apply_paper_plot_style(font_size: float = 8, extra: dict[str, Any] | None = None) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(paper_matplotlib_rc(font_size=font_size, extra=extra))


def save_paper_figure(fig, output_dir, stem: str, *, creator: str, dpi: int = 600):
    """Write the PDF the manuscript includes plus the PNG and SVG previews.

    The metadata differs per format because the writers accept different keys:
    the SVG writer rejects a Creator entry, and pinning its Date to None is what
    keeps a redrawn figure byte-identical to the previous one.
    """
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, metadata in (
        ("png", {"Software": "Matplotlib"}),
        ("svg", {"Date": None}),
        ("pdf", {"Creator": creator}),
    ):
        path = output_path / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, metadata=metadata)
        paths.append(path)
    return tuple(paths)

