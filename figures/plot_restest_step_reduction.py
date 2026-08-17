"""Rollout cost of the resampled test set against the original-grid test set.

Inference cost of a recurrent rollout is the step count times a per-step cost
that the two runs share exactly: same weights, same architecture, same input
width.  The step ratio is therefore the compute ratio, with nothing estimated.

All three panels are the same paired bar, so the contrast carries itself: the
cost bar collapses to about a third while the two error bars stay level.  Every
panel is annotated with ResTest as a percentage of Test, so one reading rule
covers the figure.  The bars start at zero in all three panels, which is what
makes the near-equal error bars honest rather than a truncated axis.

Row order follows the manuscript's cross-material table.  N49 reduces least
because the MagNetX release has no 50 or 80 kHz sequences for it, so its test
set holds none of the stride-6 or stride-9 groups that drive the other nine.

Input is one ``--results-dir`` holding a subdirectory per material, each written
by ``scripts/evaluate_resampled_holdout.py --out-dir``.  Step totals are summed
from the per-sequence CSV; the four error values are read from the summary JSON
rather than recomputed, so the P95 here is the same statistic the evaluator
reported and cannot drift on a percentile convention.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from figures.paper_figure_sizes import figure_size
from figures.paper_plot_style import apply_paper_plot_style, save_paper_figure


STEM = "restest_step_reduction"
W, H = figure_size(STEM)

FS = 8.0
FS_NOTE = 7.0

TEST_COLOR = "#b9cfe3"
RES_COLOR = "#1f77b4"

MATERIAL_ORDER = ("3F4", "N87", "N27", "N49", "78", "77", "3C94", "3C90", "N30", "3E6")

# title, (test key, resampled key), scale, axis label, headroom for the annotation
PANELS = (
    ("(a) Rollout steps", ("raw_steps", "res_steps"), 1e-6,
     "Total rollout steps (millions)", 1.42),
    ("(b) Mean sequence error", ("test_mean", "restest_mean"), 1.0,
     "Sequence error (%)", 1.34),
    ("(c) P95 sequence error", ("test_p95", "restest_p95"), 1.0,
     "Sequence error (%)", 1.34),
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "output"


def load_material_records(results_dir):
    """Read one record per material from the evaluator's output directories."""
    records = {}
    for summary_path in sorted(Path(results_dir).rglob("resampled_holdout_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows_path = summary_path.with_name("resampled_holdout_row_errors.csv")
        if not rows_path.exists():
            raise FileNotFoundError(f"{summary_path} has no sibling {rows_path.name}")
        raw_steps = res_steps = 0
        with rows_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw_steps += int(row["raw_length"])
                res_steps += int(row["resampled_length"])
        records[summary["material"]] = {
            "raw_steps": raw_steps,
            "res_steps": res_steps,
            "test_mean": summary["heldout_mean_percent"],
            "restest_mean": summary["resampled_heldout_mean_percent"],
            "test_p95": summary["heldout_p95_percent"],
            "restest_p95": summary["resampled_heldout_p95_percent"],
        }
    missing = [name for name in MATERIAL_ORDER if name not in records]
    if missing:
        raise ValueError(
            f"No evaluation output under {results_dir} for: {', '.join(missing)}. "
            "Run scripts/evaluate_resampled_holdout.py --out-dir once per material."
        )
    return [(name, records[name]) for name in MATERIAL_ORDER]


def make_figure(records, output_dir=None):
    apply_paper_plot_style(font_size=FS, extra={"svg.hashsalt": STEM})

    fig, axes = plt.subplots(1, 3, figsize=(W, H))
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.235, top=0.918, wspace=0.13)

    rows = list(range(len(records)))[::-1]
    height = 0.34

    for ax, (title, (test_key, res_key), scale, xlabel, headroom) in zip(axes, PANELS):
        largest = max(record[test_key] for _, record in records) * scale
        for row, (_, record) in zip(rows, records):
            test, res = record[test_key] * scale, record[res_key] * scale
            ax.barh(row + 0.19, test, height=height, color=TEST_COLOR, zorder=2,
                    label="Test (original grid)" if row == rows[0] else None)
            ax.barh(row - 0.19, res, height=height, color=RES_COLOR, zorder=2,
                    label="ResTest (resampled)" if row == rows[0] else None)
            ax.text(test + 0.022 * largest, row, f"{100 * res / test:.0f}%",
                    va="center", ha="left", fontsize=FS_NOTE, color="#333333")

        ax.set_xlim(0, largest * headroom)
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_title(title, fontsize=FS, pad=4)
        ax.set_yticks(rows)
        ax.set_yticklabels([name for name, _ in records] if ax is axes[0] else [])
        ax.set_ylim(-0.72, len(records) - 0.28)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS, length=2.5, width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, -0.014), handlelength=1.3, columnspacing=2.0)

    outputs = save_paper_figure(
        fig,
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR,
        STEM,
        creator="plot_restest_step_reduction.py",
    )
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory of per-material evaluate_resampled_holdout.py outputs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    records = load_material_records(args.results_dir)
    total_raw = sum(record["raw_steps"] for _, record in records)
    total_res = sum(record["res_steps"] for _, record in records)
    print(f"Rollout steps: {total_raw:,} -> {total_res:,} "
          f"({100 * total_res / total_raw:.1f}% of the original grid)")
    for path in make_figure(records, args.output_dir):
        print(f"[figure] output={path}")


if __name__ == "__main__":
    main()
