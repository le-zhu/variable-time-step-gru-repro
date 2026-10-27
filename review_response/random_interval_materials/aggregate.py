"""Aggregate the ten-material random-grid summary into the response-letter numbers.

Every material x frequency-group cell of ``random_interval_materials_summary.csv``
is one row. The letter's conventions are:

* per-material and overall errors are sequence-weighted means of the cell
  means (weight = ``row_count``), in percent;
* the per-material and overall ratio is weighted random / weighted original;
* the overall P95 is the sequence-weighted mean of the cell P95 values;
* per-cell ratios are cell random / cell original, summarized by min, max,
  median, mean, and the number of cells below 1;
* the per-factor ratio is the plain mean of the cell ratios sharing a training
  downsampling factor;
* the step fraction is the sequence-weighted retained length over the
  sequence-weighted original length (first seed's grid).

    python -m review_response.random_interval_materials.aggregate `
      --summary results\\random_interval_materials\\random_interval_materials_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MATERIALS = ("N87", "N49", "3C94", "3E6", "78", "3F4", "77", "N30", "N27", "3C90")
FACTORS = (2, 3, 4, 6, 9)


def load_cells(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cells = []
    for row in rows:
        cells.append({
            "material": row["material"],
            "frequency_khz": int(float(row["frequency_khz"])),
            "factor": int(row["training_downsampling_factor"]),
            "rows": int(row["row_count"]),
            "original_length": int(row["original_length"]),
            "retained_length": float(row["retained_length_mean"]),
            "retention_ratio": float(row["retention_ratio_mean"]),
            "random": float(row["mean_err"]),
            "random_std": float(row["mean_err_std"]),
            "random_p95": float(row["p95_err"]),
            "original": float(row["original_mean_err"]),
            "original_p95": float(row["original_p95_err"]),
        })
    return cells


def weighted(cells, key):
    weights = np.array([cell["rows"] for cell in cells], dtype=np.float64)
    values = np.array([cell[key] for cell in cells], dtype=np.float64)
    return float((weights * values).sum() / weights.sum())


def aggregate(cells):
    per_material = []
    order = [m for m in MATERIALS if any(c["material"] == m for c in cells)]
    order += sorted({c["material"] for c in cells} - set(order))
    for material in order:
        group = [c for c in cells if c["material"] == material]
        random_mean, original_mean = weighted(group, "random"), weighted(group, "original")
        per_material.append({
            "material": material,
            "groups": len(group),
            "sequences": sum(c["rows"] for c in group),
            "original_pct": 100 * original_mean,
            "random_pct": 100 * random_mean,
            "ratio": random_mean / original_mean,
        })

    ratios = np.array([c["random"] / c["original"] for c in cells], dtype=np.float64)
    material_ratios = [row["ratio"] for row in per_material]
    lowest = min(per_material, key=lambda row: row["ratio"])
    highest = max(per_material, key=lambda row: row["ratio"])
    random_all, original_all = weighted(cells, "random"), weighted(cells, "original")
    n87 = {c["frequency_khz"]: c for c in cells if c["material"] == "N87"}
    weights = np.array([c["rows"] for c in cells], dtype=np.float64)
    return {
        "per_material": per_material,
        "total": {
            "cells": len(cells),
            "sequences": int(weights.sum()),
            "original_pct": 100 * original_all,
            "random_pct": 100 * random_all,
            "ratio": random_all / original_all,
            "original_p95_pct": 100 * weighted(cells, "original_p95"),
            "random_p95_pct": 100 * weighted(cells, "random_p95"),
        },
        "cell_ratio": {
            "min": float(ratios.min()),
            "max": float(ratios.max()),
            "median": float(np.median(ratios)),
            "mean": float(ratios.mean()),
            "below_one": int((ratios < 1).sum()),
        },
        "material_ratio": {
            "min": lowest["ratio"], "min_material": lowest["material"],
            "max": highest["ratio"], "max_material": highest["material"],
            "below_one": int(sum(ratio < 1 for ratio in material_ratios)),
            "materials": len(material_ratios),
        },
        "factor_mean_ratio": {
            str(factor): {
                "cells": int(sum(c["factor"] == factor for c in cells)),
                "mean_ratio": float(np.mean([c["random"] / c["original"] for c in cells
                                             if c["factor"] == factor])),
            }
            for factor in FACTORS if any(c["factor"] == factor for c in cells)
        },
        "step_fraction": {
            "weighted_retained_over_original": float(
                (weights * np.array([c["retained_length"] for c in cells])).sum()
                / (weights * np.array([c["original_length"] for c in cells])).sum()
            ),
            "mean_cell_retention_ratio": float(np.mean([c["retention_ratio"] for c in cells])),
        },
        "n87_by_frequency": {
            str(khz): {"random_pct": 100 * n87[khz]["random"], "random_std_pct": 100 * n87[khz]["random_std"],
                       "original_pct": 100 * n87[khz]["original"]}
            for khz in sorted(n87, reverse=True)
        },
    }


def letter_lines(result):
    """The values as the letter prints them."""
    lines = ["material groups sequences original random ratio"]
    for row in result["per_material"]:
        lines.append(f"{row['material']} {row['groups']} {row['sequences']} "
                     f"{row['original_pct']:.3f} {row['random_pct']:.3f} {row['ratio']:.3f}")
    total = result["total"]
    lines.append(f"total {total['cells']} {total['sequences']} {total['original_pct']:.3f} "
                 f"{total['random_pct']:.3f} {total['ratio']:.3f}")
    lines.append(f"P95 random {total['random_p95_pct']:.3f} original {total['original_p95_pct']:.3f}")
    cell = result["cell_ratio"]
    lines.append(f"cell ratio {cell['min']:.3f}-{cell['max']:.3f} median {cell['median']:.3f} "
                 f"mean {cell['mean']:.3f} below1 {cell['below_one']}/{total['cells']}")
    mat = result["material_ratio"]
    lines.append(f"material ratio {mat['min']:.3f} ({mat['min_material']}) - {mat['max']:.3f} "
                 f"({mat['max_material']}) below1 {mat['below_one']}/{mat['materials']}")
    lines.append("factor mean ratio " + ", ".join(
        f"{factor}: {value['mean_ratio']:.3f} ({value['cells']} cells)"
        for factor, value in result["factor_mean_ratio"].items()))
    step = result["step_fraction"]
    lines.append(f"step fraction {step['weighted_retained_over_original']:.3f} (weighted), "
                 f"{step['mean_cell_retention_ratio']:.3f} (mean cell)")
    lines.append("N87 " + ", ".join(
        f"{khz} kHz random {v['random_pct']:.2f} original {v['original_pct']:.2f}"
        for khz, v in result["n87_by_frequency"].items()))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True,
                        help="random_interval_materials_summary.csv written by run_materials.py.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Aggregate JSON; default is aggregate.json next to the summary.")
    args = parser.parse_args()

    result = aggregate(load_cells(args.summary))
    output = args.output or args.summary.with_name("aggregate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\n".join(letter_lines(result)))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
