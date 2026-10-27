"""Per-frequency-group Test and ResTest errors from ``evaluate_resampled_holdout.py``.

The N87 discussion of Table IV breaks the Test and ResTest means down by
frequency group (for example DtInputGRU 10.104% vs 5.453% and RateGRU 5.767% vs
5.480% in the 50 kHz group). This script regroups the per-sequence errors that
``scripts/evaluate_resampled_holdout.py --out-dir`` writes, so no model is rerun.

Each MagNetX sequence holds 100 subcycles, so ``round(raw_length / 100)`` is the
number of samples per period, which identifies the nominal frequency group.

    python scripts/evaluate_resampled_holdout.py --checkpoint n87:dtinputgru `
      --processed-data-dir <processed N87 split> --out-dir results\\table4_by_length\\dtinputgru
    python scripts/summarize_holdout_by_length.py results\\table4_by_length\\dtinputgru
"""

import argparse
import csv
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.paper_eval import load_checkpoint_entries  # noqa: E402


SAMPLE_RATE_HZ = 16_000_000.0
SUBCYCLES_PER_SEQUENCE = 100
NOMINAL_FREQUENCY_KHZ = {20: 800.0, 32: 500.0, 50: 320.0, 80: 200.0, 128: 125.0, 200: 80.0, 320: 50.0}
ROW_ERRORS_NAME = "resampled_holdout_row_errors.csv"
SUMMARY_NAME = "resampled_holdout_summary.json"
FIELDNAMES = [
    "model",
    "raw_length",
    "samples_per_period",
    "nominal_frequency_khz",
    "resample_stride",
    "n_rows",
    "test_mean_pct",
    "resampled_test_mean_pct",
    "difference_pct_points",
]


def samples_per_period(raw_length):
    return int(round(int(raw_length) / SUBCYCLES_PER_SEQUENCE))


def nominal_frequency_khz(raw_length):
    spp = samples_per_period(raw_length)
    return NOMINAL_FREQUENCY_KHZ.get(spp, SAMPLE_RATE_HZ / spp / 1000.0)


def read_row_errors(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_length(rows, model):
    """Mean Test/ResTest error (percent) of each raw length group."""
    groups = {}
    for row in rows:
        groups.setdefault(int(row["raw_length"]), []).append(row)
    out = []
    for length in sorted(groups):
        items = groups[length]
        strides = sorted({int(item["stride"]) for item in items})
        test = sum(float(item["holdout_error"]) for item in items) / len(items) * 100.0
        restest = sum(float(item["resampled_holdout_error"]) for item in items) / len(items) * 100.0
        out.append({
            "model": model,
            "raw_length": length,
            "samples_per_period": samples_per_period(length),
            "nominal_frequency_khz": nominal_frequency_khz(length),
            "resample_stride": "/".join(str(value) for value in strides),
            "n_rows": len(items),
            "test_mean_pct": test,
            "resampled_test_mean_pct": restest,
            # Rounded so that float noise on unresampled groups prints as 0.000.
            "difference_pct_points": round(test - restest, 6) + 0.0,
        })
    return out


def model_label(results_dir):
    """Paper name of the evaluated checkpoint, read from the evaluator's summary."""
    summary_path = Path(results_dir) / SUMMARY_NAME
    if not summary_path.exists():
        return Path(results_dir).name
    checkpoint = json.loads(summary_path.read_text(encoding="utf-8")).get("checkpoint", Path(results_dir).name)
    return load_checkpoint_entries().get(checkpoint, {}).get("paper_name", checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_dirs",
        nargs="+",
        type=Path,
        help=f"Directories written by evaluate_resampled_holdout.py --out-dir (each holds {ROW_ERRORS_NAME}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results") / "table4_by_length" / "per_group_mean_error.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table = []
    for results_dir in args.results_dirs:
        rows = read_row_errors(results_dir / ROW_ERRORS_NAME)
        table.extend(group_by_length(rows, model_label(results_dir)))

    print(f"{'model':<12} {'length':>6} {'kHz':>5} {'stride':>6} {'n':>4} {'Test %':>8} {'ResTest %':>9} {'diff':>7}")
    for row in table:
        print(
            f"{row['model']:<12} {row['raw_length']:>6} {row['nominal_frequency_khz']:>5g} "
            f"{row['resample_stride']:>6} {row['n_rows']:>4} {row['test_mean_pct']:>8.3f} "
            f"{row['resampled_test_mean_pct']:>9.3f} {row['difference_pct_points']:>7.3f}"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in table:
            out = dict(row)
            for key in ("test_mean_pct", "resampled_test_mean_pct", "difference_pct_points"):
                out[key] = f"{row[key]:.3f}"
            out["nominal_frequency_khz"] = f"{row['nominal_frequency_khz']:g}"
            writer.writerow(out)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
