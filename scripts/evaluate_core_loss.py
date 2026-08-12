"""Core-loss error of a released checkpoint on the held-out split.

Each sequence in the dataset is one complete quasi-steady-state cycle, so B
returns to its initial value and the closed-loop integral of H dB is the
dissipated energy. A hybrid H sequence uses the measured values during warmup
and the model prediction afterwards, following the same split as the
sequence-error metric. With W the measured loop area and W_hyb the hybrid one,

    epsilon_W = |W_hyb - W| / W

is reported as a mean and a 95th percentile over the held-out sequences, the
"epsilon_W" columns of the paper's cross-material table.

Both areas use the same B samples and the same trapezoidal quadrature, so the
leading discretization bias cancels in the ratio. W_hyb is kept signed: a
negative predicted area is a physically impossible loop, and taking its
magnitude would silently report such a sequence as a small error, so the count
of negative predicted areas is reported instead.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.paper_eval import (  # noqa: E402
    check_normalization,
    core_loss_stats,
    evaluate_loop_areas,
    load_checkpoint_entries,
    load_holdout_arrays,
    load_model,
    resolve_device,
)


def parse_args(entries):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", choices=entries, required=True)
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        required=True,
        help="Folder produced by process.py for the same material, split, and dt normalization.",
    )
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup length in original-grid steps.")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Optional directory for the per-sequence CSV.")
    return parser.parse_args()


def main():
    entries = load_checkpoint_entries()
    args = parse_args(entries)
    entry = entries[args.checkpoint]
    data_dir = args.processed_data_dir.expanduser().resolve()
    check_normalization(data_dir, entry)

    device = resolve_device(args.device)
    model = load_model(entry, device)
    data, lengths, deltas = load_holdout_arrays(data_dir)
    w_true, w_hyb, w_gross = evaluate_loop_areas(
        model, data, lengths, deltas,
        warmup=args.warmup, batch_size=args.batch_size, device=device,
    )
    stats, rel_err = core_loss_stats(w_true, w_hyb, w_gross)

    print(f"Checkpoint: {args.checkpoint}   sequences: {stats['sequences']}")
    print(f"epsilon_W mean:                 {stats['mean_percent']:.3f}%")
    print(f"epsilon_W P95:                  {stats['p95_percent']:.3f}%")
    print(f"epsilon_W median:               {stats['median_percent']:.3f}%")
    print(f"loss-weighted (WAPE):           {stats['wape_percent']:.3f}%")
    print(f"signed bias:                    {stats['signed_bias_percent']:+.3f}%")
    print(f"lambda median:                  {stats['lambda_median']:.4f}")
    print(f"negative predicted loop areas:  {stats['negative_predicted_area']} / {stats['sequences']}")

    for key, value in (("core_loss_mean", stats["mean_percent"]),
                       ("core_loss_p95", stats["p95_percent"])):
        if key in entry and abs(entry[key] - value) > 5e-3:
            print(f"  note: manifest records {key}={entry[key]:.3f}%, recomputed {value:.3f}%")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "core_loss_by_sequence.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "sequence_length", "loop_area_measured",
                             "loop_area_hybrid", "gross_absolute_work", "lambda",
                             "abs_rel_error_percent", "signed_rel_error_percent"])
            for idx in range(len(lengths)):
                writer.writerow([
                    idx, int(lengths[idx]), f"{w_true[idx]:.10g}", f"{w_hyb[idx]:.10g}",
                    f"{w_gross[idx]:.10g}", f"{abs(w_true[idx]) / w_gross[idx]:.8f}",
                    f"{rel_err[idx]:.8f}",
                    f"{100.0 * (w_hyb[idx] - w_true[idx]) / abs(w_true[idx]):.8f}",
                ])
        (args.out_dir / "core_loss_summary.json").write_text(
            json.dumps({"checkpoint": args.checkpoint,
                        "warmup_original_steps": args.warmup,
                        "metric": "complete hybrid B-H loop area, trapezoidal, original grid",
                        **stats}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
