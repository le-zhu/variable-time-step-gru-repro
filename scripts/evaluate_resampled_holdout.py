"""Held-out sequence error of a released checkpoint, on the original and the
training-rule-resampled sampling grids.

These are the "Test" and "ResTest" columns of the paper's cross-material table.
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
    build_resampled_holdout,
    check_normalization,
    evaluate_sequence_error,
    load_checkpoint_entries,
    load_holdout_arrays,
    load_model,
    resolve_device,
)


def parse_args(entries):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=entries, required=True)
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        required=True,
        help="Folder produced by process.py for the same material, split, and dt normalization.",
    )
    parser.add_argument("--max-len", type=int, default=4000,
                        help="Resampling target length; the training value is 4000.")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup length in original-grid steps.")
    parser.add_argument("--batch-size", type=int, default=1024)
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

    raw_mean, raw_p95, raw_errors = evaluate_sequence_error(
        model, data, lengths, deltas,
        original_strides=np.ones(len(lengths), dtype=np.int64),
        warmup=args.warmup, batch_size=args.batch_size, device=device,
    )
    res_data, res_lengths, res_deltas, strides = build_resampled_holdout(
        data, lengths, deltas, max_len=args.max_len
    )
    res_mean, res_p95, res_errors = evaluate_sequence_error(
        model, res_data, res_lengths, res_deltas,
        original_strides=strides, warmup=args.warmup, batch_size=args.batch_size, device=device,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Held-out mean relative error:            {raw_mean * 100:.3f}%")
    print(f"Held-out P95 relative error:             {raw_p95 * 100:.3f}%")
    print(f"Resampled held-out mean relative error:  {res_mean * 100:.3f}%")
    print(f"Resampled held-out P95 relative error:   {res_p95 * 100:.3f}%")
    print(f"Resampling stride range:                 {int(strides.min())}--{int(strides.max())}")

    for key, value in (("test_mean", raw_mean * 100), ("test_p95", raw_p95 * 100),
                       ("resampled_heldout_mean", res_mean * 100),
                       ("resampled_heldout_p95", res_p95 * 100)):
        if key in entry and abs(entry[key] - value) > 5e-3:
            print(f"  note: manifest records {key}={entry[key]:.3f}%, recomputed {value:.3f}%")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "resampled_holdout_row_errors.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "raw_length", "resampled_length", "stride",
                             "holdout_error", "resampled_holdout_error"])
            for idx in range(len(lengths)):
                writer.writerow([idx, int(lengths[idx]), int(res_lengths[idx]), int(strides[idx]),
                                 f"{raw_errors[idx]:.8f}", f"{res_errors[idx]:.8f}"])
        summary = {
            "checkpoint": args.checkpoint, "material": entry.get("material", entry.get("paper_name")),
            "max_len": args.max_len, "warmup_original_steps": args.warmup,
            "heldout_mean_percent": raw_mean * 100, "heldout_p95_percent": raw_p95 * 100,
            "resampled_heldout_mean_percent": res_mean * 100,
            "resampled_heldout_p95_percent": res_p95 * 100,
            "resampling_rule": "stride=ceil(length/max_len); series[::stride]; "
                               "dt scaled by stride; dB/dt recomputed after slicing",
        }
        (args.out_dir / "resampled_holdout_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
