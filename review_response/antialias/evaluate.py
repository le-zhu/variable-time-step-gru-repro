"""Model error under direct versus antialiased uniform downsampling (50 kHz group).

Produces the antialiasing table of the response letter. The direct input is the
one of the manuscript's downsampling figure (``make_downsampled_batch``: keep
every ``r``-th point, recompute dB/dt). The antialiased input first applies a
zero-phase ideal FFT low-pass with cutoff ``1 / (2 r dt0)`` to the B channel of
each sequence, then decimates in the same way, so dB/dt is recomputed from the
filtered and decimated B; temperature and the target H are left unfiltered.
Both inputs share the evaluation protocol (100-point warmup on the original
grid, per-sequence relative RMS). The released checkpoints were all trained on
unfiltered decimated data, so the antialiased input is a mild distribution
shift for them.

    python -m review_response.antialias.evaluate `
      --processed-data-dir runs\\SoftplusRateExpCapGRU\\data\\paper_n87_rategru_data
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_n87_longest_downsample import evaluate_batch, make_downsampled_batch  # noqa: E402
from scripts.paper_eval import (  # noqa: E402
    check_normalization, load_checkpoint_entries, load_holdout_arrays, load_model, resolve_device,
)


MODELS = {
    "rategru": "n87:rategru",
    "dtscaledgru": "n87:dtscaledgru",
    "dtinputgru": "n87:dtinputgru",
}
WARMUP = 100


def lowpass(x, factor):
    """Ideal zero-phase low-pass at 1/(2 r) of the original sampling frequency."""
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(x.size, 1.0) > 0.5 / factor] = 0.0
    return np.fft.irfft(spectrum, n=x.size)


def antialiased_copy(data, lengths, factor):
    """Copy of ``data`` with only the B channel filtered; factor 1 is returned as is."""
    if factor == 1:
        return data
    out = data.copy()
    for row in range(len(lengths)):
        n = int(lengths[row])
        out[row, :n, 0] = lowpass(data[row, :n, 0].astype(np.float64), factor).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="Processed N87 split of experiment n87_rategru (std_db_dt).")
    parser.add_argument("--length", type=int, default=32016,
                        help="Original sequence length of the group (32016 = 50 kHz).")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 9, 16, 32])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "antialias")
    args = parser.parse_args()

    entries = load_checkpoint_entries()
    device = resolve_device(args.device)
    data, lengths, deltas = load_holdout_arrays(args.processed_data_dir)
    rows = np.flatnonzero(lengths == args.length)
    if rows.size == 0:
        raise ValueError(f"No origin_test sequence of length {args.length}")
    data, lengths, deltas = data[rows], lengths[rows], deltas[rows]
    local = list(range(len(rows)))

    inputs = {}
    for factor in args.factors:
        filtered = antialiased_copy(data, lengths, factor)
        inputs[factor] = {
            "direct": make_downsampled_batch(data, lengths, deltas, local, factor),
            "antialias": make_downsampled_batch(filtered, lengths, deltas, local, factor),
        }

    results = []
    for name in args.models:
        entry = entries[MODELS[name]]
        check_normalization(args.processed_data_dir, entry)
        model = load_model(entry, device)
        for factor in args.factors:
            result = {"model": name, "model_key": entry["model_key"], "length": args.length,
                      "factor": factor, "rows": len(local)}
            for mode, (batch, batch_lengths, batch_deltas) in inputs[factor].items():
                row_err, _pred, _gt = evaluate_batch(
                    model, batch, batch_lengths, batch_deltas, WARMUP, factor, device
                )
                valid = row_err[np.isfinite(row_err)]
                result[f"{mode}_mean"] = float(np.mean(valid))
                result[f"{mode}_p95"] = float(np.percentile(valid, 95))
            results.append(result)
            print(f"{name:12s} r={factor:<3d} direct={result['direct_mean']:.4%} "
                  f"antialias={result['antialias_mean']:.4%}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "antialias_downsampling.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "split": "origin_test",
        "length": args.length,
        "rows": len(local),
        "warmup_original_steps": WARMUP,
        "checkpoints": {name: {"key": MODELS[name], "path": entries[MODELS[name]]["path"],
                               "sha256": entries[MODELS[name]]["sha256"]} for name in args.models},
        "antialias": ("zero-phase ideal FFT low-pass on B, cutoff 1/(2 r dt0), then [::r]; "
                      "dB/dt recomputed from the decimated B; T and H unfiltered"),
        "device": str(device),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output_dir / 'antialias_downsampling.csv'}")


if __name__ == "__main__":
    main()
