"""Zero-shot TimesFM-3 on the N87 held-out split, sweeping the history fraction.

The target is H. B and dB/dt are passed as past-and-future covariates, the
excitation the GRU models also receive; dB/dt is z-normalized per sequence.
Temperature is not passed: it is constant within every sequence, so as a
per-time-step covariate it carries no information.

The weights are used as released (no training, no fine-tuning). Sequences longer
than the 15,360-point context limit keep their last 15,360 points, and the whole
forecast horizon is produced in one call, without rolling windows. See
``protocol.py`` for the split and the error.

Run inside the TimesFM environment described in README.md, for example:

    .\\review_response\\timesfm3\\.venv\\Scripts\\python.exe `
      -m review_response.timesfm3.timesfm3_context_sweep `
      --processed-data-dir runs\\SoftplusRateExpCapGRU\\data\\paper_n87_rategru_data
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from review_response.timesfm3.download_model import REPO_ID, REVISION  # noqa: E402
from review_response.timesfm3.protocol import (  # noqa: E402
    CONTEXT_LIMIT, FRACTIONS, MIN_CONTEXT, SPLIT, group_by_truncated_length,
    load_split, parse_fractions, relative_rms, split_point, summarize, truncation_window,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model"


def covariates(window, channels):
    """Past-and-future covariates for one truncated sequence."""
    out = []
    if "B" in channels:
        out.append(window[:, 0].astype(np.float32))
    if "dBdt" in channels:
        db_dt = window[:, 1].astype(np.float64)
        out.append(((db_dt - db_dt.mean()) / (db_dt.std() + 1e-9)).astype(np.float32))
    return np.stack(out) if out else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="Processed N87 split of experiment n87_rategru.")
    parser.add_argument("--split", default=SPLIT, choices=["origin_test", "origin_val"])
    parser.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    parser.add_argument("--covariates", default="B,dBdt", choices=["B,dBdt", "B", "none"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Folder written by download_model.py.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length-groups", type=int, default=0,
                        help="Smoke test: evaluate only the N longest length groups.")
    parser.add_argument("--output", type=Path,
                        default=Path("results") / "timesfm3" / f"timesfm3_{SPLIT}_trunc{CONTEXT_LIMIT}.json")
    args = parser.parse_args()

    import torch
    from timesfm3 import ModelConfig, TimesFM3Forecaster

    data, lengths, _deltas = load_split(args.processed_data_dir, args.split)
    groups = group_by_truncated_length(lengths)
    order = list(groups)
    if args.max_length_groups:
        order = order[-args.max_length_groups:]

    forecaster = TimesFM3Forecaster(ModelConfig(
        checkpoint_path=str(args.checkpoint), device=args.device,
        per_core_batch_size=args.batch_size, local_files_only=True,
    ))
    parameter_count = sum(p.numel() for p in forecaster.model.parameters())
    print(f"parameters={parameter_count} global_context={forecaster.global_context} "
          f"sequences={len(lengths)} length_groups={ {k: len(groups[k]) for k in order} }", flush=True)

    def predict(windows, cut, horizon):
        contexts = [w[:cut, 3].astype(np.float32) for w in windows]
        future = None if args.covariates == "none" else [covariates(w, args.covariates) for w in windows]
        with torch.inference_mode():
            return list(forecaster.predict_batch(
                contexts=contexts, horizon=horizon, past_future_covariates=future,
                return_quantiles=False, use_symmetric_averaging=False,
                make_positive=False, padding_mode="none",
            ))

    rows = []
    for fraction in parse_fractions(args.fractions):
        errors = np.full(len(lengths), np.nan, dtype=np.float64)
        per_length = []
        for truncated_length in order:
            members = groups[truncated_length]
            cut, horizon = split_point(fraction, truncated_length)
            if cut < MIN_CONTEXT or horizon < 1:
                continue
            started = time.perf_counter()
            group_errors = []
            for start in range(0, len(members), args.batch_size):
                batch = members[start:start + args.batch_size]
                windows = [np.asarray(data[i, truncation_window(lengths[i], truncated_length)])
                           for i in batch]
                try:
                    outputs = predict(windows, cut, horizon)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    print(f"  out of memory at length={truncated_length}; retrying one by one", flush=True)
                    outputs = [out for w in windows for out in predict([w], cut, horizon)]
                for i, window, output in zip(batch, windows, outputs):
                    pred = np.asarray(output.forecast, dtype=np.float64)[:horizon]
                    errors[i] = relative_rms(pred, window[cut:, 3])
                    group_errors.append(errors[i])
            per_length.append({
                "truncated_length": truncated_length, "n": len(members),
                "context": cut, "horizon": horizon,
                "mean_rel_rms": float(np.nanmean(group_errors)),
                "seconds": time.perf_counter() - started,
            })
            print(json.dumps({"fraction": fraction, **per_length[-1]}), flush=True)
        rows.append({**summarize(fraction, errors), "per_length": per_length})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "per_length"}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": REPO_ID,
        "revision": REVISION,
        "parameter_count": parameter_count,
        "split": args.split,
        "context_limit": CONTEXT_LIMIT,
        "covariates": args.covariates,
        "target": "H",
        "error": "per-sequence relative RMS over the forecast points, then averaged",
        "device": args.device,
        "batch_size": args.batch_size,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
