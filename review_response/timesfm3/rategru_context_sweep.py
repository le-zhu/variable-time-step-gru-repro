"""RateGRU under the TimesFM-3 context-sweep protocol (the RateGRU columns).

The released N87 RateGRU checkpoint (``n87:rategru``) receives the measured H
for the first ``cut`` points of every truncated sequence and then runs freely;
the remaining points are scored exactly as in ``timesfm3_context_sweep.py``.
Nothing is retrained. Runs on CPU in a few minutes; a GPU is used if present.

    python -m review_response.timesfm3.rategru_context_sweep `
      --processed-data-dir runs\\SoftplusRateExpCapGRU\\data\\paper_n87_rategru_data
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from review_response.timesfm3.protocol import (  # noqa: E402
    CONTEXT_LIMIT, FRACTIONS, MIN_CONTEXT, SPLIT, group_by_truncated_length,
    load_split, parse_fractions, split_point, summarize, truncation_window,
)
from scripts.paper_eval import load_checkpoint_entries, load_model, resolve_device  # noqa: E402


CHECKPOINT_KEY = "n87:rategru"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="Processed N87 split of experiment n87_rategru.")
    parser.add_argument("--split", default=SPLIT, choices=["origin_test", "origin_val"])
    parser.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("results") / "timesfm3" / f"rategru_{SPLIT}_trunc{CONTEXT_LIMIT}.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    entry = load_checkpoint_entries()[CHECKPOINT_KEY]
    model = load_model(entry, device)
    print(f"checkpoint={entry['path']} device={device}", flush=True)

    data, lengths, deltas = load_split(args.processed_data_dir, args.split)
    groups = group_by_truncated_length(lengths)

    rows = []
    for fraction in parse_fractions(args.fractions):
        errors = np.full(len(lengths), np.nan, dtype=np.float64)
        for truncated_length, members in groups.items():
            cut, horizon = split_point(fraction, truncated_length)
            if cut < MIN_CONTEXT or horizon < 1:
                continue
            for start in range(0, len(members), args.batch_size):
                batch = members[start:start + args.batch_size]
                windows = [data[i, truncation_window(lengths[i], truncated_length)] for i in batch]
                x = torch.from_numpy(np.stack([w[:, :3] for w in windows]).astype(np.float32)).to(device)
                y = torch.from_numpy(np.stack([w[:, 3] for w in windows]).astype(np.float32)).to(device)
                dt = torch.from_numpy(deltas[batch]).to(device).view(-1, 1)
                y = y.unsqueeze(-1)
                h = y[:, 0]
                hs = torch.zeros(len(batch), model.hidden_size, device=device)
                preds = [h]
                with torch.inference_mode():
                    for t in range(truncated_length - 1):
                        # Measured H through the history, own prediction afterwards.
                        h_in = y[:, t] if t < cut else h
                        h, hs = model.forward_step(x[:, t, :], h_in, hs, dt=dt)
                        preds.append(h)
                pred = torch.stack(preds, dim=1)[..., 0][:, cut:cut + horizon]
                target = y[..., 0][:, cut:cut + horizon]
                err = torch.sqrt(((pred - target) ** 2).mean(dim=1)
                                 / (target ** 2).mean(dim=1).clamp_min(1e-12))
                errors[batch] = err.double().cpu().numpy()
        rows.append(summarize(fraction, errors))
        print(json.dumps(rows[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": "RateGRU",
        "checkpoint": entry["path"],
        "checkpoint_sha256": entry["sha256"],
        "split": args.split,
        "context_limit": CONTEXT_LIMIT,
        "protocol": "first `cut` points receive the measured H, then free running; "
                    "per-sequence relative RMS over the remaining points",
        "device": str(device),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
