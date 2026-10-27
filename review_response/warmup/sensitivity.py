"""Sensitivity of the N87 RateGRU error to the warmup length.

Produces the warmup table of the response letter with the released checkpoint
``n87:rategru``; nothing is retrained. During the warmup the model receives the
measured H as its feedback input and afterwards runs on its own prediction.
Every setting is scored on the same points through two fixed windows:

* window A: outputs after original point 100 (the manuscript's protocol), for
  warmups up to 100 points;
* window B: outputs after original point 1000, for every warmup.

"none" starts from H = 0 with no measured H at all; "H0 only" feeds only the
measured initial value, which is what a zero-length warmup reduces to.

    python -m review_response.warmup.sensitivity `
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

from scripts.paper_eval import (  # noqa: E402
    check_normalization, load_checkpoint_entries, load_model, resolve_device,
)


CHECKPOINT_KEY = "n87:rategru"
# (label, warmup length in points, start from H = 0 instead of the measured H0)
SETTINGS = [("none", 0, True), ("H0 only", 0, False), ("10", 10, False), ("50", 50, False),
            ("100", 100, False), ("500", 500, False), ("1000", 1000, False)]
WINDOWS = {"A_after_100": 100, "B_after_1000": 1000}


def rollout(model, x, y, lengths, dt, warmup, zero_init):
    """Measured H while ``t < warmup``, own prediction afterwards; padded steps frozen."""
    count, seq_len, _ = x.shape
    h = torch.zeros_like(y[:, 0]) if zero_init else y[:, 0]
    hs = torch.zeros(count, model.hidden_size, device=x.device)
    preds = [h]
    for t in range(seq_len - 1):
        valid = (t < (lengths - 1)).unsqueeze(1)
        h_in = torch.where(valid & (t < warmup), y[:, t], h)
        h_next, hs_next = model.forward_step(x[:, t], h_in, hs, dt=dt)
        hs = torch.where(valid, hs_next, hs)
        h = torch.where(valid, h_next, h)
        preds.append(h)
    return torch.stack(preds, dim=1)[..., 0]


def window_errors(pred, gt, in_len, idx, start):
    mask = (in_len & (idx > start)).float()
    return torch.sqrt((mask * (pred - gt) ** 2).sum(1)
                      / (mask * gt ** 2).sum(1).clamp_min(1e-12)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="Processed N87 split of experiment n87_rategru (std_db_dt).")
    parser.add_argument("--split", default="origin_test", choices=["origin_test", "origin_val"])
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Sequences per rollout; 0 rolls the whole split out at once, the "
                             "setting the reported numbers used.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: results/warmup/rategru_rms_<split>.json.")
    args = parser.parse_args()

    entry = load_checkpoint_entries()[CHECKPOINT_KEY]
    check_normalization(args.processed_data_dir, entry)
    device = resolve_device(args.device)
    model = load_model(entry, device)
    data_dir = args.processed_data_dir
    data = np.load(data_dir / f"{args.split}.npy")
    lengths_all = np.load(data_dir / f"{args.split}_lengths.npy")
    deltas_all = np.load(data_dir / f"{args.split}_delta_t.npy").astype(np.float32)
    batch_size = args.batch_size or len(lengths_all)

    errors = {(name, window): [] for name, _w, _z in SETTINGS for window in WINDOWS}
    with torch.inference_mode():
        for start in range(0, len(lengths_all), batch_size):
            stop = min(start + batch_size, len(lengths_all))
            chunk = torch.from_numpy(data[start:stop]).float().to(device)
            lengths = torch.from_numpy(lengths_all[start:stop]).long().to(device)
            dt = torch.from_numpy(deltas_all[start:stop]).to(device).view(-1, 1)
            x, y = chunk[:, :, :-1], chunk[:, :, -1:]
            gt = y[..., 0]
            idx = torch.arange(gt.shape[1], device=device).unsqueeze(0)
            in_len = idx < lengths.unsqueeze(1)
            for name, warmup, zero_init in SETTINGS:
                pred = rollout(model, x, y, lengths, dt, warmup, zero_init)
                for window, window_start in WINDOWS.items():
                    if warmup <= window_start:
                        errors[(name, window)].append(window_errors(pred, gt, in_len, idx, window_start))
                del pred
            del chunk, x, y, gt, in_len
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = []
    for name, warmup, _zero_init in SETTINGS:
        row = {"warmup": name}
        for window, window_start in WINDOWS.items():
            if warmup > window_start:
                row[window] = None
                continue
            err = np.concatenate(errors[(name, window)])
            row[window] = {"mean": float(err.mean()), "median": float(np.median(err)),
                           "p95": float(np.percentile(err, 95))}
        rows.append(row)
        print(json.dumps(row), flush=True)

    output = args.output or Path("results") / "warmup" / f"rategru_rms_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "checkpoint": entry["path"],
        "checkpoint_sha256": entry["sha256"],
        "split": args.split,
        "sequences": int(len(lengths_all)),
        "windows": {name: f"steps > {start}" for name, start in WINDOWS.items()},
        "device": str(device),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}", flush=True)
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
