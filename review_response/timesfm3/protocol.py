"""Evaluation protocol shared by the TimesFM-3 and RateGRU context sweeps.

Both scripts must truncate, split, and score every sequence in exactly the same
way, so the rules live here and nowhere else:

* A sequence longer than ``CONTEXT_LIMIT`` points keeps only its last
  ``CONTEXT_LIMIT`` points, the truncation TimesFM-3 applies internally and the
  official benchmark scripts rely on.
* A history fraction ``f`` splits the (truncated) sequence at
  ``cut = round(f * length)``: the first ``cut`` points supply the measured H,
  and the remaining ``length - cut`` points are predicted and scored.
* The error is the per-sequence relative RMS error over the predicted points,
  then averaged over sequences: the same formula as the manuscript tables, but
  over a different interval, so the numbers are not comparable with them.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


CONTEXT_LIMIT = 15360
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# TimesFM-3 needs at least one 32-point patch of context.
MIN_CONTEXT = 32
SPLIT = "origin_test"


def load_split(data_dir, split=SPLIT):
    """Arrays written by ``process.py``: channels B, dB/dt, T, H."""
    data_dir = Path(data_dir)
    config_path = data_dir / "process_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Processed-data configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("dt_norm_mode") != "std_db_dt":
        raise ValueError(
            f"{data_dir} uses dt_norm_mode={config.get('dt_norm_mode')!r}; the "
            "released RateGRU expects 'std_db_dt' (experiment n87_rategru)."
        )
    data = np.load(data_dir / f"{split}.npy", mmap_mode="r")
    lengths = np.load(data_dir / f"{split}_lengths.npy").astype(np.int64)
    deltas = np.load(data_dir / f"{split}_delta_t.npy").astype(np.float32)
    return data, lengths, deltas


def group_by_truncated_length(lengths, limit=CONTEXT_LIMIT):
    """Sequence indices keyed by their length after truncation."""
    groups = defaultdict(list)
    for index, length in enumerate(lengths):
        groups[min(int(length), limit)].append(index)
    return dict(sorted(groups.items()))


def truncation_window(length, truncated_length):
    """Slice that keeps the last ``truncated_length`` points of a sequence."""
    end = int(length)
    return slice(end - truncated_length, end)


def split_point(fraction, truncated_length):
    """Number of measured history points and of predicted points."""
    cut = int(round(fraction * truncated_length))
    return cut, truncated_length - cut


def relative_rms(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denom = float(np.sqrt(np.mean(target ** 2)))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)) / denom)


def summarize(fraction, errors):
    valid = errors[np.isfinite(errors)]
    return {
        "fraction": fraction,
        "n_scored": int(valid.size),
        "mean_rel_rms": float(valid.mean()),
        "median_rel_rms": float(np.median(valid)),
        "p95_rel_rms": float(np.percentile(valid, 95)),
    }


def parse_fractions(text):
    return [float(value) for value in text.split(",") if value.strip()]
