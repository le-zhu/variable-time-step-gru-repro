"""Sequence loading and rollout helpers shared by the RateGRU-NF waveform figure.

Split out of the working error-accumulation script so that only the manuscript
figure remains an entry point. ``evaluate_rows`` takes an optional
``eval_warmup_steps`` so that a model rolled out without teacher forcing can be
scored over the same steps as one that uses a warmup.
"""

import json
from pathlib import Path

import numpy as np
import torch

from models import get_model_entry
from rollout import rollout_sequence, sequence_relative_rms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "no_h"
DEFAULT_H_FEEDBACK_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "rategru"
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path):
    with require_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(run_dir, device):
    run_config = load_json(run_dir / "run_config.json")
    model_key = run_config["model_key"]
    model = get_model_entry(model_key).model_class().to(device)
    model_path = require_file(run_dir / "model.pth")
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, run_config, model_path


def load_longest_origin_split(data_dir, split="test"):
    split = str(split).lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split!r}; expected 'val' or 'test'.")

    prefix = f"origin_{split}"
    data = np.load(require_file(data_dir / f"{prefix}.npy"))
    lengths = np.load(require_file(data_dir / f"{prefix}_lengths.npy")).astype(np.int64)
    deltas = np.load(require_file(data_dir / f"{prefix}_delta_t.npy")).astype(np.float32)
    norm_stats = load_json(data_dir / "norm_stats.json")

    max_len = int(lengths.max())
    row_indices = np.flatnonzero(lengths == max_len).astype(int)
    if len(row_indices) == 0:
        raise RuntimeError(f"No {prefix} rows found.")

    selected = data[row_indices, :max_len, :].astype(np.float32, copy=False)
    return selected, lengths[row_indices], deltas[row_indices], norm_stats, row_indices


def evaluate_rows(model, batch_data, batch_lengths, batch_deltas, warmup_steps, device, eval_warmup_steps=None):
    """Roll out with ``warmup_steps`` of teacher forcing and score steps after
    ``eval_warmup_steps`` (defaults to ``warmup_steps``)."""
    if eval_warmup_steps is None:
        eval_warmup_steps = warmup_steps
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    dt = torch.from_numpy(batch_deltas).float().to(device).view(-1, 1)
    lengths_tensor = torch.from_numpy(batch_lengths).to(device)

    with torch.no_grad():
        pred, target_mask = rollout_sequence(
            model,
            input_seq,
            h_gt,
            dt,
            teacher_forcing_boundary=warmup_steps,
            device=device,
            lengths=lengths_tensor,
        )
        time_idx = torch.arange(batch_data.shape[1], device=device).unsqueeze(0)
        eval_mask = target_mask & (time_idx > eval_warmup_steps)
        _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)

    return (
        row_err.detach().cpu().numpy().astype(np.float64),
        pred.detach().cpu().numpy().squeeze(-1),
        h_gt.detach().cpu().numpy().squeeze(-1),
    )
