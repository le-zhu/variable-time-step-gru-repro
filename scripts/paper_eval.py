"""Held-out metrics reported in the paper's cross-material table.

Three quantities are defined here, all evaluated on the original 16 MHz sampling
grid of the held-out split:

* the sequence relative RMS error on that grid ("Test" in the paper);
* the same error after the training resampling rule is applied to the held-out
  sequences ("ResTest");
* the core-loss error, the relative error of the hybrid B--H loop area
  ("epsilon_W").

The rollout, the resampling rule, and the loop-area quadrature are the ones used
to produce the published numbers and must not be altered casually: they define
the metrics rather than merely computing them.
"""

import json
from pathlib import Path

import numpy as np
import torch

from models import get_model_entry
from process import compute_resample_stride
from rollout import rollout_sequence_original_grid, sequence_relative_rms


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint_entries():
    """Map ``n87:<folder>`` and ``material:<material>`` to manifest entries.

    The N87 key is the checkpoint folder rather than the paper name, because the
    joint ablation reports four of the models under both time-step
    normalizations: keying on the paper name alone would let one entry shadow
    another and silently evaluate the wrong weights.
    """
    manifest = json.loads(
        (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
    )
    entries = {}
    for entry in manifest["n87_paper_models"]:
        entries[f"n87:{Path(entry['path']).parent.name}"] = entry
    for entry in manifest["ten_material_rategru"]:
        entries[f"material:{entry['material']}"] = entry
    expected = len(manifest["n87_paper_models"]) + len(manifest["ten_material_rategru"])
    if len(entries) != expected:
        raise ValueError(f"Checkpoint keys collide: {len(entries)} keys for {expected} entries.")
    return entries


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(entry, device):
    model = get_model_entry(entry["model_key"]).model_class().to(device)
    path = REPOSITORY_ROOT / "checkpoints" / entry["path"]
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def check_normalization(data_dir, entry):
    """Fail early when the processed split does not match the checkpoint."""
    config_path = Path(data_dir) / "process_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Processed-data configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = entry.get("dt_normalization", "std_db_dt")
    if config.get("dt_norm_mode") != expected:
        raise ValueError(
            f"Processed data uses dt_norm_mode={config.get('dt_norm_mode')!r}; "
            f"checkpoint expects {expected!r}."
        )


def load_holdout_arrays(data_dir):
    data_dir = Path(data_dir)
    data = np.load(data_dir / "origin_test.npy")
    lengths = np.load(data_dir / "origin_test_lengths.npy").astype(np.int64)
    deltas = np.load(data_dir / "origin_test_delta_t.npy").astype(np.float32)
    return data, lengths, deltas


def pad_items(items):
    max_item_len = max(len(item) for item in items)
    feature_count = items[0].shape[1]
    padded = np.zeros((len(items), max_item_len, feature_count), dtype=np.float32)
    for idx, item in enumerate(items):
        padded[idx, : len(item), :] = item
    return padded


def build_resampled_holdout(data, lengths, deltas, *, max_len):
    """Apply the training resampling rule to the held-out sequences.

    ``stride = ceil(length / max_len)``; the series is sliced with that stride,
    dt is multiplied by it, and dB/dt is recomputed after slicing.
    """
    items, resampled_lengths, resampled_deltas, strides = [], [], [], []
    for idx, length in enumerate(lengths):
        length = int(length)
        stride = compute_resample_stride(length, max_len)
        resampled = data[idx, :length, :][::stride, :].copy()
        dt_scaled = float(deltas[idx] * stride)
        b_norm = resampled[:, 0]
        db_dt = np.zeros_like(b_norm, dtype=np.float32)
        if len(b_norm) > 1:
            db_dt[:-1] = (b_norm[1:] - b_norm[:-1]) / dt_scaled
        resampled[:, 1] = db_dt
        items.append(resampled.astype(np.float32, copy=False))
        resampled_lengths.append(len(resampled))
        resampled_deltas.append(dt_scaled)
        strides.append(stride)
    return (
        pad_items(items),
        np.array(resampled_lengths, dtype=np.int64),
        np.array(resampled_deltas, dtype=np.float32),
        np.array(strides, dtype=np.int64),
    )


def evaluate_sequence_error(model, data, lengths, deltas, *, original_strides, warmup,
                            batch_size, device):
    """Per-sequence relative RMS error over the autoregressive interval."""
    original_strides = np.asarray(original_strides, dtype=np.int64)
    if original_strides.shape != lengths.shape:
        raise ValueError(
            f"original_strides shape {original_strides.shape} does not match lengths {lengths.shape}"
        )
    if np.any(original_strides < 1):
        raise ValueError("original_strides must be positive integers")

    count = data.shape[0]
    row_errors = np.full(count, np.nan, dtype=np.float64)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch_lengths_np = lengths[start:end].astype(np.int64)
        batch_max_len = int(batch_lengths_np.max())
        input_seq = torch.from_numpy(
            np.array(data[start:end, :batch_max_len, :3], dtype=np.float32, copy=True)
        ).to(device)
        h_gt = torch.from_numpy(
            np.array(data[start:end, :batch_max_len, 3:4], dtype=np.float32, copy=True)
        ).to(device)
        dt = torch.from_numpy(deltas[start:end]).float().view(-1, 1).to(device)
        batch_lengths = torch.from_numpy(batch_lengths_np).to(device)
        original_indices = torch.from_numpy(
            np.arange(batch_max_len, dtype=np.int64)[None, :] * original_strides[start:end, None]
        ).to(device)
        with torch.inference_mode():
            pred, _target_mask, eval_mask = rollout_sequence_original_grid(
                model, input_seq, h_gt, dt, original_indices,
                warmup_original_steps=warmup, device=device, lengths=batch_lengths,
            )
            _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)
        row_errors[start:end] = row_err.detach().cpu().numpy().astype(np.float64)
        del input_seq, h_gt, dt, batch_lengths, original_indices, pred, eval_mask, row_err
        if device.type == "cuda":
            torch.cuda.empty_cache()

    valid = row_errors[np.isfinite(row_errors)]
    if valid.size == 0:
        raise ValueError("No valid row errors")
    return float(np.mean(valid)), float(np.percentile(valid, 95)), row_errors


def loop_area(h_seq, b_seq):
    """Signed trapezoidal line integral of H dB over the full sequence."""
    db = b_seq[:, 1:] - b_seq[:, :-1]
    return (0.5 * (h_seq[:, :-1] + h_seq[:, 1:]) * db).sum(dim=1)


def total_absolute_work(h_seq, b_seq):
    """Trapezoidal integral of |H dB|, the gross scale of the H dB exchange.

    ``loop_area`` is the net dissipation left after the stored-and-returned part
    cancels; this is the traffic before that cancellation. Their ratio (lambda)
    says how thin the loop is, which sets how much a given absolute area error is
    amplified when expressed relative to the loss.
    """
    db = b_seq[:, 1:] - b_seq[:, :-1]
    return (0.5 * (h_seq[:, :-1] + h_seq[:, 1:]) * db).abs().sum(dim=1)


def evaluate_loop_areas(model, data, lengths, deltas, *, warmup, batch_size, device):
    """Measured, hybrid, and gross loop areas for every held-out sequence."""
    count = len(lengths)
    w_true = np.full(count, np.nan, dtype=np.float64)
    w_hyb = np.full(count, np.nan, dtype=np.float64)
    w_gross = np.full(count, np.nan, dtype=np.float64)

    for length in sorted(np.unique(lengths)):
        ids = np.flatnonzero(lengths == length)
        seq_len = int(length)
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            block = np.array(data[chunk, :seq_len, :], dtype=np.float32, copy=True)
            inputs = torch.from_numpy(block[:, :, :3]).to(device)
            h_gt = torch.from_numpy(block[:, :, 3:4]).to(device)
            b_seq = torch.from_numpy(block[:, :, 0]).to(device)
            dt = torch.from_numpy(deltas[chunk]).float().view(-1, 1).to(device)
            original_indices = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(len(chunk), seq_len)
            )
            with torch.inference_mode():
                pred, _target_mask, eval_mask = rollout_sequence_original_grid(
                    model, inputs, h_gt, dt, original_indices,
                    warmup_original_steps=warmup, device=device,
                )
                # True H during warmup, prediction afterwards: the same split the
                # sequence-error metric uses.
                hybrid = torch.where(eval_mask.unsqueeze(-1), pred, h_gt)
                area_true = loop_area(h_gt.squeeze(-1), b_seq)
                area_hyb = loop_area(hybrid.squeeze(-1), b_seq)
                area_gross = total_absolute_work(h_gt.squeeze(-1), b_seq)
            w_true[chunk] = area_true.double().cpu().numpy()
            w_hyb[chunk] = area_hyb.double().cpu().numpy()
            w_gross[chunk] = area_gross.double().cpu().numpy()
            del inputs, h_gt, b_seq, dt, pred, eval_mask, hybrid
            del area_true, area_hyb, area_gross
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return w_true, w_hyb, w_gross


def core_loss_stats(w_true, w_hyb, w_gross):
    abs_err = np.abs(w_hyb - w_true)
    rel_err = 100.0 * abs_err / np.abs(w_true)
    stats = {
        "sequences": int(len(w_true)),
        "negative_predicted_area": int((w_hyb < 0).sum()),
        "negative_measured_area": int((w_true < 0).sum()),
        "median_percent": float(np.median(rel_err)),
        "mean_percent": float(np.mean(rel_err)),
        "p95_percent": float(np.percentile(rel_err, 95)),
        # Loss-weighted rather than per-sequence: the relative error of the summed
        # energy over the whole held-out set.
        "wape_percent": float(100.0 * abs_err.sum() / np.abs(w_true).sum()),
        "signed_bias_percent": float(100.0 * (w_hyb - w_true).sum() / np.abs(w_true).sum()),
        # lambda: net dissipation as a fraction of the gross H dB exchange.
        "lambda_median": float(np.median(np.abs(w_true) / w_gross)),
        "gross_normalized_error_median_percent": float(np.median(100.0 * abs_err / w_gross)),
    }
    return stats, rel_err
