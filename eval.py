import numpy as np
import torch
from tqdm import tqdm

from rollout import rollout_sequence, sequence_relative_rms


def iter_evaluation_rollouts(model, dataset, indices, cfg, desc, include_predictions=False):
    model.eval()
    indices_list = list(indices)
    loop = tqdm(range(0, len(indices_list), cfg.eval_batch_size), desc=desc, leave=False)
    with torch.no_grad():
        for start_idx in loop:
            batch_indices = indices_list[start_idx:start_idx + cfg.eval_batch_size]
            batch_lengths = np.array([int(dataset.lengths[i]) for i in batch_indices], dtype=np.int64)
            batch_size = len(batch_indices)
            if batch_lengths.size == 0:
                continue
            batch_max_len = int(batch_lengths.max())
            if batch_max_len <= 1:
                yield {
                    "batch_indices": batch_indices,
                    "batch_lengths": batch_lengths,
                    "row_err": np.full(batch_size, np.nan, dtype=np.float64),
                    "pred": None,
                    "gt": None,
                }
                continue

            batch_dts = np.array([float(dataset.deltas[i]) for i in batch_indices], dtype=np.float32)
            dt = torch.from_numpy(batch_dts).to(cfg.device).view(-1, 1)
            input_seq = dataset.X[batch_indices, :batch_max_len, :].to(cfg.device)
            h_gt = dataset.Y_traj[batch_indices, :batch_max_len, :].to(cfg.device)
            lengths_tensor = torch.from_numpy(batch_lengths).to(cfg.device)

            pred, target_mask = rollout_sequence(
                model,
                input_seq,
                h_gt,
                dt,
                teacher_forcing_boundary=cfg.eval_warmup_len,
                device=cfg.device,
                lengths=lengths_tensor,
            )
            time_idx = torch.arange(batch_max_len, device=cfg.device).unsqueeze(0)
            eval_mask = target_mask & (time_idx > cfg.eval_warmup_len)
            _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)

            batch = {
                "batch_indices": batch_indices,
                "batch_lengths": batch_lengths,
                "row_err": row_err.detach().cpu().numpy().astype(np.float64),
                "pred": None,
                "gt": None,
            }
            if include_predictions:
                batch["pred"] = pred.detach().cpu().numpy().squeeze(-1)
                batch["gt"] = h_gt.detach().cpu().numpy().squeeze(-1)
            yield batch


def iter_validation_rollouts(model, dataset, val_indices, cfg, desc, include_predictions=False):
    return iter_evaluation_rollouts(model, dataset, val_indices, cfg, desc, include_predictions)


def summarize_row_errors(row_err):
    row_err_arr = np.array(row_err, dtype=np.float64)
    valid_row = row_err_arr[~np.isnan(row_err_arr)]
    mean_row_err = float(np.mean(valid_row)) if valid_row.size > 0 else float("nan")
    p95_row_err = float(np.percentile(valid_row, 95)) if valid_row.size > 0 else float("nan")
    stats = {
        "batch_loss_std": float(np.std(valid_row)) if valid_row.size > 0 else 0.0,
        "p95_row_err": p95_row_err,
    }
    return mean_row_err, stats, valid_row


def evaluate_full_length_loss(model, dataset, indices, cfg, desc="  [Val Full]"):
    if not indices:
        return float("nan"), {}
    row_err = []
    for batch in iter_evaluation_rollouts(model, dataset, indices, cfg, desc):
        row_err.extend(batch["row_err"].tolist())
    mean_row_err, stats, _valid_row = summarize_row_errors(row_err)
    return mean_row_err, stats


def evaluate_on_dataset(model, dataset, indices, cfg, *, label, include_predictions=False):
    print("")
    print(f">>> Running {label} inference...")
    if not indices:
        print(f"No {label} indices provided.")
        return {
            "mean_row_err": float("nan"),
            "p95_row_err": float("nan"),
            "row_errors": np.array([], dtype=np.float64),
            "batches": [],
        }

    row_err = []
    batches = []
    desc_label = label.title()
    for batch in iter_evaluation_rollouts(
        model, dataset, indices, cfg, f"  [{desc_label} Infer]", include_predictions=include_predictions
    ):
        row_err.extend(batch["row_err"].tolist())
        if include_predictions:
            batches.append(batch)

    mean_row_err, _stats, valid_row = summarize_row_errors(row_err)
    if valid_row.size == 0:
        print("No valid evaluation points found.")
        return {
            "mean_row_err": float("nan"),
            "p95_row_err": float("nan"),
            "row_errors": valid_row,
            "batches": batches,
        }

    p95_row_err = _stats["p95_row_err"]
    print(f"Err_seq (mean per-row, excluding warmup): {mean_row_err:.4f}")
    print(f"Err_seq (P95 per-row, excluding warmup): {p95_row_err:.4f}")
    return {
        "mean_row_err": mean_row_err,
        "p95_row_err": p95_row_err,
        "row_errors": valid_row,
        "batches": batches,
    }


def evaluate_on_validation(model, dataset, val_indices, cfg, include_predictions=False):
    return evaluate_on_dataset(
        model, dataset, val_indices, cfg, label="validation", include_predictions=include_predictions
    )


def evaluate_on_test(model, dataset, test_indices, cfg, include_predictions=False):
    return evaluate_on_dataset(
        model, dataset, test_indices, cfg, label="test", include_predictions=include_predictions
    )
