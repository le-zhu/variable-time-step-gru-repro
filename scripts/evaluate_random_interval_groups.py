"""Random nonuniform grids on every resampled frequency group of one material.

This generalizes ``evaluate_n87_longest_random_interval`` from the longest N87
group to every frequency group of any released checkpoint, and produces the
nonuniform-grid table of the revised manuscript (N87, five frequency groups)
as well as the ten-material extension in the review response.

Grid construction is the one of ``make_random_interval_batch``: the retained
points start at the first original sample, each interval is an independent
random integer in ``[1, max_step]`` original steps, and sampling stops before
the next index would pass the end of the sequence. The trailing partial
interval is discarded, so every retained interval is bounded by two measured
points. By default ``max_step`` is the group's own training downsampling factor,
``ceil(length / max_len)``, so a random step never leaves the sampling density
the model was trained on; groups whose factor is 1 (800 and 500 kHz) carry no
variable step and are skipped. Each seed draws one grid per group, all seeds of
a group are rolled out in one batch, and the same rows are also evaluated on
the original measured grid (stride 1) as the comparison column.

    python -m scripts.evaluate_random_interval_groups --checkpoint n87:rategru `
      --processed-data-dir runs\\SoftplusRateExpCapGRU\\data\\paper_n87_rategru_data
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from process import compute_resample_stride  # noqa: E402
from rollout import rollout_sequence_original_grid, sequence_relative_rms  # noqa: E402
from scripts.evaluate_n87_longest_downsample import make_downsampled_batch  # noqa: E402
from scripts.evaluate_n87_longest_random_interval import make_random_interval_batch  # noqa: E402
from scripts.paper_eval import (  # noqa: E402
    check_normalization, evaluate_sequence_error, load_checkpoint_entries, load_model,
    resolve_device,
)


DEFAULT_SEEDS = (42, 43, 44, 45)
DEFAULT_WARMUP = 100
# Every MagNetX sequence holds 100 subcycles at 16 MHz, so the number of samples
# per subcycle identifies the nominal excitation frequency (kHz).
SAMPLES_PER_SUBCYCLE_TO_KHZ = {20: 800, 32: 500, 50: 320, 80: 200, 128: 125, 200: 80, 320: 50}
SUMMARY_FIELDS = [
    "material", "frequency_khz", "original_length", "row_count",
    "training_downsampling_factor", "max_step", "warmup_original_steps",
    "retained_length_mean", "retention_ratio_mean", "step_min", "step_max", "step_mean",
    "mean_err", "mean_err_std", "p95_err", "p95_err_std",
    "uniform_mean_err", "uniform_p95_err", "random_over_uniform",
    "original_mean_err", "original_p95_err", "random_over_original",
]


def frequency_khz(length):
    samples_per_subcycle = length / 100.0
    for samples, khz in SAMPLES_PER_SUBCYCLE_TO_KHZ.items():
        if abs(samples_per_subcycle - samples) <= 2.0:
            return khz
    return None


def material_of(checkpoint_key, entry):
    return entry.get("material", "N87") if checkpoint_key.startswith("material:") else "N87"


def load_split(data_dir, split):
    data_dir = Path(data_dir)
    data = np.load(data_dir / f"{split}.npy")
    lengths = np.load(data_dir / f"{split}_lengths.npy").astype(np.int64)
    deltas = np.load(data_dir / f"{split}_delta_t.npy").astype(np.float32)
    config = json.loads((data_dir / "process_config.json").read_text(encoding="utf-8"))
    return data, lengths, deltas, config


def group_rows_by_length(lengths):
    groups = {}
    for row_idx, length in enumerate(lengths.tolist()):
        groups.setdefault(int(length), []).append(row_idx)
    return dict(sorted(groups.items()))


def run_with_oom_retry(fn, batch_size, device):
    """Call ``fn(batch_size)``, halving the batch on CUDA out-of-memory.

    A smaller batch can change the last bits of a GPU reduction, so a retry is
    reported rather than silently absorbed.
    """
    while True:
        try:
            return fn(batch_size)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(batch_size // 2, 1)
            print(f"CUDA out of memory; retrying with batch size {batch_size}", flush=True)


def rollout_variable_dt(model, data, lengths, dt_steps, original_indices, *, warmup, device,
                        batch_size):
    """Per-row relative RMS on a grid with one dt per step."""
    row_errors = np.full(len(data), np.nan, dtype=np.float64)
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        chunk_lengths = lengths[start:end]
        chunk_len = int(chunk_lengths.max())
        input_seq = torch.from_numpy(np.ascontiguousarray(data[start:end, :chunk_len, :-1])).float().to(device)
        h_gt = torch.from_numpy(np.ascontiguousarray(data[start:end, :chunk_len, -1:])).float().to(device)
        dt = torch.from_numpy(
            np.ascontiguousarray(dt_steps[start:end, : max(chunk_len - 1, 1)])
        ).float().to(device)
        indices = torch.from_numpy(np.ascontiguousarray(original_indices[start:end, :chunk_len])).to(device)
        lengths_tensor = torch.from_numpy(chunk_lengths).to(device)
        with torch.inference_mode():
            pred, _target_mask, eval_mask = rollout_sequence_original_grid(
                model, input_seq, h_gt, dt, indices,
                warmup_original_steps=warmup, device=device, lengths=lengths_tensor,
            )
            _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)
        row_errors[start:end] = row_err.detach().cpu().numpy().astype(np.float64)
        del input_seq, h_gt, dt, indices, lengths_tensor, pred, eval_mask, row_err
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row_errors


def stack_seed_grids(data, lengths, deltas, row_indices, seeds, max_step):
    """One random grid per seed, stacked so the recurrence runs once per group."""
    prepared = [make_random_interval_batch(data, lengths, deltas, row_indices, seed, max_step)
                for seed in seeds]
    grid_lengths = np.concatenate([item[1] for item in prepared])
    longest = int(grid_lengths.max())
    stacked = np.zeros((len(grid_lengths), longest, data.shape[2]), dtype=np.float32)
    stacked_indices = np.zeros((len(grid_lengths), longest), dtype=np.int64)
    stacked_dt = np.zeros((len(grid_lengths), max(longest - 1, 1)), dtype=np.float32)
    offset = 0
    per_seed_indices = []
    for padded, grid_len, dt_steps, indices, _stats in prepared:
        count, width = len(grid_len), padded.shape[1]
        stacked[offset:offset + count, :width] = padded
        stacked_indices[offset:offset + count, :width] = indices
        stacked_dt[offset:offset + count, : max(width - 1, 1)] = dt_steps[:, : max(width - 1, 1)]
        per_seed_indices.append([indices[i, : grid_len[i]] for i in range(count)])
        offset += count
    return stacked, grid_lengths, stacked_dt, stacked_indices, per_seed_indices


def grid_statistics(per_row_indices, source_lengths):
    """Retained length, retention ratio, and pooled step statistics of one grid."""
    retained = np.array([len(indices) for indices in per_row_indices], dtype=np.float64)
    steps = np.concatenate([np.diff(indices) for indices in per_row_indices if len(indices) > 1])
    return {
        "retained_length_mean": float(retained.mean()),
        "retention_ratio_mean": float(np.mean(retained / np.asarray(source_lengths, dtype=np.float64))),
        "step_min": int(steps.min()),
        "step_max": int(steps.max()),
        "step_mean": float(steps.mean()),
    }


def summarize_seeds(errors_by_seed):
    means = np.array([np.mean(e[np.isfinite(e)]) for e in errors_by_seed], dtype=np.float64)
    p95s = np.array([np.percentile(e[np.isfinite(e)], 95) for e in errors_by_seed], dtype=np.float64)
    ddof = 1 if len(means) > 1 else 0
    return {
        "mean_err": float(means.mean()),
        "mean_err_std": float(means.std(ddof=ddof)),
        "p95_err": float(p95s.mean()),
        "p95_err_std": float(p95s.std(ddof=ddof)),
        "per_seed_mean_err": means.tolist(),
        "per_seed_p95_err": p95s.tolist(),
    }


def evaluate_uniform(model, data, lengths, deltas, row_indices, stride, *, warmup, device, batch_size):
    grid, grid_lengths, grid_deltas = make_downsampled_batch(data, lengths, deltas, row_indices, stride)
    strides = np.full(len(row_indices), stride, dtype=np.int64)
    mean, p95, errors = run_with_oom_retry(
        lambda size: evaluate_sequence_error(
            model, grid, grid_lengths, grid_deltas, original_strides=strides,
            warmup=warmup, batch_size=size, device=device,
        ),
        batch_size, device,
    )
    return {"mean_err": mean, "p95_err": p95, "row_errors": errors}


def evaluate_groups(model, data, lengths, deltas, *, material, train_max_len, seeds=DEFAULT_SEEDS,
                    warmup=DEFAULT_WARMUP, max_step=None, include_unresampled=False, lengths_filter=None,
                    original_comparison=True, uniform_comparison=False, device, batch_size=4096,
                    original_batch_size=512, log=print):
    """Evaluate every frequency group of one split; returns summary and row records."""
    summary, groups_detail, row_records = [], {}, []
    for length, row_indices in group_rows_by_length(lengths).items():
        if lengths_filter and length not in lengths_filter:
            continue
        factor = int(compute_resample_stride(length, train_max_len))
        if factor <= 1 and not include_unresampled:
            continue
        group_max_step = int(max_step) if max_step is not None else max(factor, 1)
        khz = frequency_khz(length)

        stacked, grid_lengths, stacked_dt, stacked_indices, per_seed_indices = stack_seed_grids(
            data, lengths, deltas, row_indices, seeds, group_max_step
        )
        stacked_errors = run_with_oom_retry(
            lambda size: rollout_variable_dt(
                model, stacked, grid_lengths, stacked_dt, stacked_indices,
                warmup=warmup, device=device, batch_size=size,
            ),
            batch_size, device,
        )
        del stacked, stacked_dt, stacked_indices
        count = len(row_indices)
        errors_by_seed = [stacked_errors[i * count:(i + 1) * count] for i in range(len(seeds))]
        stats = summarize_seeds(errors_by_seed)
        grid_stats = grid_statistics(per_seed_indices[0], lengths[row_indices])
        for seed_indices in per_seed_indices:
            seed_stats = grid_statistics(seed_indices, lengths[row_indices])
            if seed_stats["step_min"] < 1 or seed_stats["step_max"] > group_max_step:
                raise RuntimeError(f"Random step outside [1, {group_max_step}]: {seed_stats}")

        record = {
            "material": material,
            "frequency_khz": khz if khz is not None else "",
            "original_length": int(length),
            "row_count": count,
            "training_downsampling_factor": factor,
            "max_step": group_max_step,
            "warmup_original_steps": int(warmup),
            # Grid descriptors are those of the first seed.
            "retained_length_mean": f"{grid_stats['retained_length_mean']:.1f}",
            "retention_ratio_mean": f"{grid_stats['retention_ratio_mean']:.4f}",
            "step_min": grid_stats["step_min"],
            "step_max": grid_stats["step_max"],
            "step_mean": f"{grid_stats['step_mean']:.4f}",
            "mean_err": f"{stats['mean_err']:.6f}",
            "mean_err_std": f"{stats['mean_err_std']:.6f}",
            "p95_err": f"{stats['p95_err']:.6f}",
            "p95_err_std": f"{stats['p95_err_std']:.6f}",
        }
        detail = {"frequency_khz": khz, "training_downsampling_factor": factor,
                  "max_step": group_max_step, "row_count": count, "random": stats,
                  "grid_first_seed": grid_stats}
        original_errors = None
        if uniform_comparison:
            uniform = evaluate_uniform(model, data, lengths, deltas, row_indices, max(factor, 1),
                                       warmup=warmup, device=device, batch_size=batch_size)
            record["uniform_mean_err"] = f"{uniform['mean_err']:.6f}"
            record["uniform_p95_err"] = f"{uniform['p95_err']:.6f}"
            record["random_over_uniform"] = f"{stats['mean_err'] / uniform['mean_err']:.4f}"
            detail["uniform_training_stride"] = {"mean_err": uniform["mean_err"], "p95_err": uniform["p95_err"]}
        if original_comparison:
            original = evaluate_uniform(model, data, lengths, deltas, row_indices, 1,
                                        warmup=warmup, device=device, batch_size=original_batch_size)
            original_errors = original["row_errors"]
            record["original_mean_err"] = f"{original['mean_err']:.6f}"
            record["original_p95_err"] = f"{original['p95_err']:.6f}"
            record["random_over_original"] = f"{stats['mean_err'] / original['mean_err']:.4f}"
            detail["original_grid"] = {"mean_err": original["mean_err"], "p95_err": original["p95_err"]}
        summary.append(record)
        groups_detail[str(length)] = detail

        for seed, seed_errors in zip(seeds, errors_by_seed):
            for position, row_idx in enumerate(row_indices):
                row_records.append({
                    "material": material, "frequency_khz": khz if khz is not None else "",
                    "original_length": int(length), "row_index": int(row_idx), "seed": int(seed),
                    "max_step": group_max_step,
                    "random_row_error": f"{float(seed_errors[position]):.8f}",
                    "original_row_error": (f"{float(original_errors[position]):.8f}"
                                           if original_errors is not None else ""),
                })
        message = (f"{material:>5s} {khz if khz else '?':>4} kHz len={length:>6d} rows={count:>4d} "
                   f"factor={factor} max_step={group_max_step} "
                   f"random={stats['mean_err'] * 100:.3f}+/-{stats['mean_err_std'] * 100:.3f}% "
                   f"p95={stats['p95_err'] * 100:.3f}%")
        if original_comparison:
            message += f" original={detail['original_grid']['mean_err'] * 100:.3f}%"
        log(message)
    return summary, groups_detail, row_records


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = [key for key in SUMMARY_FIELDS if any(key in row for row in rows)]
        fieldnames += [key for key in rows[0] if key not in fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protocol_metadata(args_like, train_max_len, process_config):
    return {
        "seeds": [int(seed) for seed in args_like.seeds],
        "warmup_original_steps": int(args_like.warmup),
        "step_cap": ("fixed max_step" if args_like.max_step is not None
                     else "group training downsampling factor ceil(length / max_len)"),
        "max_step": args_like.max_step,
        "train_max_len": int(train_max_len),
        "split": args_like.split,
        "processed_data": {key: process_config.get(key) for key in (
            "dt_norm_mode", "min_len", "max_len", "validation_split", "test_split", "split_seed")},
        "random_grid": ("retained points start at original index 0; each interval is an "
                        "independent random integer in [1, max_step]; stop before the next index "
                        "passes the end; the trailing partial interval is discarded"),
        "rng": "numpy default_rng(seed), one generator per seed and frequency group, rows in index order",
        "db_dt": "recomputed after sampling as delta B_norm / per-step dt",
        "warmup": "measured H is fed while the input's original-grid index is below the warmup",
        "error": ("per-sequence relative RMS after warmup; mean and P95 over sequences per seed, "
                  "then mean and sample std (ddof=1) over seeds"),
        "original_grid": "same rows on the measured grid (stride 1), same warmup and error",
    }


def build_parser(entries):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", choices=entries, default="n87:rategru")
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="process.py output for the checkpoint's material and normalization.")
    parser.add_argument("--split", default="origin_test", choices=["origin_test", "origin_val"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Warmup in original sampling points.")
    parser.add_argument("--max-step", type=int, default=None,
                        help="Fixed random-step cap for every group; default is each group's training factor.")
    parser.add_argument("--train-max-len", type=int, default=None,
                        help="Training max_len used to derive the factors; default is the processed split's.")
    parser.add_argument("--lengths", type=int, nargs="+", default=None,
                        help="Restrict to these original sequence lengths (e.g. 32016).")
    parser.add_argument("--include-unresampled-groups", action="store_true",
                        help="Also evaluate groups whose training factor is 1.")
    parser.add_argument("--uniform-comparison", action="store_true",
                        help="Also evaluate a uniform grid at each group's training stride.")
    parser.add_argument("--no-original-comparison", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Rows per rollout chunk of the random grids (all seeds of a group).")
    parser.add_argument("--original-batch-size", type=int, default=512,
                        help="Rows per rollout chunk on the original grid.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/random_interval_groups/<checkpoint>.")
    return parser


def main():
    entries = load_checkpoint_entries()
    args = build_parser(entries).parse_args()
    entry = entries[args.checkpoint]
    check_normalization(args.processed_data_dir, entry)
    device = resolve_device(args.device)
    model = load_model(entry, device)
    data, lengths, deltas, process_config = load_split(args.processed_data_dir, args.split)
    train_max_len = args.train_max_len or int(process_config["max_len"])
    material = material_of(args.checkpoint, entry)
    print(f"checkpoint={args.checkpoint} ({entry['path']}) device={device}", flush=True)

    summary, groups, row_records = evaluate_groups(
        model, data, lengths, deltas, material=material, train_max_len=train_max_len,
        seeds=args.seeds, warmup=args.warmup, max_step=args.max_step,
        include_unresampled=args.include_unresampled_groups, lengths_filter=args.lengths,
        original_comparison=not args.no_original_comparison,
        uniform_comparison=args.uniform_comparison, device=device,
        batch_size=args.batch_size, original_batch_size=args.original_batch_size,
        log=lambda message: print(message, flush=True),
    )
    output_dir = args.output_dir or (
        REPOSITORY_ROOT / "results" / "random_interval_groups" / args.checkpoint.replace(":", "_")
    )
    write_csv(output_dir / "random_interval_summary.csv", summary)
    write_csv(output_dir / "random_interval_row_errors.csv", row_records)
    (output_dir / "random_interval_summary.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "checkpoint_path": entry["path"],
        "checkpoint_sha256": entry["sha256"],
        "material": material,
        "protocol": protocol_metadata(args, train_max_len, process_config),
        "device": str(device),
        "groups": groups,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output_dir}", flush=True)


if __name__ == "__main__":
    main()
