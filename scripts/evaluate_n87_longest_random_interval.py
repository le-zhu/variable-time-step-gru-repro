import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import get_model_entry
from rollout import rollout_sequence_original_grid, sequence_relative_rms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "rategru"
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the controlled ExpCap bs16 checkpoint on the longest N87 "
            "origin_test rows using random 1..20-step sampling intervals."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-step", type=int, default=20)
    parser.add_argument("--warmup-original-steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path):
    with require_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_origin_test(data_dir):
    data = np.load(require_file(data_dir / "origin_test.npy"))
    lengths = np.load(require_file(data_dir / "origin_test_lengths.npy")).astype(np.int64)
    deltas = np.load(require_file(data_dir / "origin_test_delta_t.npy")).astype(np.float32)
    norm_stats = load_json(data_dir / "norm_stats.json")
    return data, lengths, deltas, norm_stats


def build_random_indices(real_len, rng, max_step):
    indices = [0]
    while True:
        step = int(rng.integers(1, max_step + 1))
        next_idx = indices[-1] + step
        if next_idx >= real_len:
            break
        indices.append(next_idx)
    return np.array(indices, dtype=np.int64)


def make_random_interval_batch(data, lengths, deltas, row_indices, seed, max_step):
    if max_step < 1:
        raise ValueError("max_step must be >= 1")

    rng = np.random.default_rng(seed)
    rows = []
    sample_indices = []
    dt_steps_list = []
    new_lengths = []
    step_mins = []
    step_maxs = []
    step_means = []

    for row_idx in row_indices:
        real_len = int(lengths[row_idx])
        source = data[row_idx, :real_len, :]
        indices = build_random_indices(real_len, rng, max_step)
        sampled = source[indices, :].astype(np.float32, copy=True)
        steps = np.diff(indices).astype(np.float32)
        dt_steps = steps * float(deltas[row_idx])

        b_norm = sampled[:, 0]
        d_b_dt = np.zeros_like(b_norm)
        if len(b_norm) > 1:
            d_b_dt[:-1] = (b_norm[1:] - b_norm[:-1]) / dt_steps
        sampled[:, 1] = d_b_dt

        rows.append(sampled)
        sample_indices.append(indices)
        dt_steps_list.append(dt_steps.astype(np.float32))
        new_lengths.append(len(indices))
        if len(steps):
            step_mins.append(int(steps.min()))
            step_maxs.append(int(steps.max()))
            step_means.append(float(steps.mean()))

    max_len = max(new_lengths)
    padded = np.zeros((len(rows), max_len, data.shape[2]), dtype=np.float32)
    padded_indices = np.zeros((len(rows), max_len), dtype=np.int64)
    padded_dt_steps = np.zeros((len(rows), max(max_len - 1, 1)), dtype=np.float32)

    for i, row in enumerate(rows):
        length = len(row)
        padded[i, :length, :] = row
        padded_indices[i, :length] = sample_indices[i]
        steps_len = len(dt_steps_list[i])
        if steps_len:
            padded_dt_steps[i, :steps_len] = dt_steps_list[i]

    stats = {
        "sample_length_min": int(np.min(new_lengths)),
        "sample_length_max": int(np.max(new_lengths)),
        "sample_length_mean": float(np.mean(new_lengths)),
        "step_min": int(np.min(step_mins)) if step_mins else 0,
        "step_max": int(np.max(step_maxs)) if step_maxs else 0,
        "step_mean": float(np.mean(step_means)) if step_means else 0.0,
    }
    return (
        padded,
        np.array(new_lengths, dtype=np.int64),
        padded_dt_steps,
        padded_indices,
        stats,
    )


def rollout_sequence_variable_dt(
    model,
    input_seq,
    h_gt,
    dt_steps,
    original_indices,
    lengths,
    *,
    warmup_original_steps,
    device,
):
    pred, _target_mask, eval_mask = rollout_sequence_original_grid(
        model,
        input_seq,
        h_gt,
        dt_steps,
        original_indices,
        warmup_original_steps=warmup_original_steps,
        device=device,
        lengths=lengths,
    )
    return pred, eval_mask


def evaluate_batch(model, batch_data, batch_lengths, dt_steps, original_indices, warmup_original_steps, device):
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    dt_steps_tensor = torch.from_numpy(dt_steps).float().to(device)
    original_indices_tensor = torch.from_numpy(original_indices).to(device)
    lengths_tensor = torch.from_numpy(batch_lengths).to(device)

    with torch.no_grad():
        pred, eval_mask = rollout_sequence_variable_dt(
            model,
            input_seq,
            h_gt,
            dt_steps_tensor,
            original_indices_tensor,
            lengths_tensor,
            warmup_original_steps=warmup_original_steps,
            device=device,
        )
        _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)

    return (
        row_err.detach().cpu().numpy().astype(np.float64),
        pred.detach().cpu().numpy().squeeze(-1),
        h_gt.detach().cpu().numpy().squeeze(-1),
    )


def relative_rms(pred, target, original_indices, warmup_original_steps, length):
    mask = np.zeros(int(length), dtype=bool)
    if length > 1:
        mask[1:] = original_indices[: length - 1] >= warmup_original_steps
    if not np.any(mask):
        return float("nan")
    pred_slice = pred[:length][mask]
    target_slice = target[:length][mask]
    rms_err = np.sqrt(np.mean((pred_slice - target_slice) ** 2))
    rms_target = np.sqrt(np.mean(target_slice ** 2))
    return float(rms_err / rms_target) if rms_target > 0 else float("nan")


def setup_output_dir(path):
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ROOT / "results" / f"local_n87_longest_random_interval_exp16_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_summary_csv(path, rows, selected_row_indices):
    row_fields = [f"row_err_{idx}" for idx in selected_row_indices]
    fieldnames = [
        "seed",
        "sample_length_min",
        "sample_length_max",
        "sample_length_mean",
        "step_min",
        "step_max",
        "step_mean",
        "mean_err",
        "p95_err",
    ] + row_fields
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            for key in ["sample_length_mean", "step_mean", "mean_err", "p95_err"]:
                if out[key] != "":
                    out[key] = f"{float(out[key]):.10g}"
            writer.writerow(out)


def save_seed_error_plot(output_dir, rows, max_step):
    seeds = [row["seed"] for row in rows]
    mean_err = [row["mean_err"] for row in rows]
    p95_err = [row["p95_err"] for row in rows]
    x = np.arange(len(seeds))
    width = 0.22
    offset = 0.18

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.bar(x - offset, mean_err, width=width, color="#4C78A8", label="Mean relative RMS")
    ax.bar(x + offset, p95_err, width=width, color="#F58518", label="P95 relative RMS")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Error")
    ax.set_title(f"N87 longest rows under random 1..{max_step}-step sampling")
    ax.set_xticks(x, seeds)
    ax.set_ylim(0.05, 0.09)
    ax.set_yticks(np.arange(0.05, 0.091, 0.01))
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"random_interval_seed_errors.{suffix}")
    plt.close(fig)


def save_waveform_plot(output_dir, original_index, h_true, h_pred, warmup_original_steps, seed, row_idx, row_err, max_step):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.plot(original_index, h_true, linewidth=1.6, label="True H")
    ax.plot(original_index, h_pred, linewidth=1.2, linestyle="--", label="Pred H")
    ax.axvline(warmup_original_steps, color="#b00020", linestyle=":", linewidth=1.2, label="Warmup end")
    ax.set_xlabel("Original sample index")
    ax.set_ylabel("H")
    ax.set_title(f"Random 1..{max_step}-step waveform, seed={seed}, row={row_idx}, err={row_err:.5f}")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "waveform_worst_seed.png")
    plt.close(fig)


def aggregate_rows(rows):
    mean_values = np.array([row["mean_err"] for row in rows], dtype=np.float64)
    p95_values = np.array([row["p95_err"] for row in rows], dtype=np.float64)
    ddof = 1 if len(rows) > 1 else 0
    return {
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "mean_err_mean": float(np.mean(mean_values)),
        "mean_err_std": float(np.std(mean_values, ddof=ddof)),
        "p95_err_mean": float(np.mean(p95_values)),
        "p95_err_std": float(np.std(p95_values, ddof=ddof)),
        "worst_seed_by_mean_err": int(rows[int(np.argmax(mean_values))]["seed"]),
        "worst_seed_by_p95_err": int(rows[int(np.argmax(p95_values))]["seed"]),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = setup_output_dir(args.output_dir)

    model, run_config, model_path = load_model(args.run_dir, device)
    data, lengths, deltas, norm_stats = load_origin_test(args.data_dir)

    max_len = int(lengths.max())
    selected_row_indices = np.flatnonzero(lengths == max_len).astype(int).tolist()
    if len(selected_row_indices) != 23 or max_len != 32016:
        raise RuntimeError(
            f"Expected 23 longest rows of length 32016, got {len(selected_row_indices)} rows length {max_len}."
        )

    std_h = float(norm_stats["std_h"])
    warmup_original_steps = (
        int(run_config["train"]["eval_warmup_len"])
        if args.warmup_original_steps is None
        else int(args.warmup_original_steps)
    )

    summary_rows = []
    per_seed_outputs = {}

    for seed in args.seeds:
        batch_data, batch_lengths, dt_steps, original_indices, sample_stats = make_random_interval_batch(
            data, lengths, deltas, selected_row_indices, seed, args.max_step
        )
        if sample_stats["step_min"] < 1 or sample_stats["step_max"] > args.max_step:
            raise RuntimeError(f"Invalid random step range for seed {seed}: {sample_stats}")

        row_err, pred_norm, h_true_norm = evaluate_batch(
            model,
            batch_data,
            batch_lengths,
            dt_steps,
            original_indices,
            warmup_original_steps,
            device,
        )
        valid = row_err[~np.isnan(row_err)]
        mean_err = float(np.mean(valid)) if valid.size else float("nan")
        p95_err = float(np.percentile(valid, 95)) if valid.size else float("nan")

        row = {
            "seed": int(seed),
            **sample_stats,
            "mean_err": mean_err,
            "p95_err": p95_err,
        }
        for row_idx, err in zip(selected_row_indices, row_err):
            row[f"row_err_{row_idx}"] = float(err)
        summary_rows.append(row)
        per_seed_outputs[int(seed)] = {
            "batch_lengths": batch_lengths,
            "original_indices": original_indices,
            "row_err": row_err,
            "pred_norm": pred_norm,
            "h_true_norm": h_true_norm,
        }
        print(
            f"seed={seed} length={sample_stats['sample_length_min']}..{sample_stats['sample_length_max']} "
            f"step={sample_stats['step_min']}..{sample_stats['step_max']} "
            f"mean={mean_err:.5f} p95={p95_err:.5f}"
        )

    aggregate = aggregate_rows(summary_rows)
    worst_seed = aggregate["worst_seed_by_mean_err"]
    worst_output = per_seed_outputs[worst_seed]
    worst_position = int(np.nanargmax(worst_output["row_err"]))
    worst_row_idx = int(selected_row_indices[worst_position])
    worst_length = int(worst_output["batch_lengths"][worst_position])
    original_index = worst_output["original_indices"][worst_position, :worst_length]
    h_true = worst_output["h_true_norm"][worst_position, :worst_length] * std_h
    h_pred = worst_output["pred_norm"][worst_position, :worst_length] * std_h
    worst_row_err = relative_rms(
        worst_output["pred_norm"][worst_position],
        worst_output["h_true_norm"][worst_position],
        worst_output["original_indices"][worst_position],
        warmup_original_steps,
        worst_length,
    )

    save_summary_csv(output_dir / "random_interval_error_summary.csv", summary_rows, selected_row_indices)
    with (output_dir / "random_interval_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    save_seed_error_plot(output_dir, summary_rows, args.max_step)
    save_waveform_plot(
        output_dir,
        original_index,
        h_true,
        h_pred,
        warmup_original_steps,
        worst_seed,
        worst_row_idx,
        worst_row_err,
        args.max_step,
    )

    metadata = {
        "run_dir": str(args.run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_key": run_config["model_key"],
        "run_name": run_config["run_name"],
        "data_name": run_config["data_name"],
        "dt_norm_mode": run_config["dt_norm_mode"],
        "batch_size": run_config["train"]["batch_size"],
        "lr_12": run_config["train"]["lr_12"],
        "lr_3": run_config["train"]["lr_3"],
        "random_sampling_policy": "start at 0, repeatedly jump a random integer in [1, max_step], stop before overflow",
        "max_step": int(args.max_step),
        "seeds": [int(seed) for seed in args.seeds],
        "warmup_policy": "teacher forcing while current original sample index is less than warmup_original_steps",
        "warmup_original_steps": int(warmup_original_steps),
        "db_dt_policy": "recomputed after random sampling using dB_norm / per-step dt",
        "dt_policy": "variable per-step dt_t = origin_dt * sampled_index_gap",
        "error_start_policy": (
            "include outputs starting with the output produced by the first "
            "autoregressive H input"
        ),
        "error_definition": "per-row relative RMS on evaluation points; mean and P95 over selected rows",
        "selected_length": max_len,
        "selected_row_count": len(selected_row_indices),
        "selected_row_indices": selected_row_indices,
        "worst_seed_policy": "largest mean_err across repeated seeds",
        "worst_seed": int(worst_seed),
        "worst_row_policy": "largest row_err within worst seed",
        "worst_row_idx": worst_row_idx,
        "device": str(device),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Aggregate mean={aggregate['mean_err_mean']:.5f} +/- {aggregate['mean_err_std']:.5f}")
    print(f"Aggregate p95={aggregate['p95_err_mean']:.5f} +/- {aggregate['p95_err_std']:.5f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
