import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import get_model_entry
from scripts.waveform_interp import DEFAULT_METHOD, INTERP_METHODS, interpolate_channel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "rategru"
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"
DEFAULT_FACTORS = [2, 5, 10, 20, 50]
VALID_LENGTH_GROUPS = ("longest", "shortest")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the controlled ExpCap bs16 checkpoint on the longest N87 "
            "origin_test rows under linear-interpolation upsampling."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--factors", type=int, nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--start-factor", type=int, default=None)
    parser.add_argument("--end-factor", type=int, default=None)
    parser.add_argument("--length-group", choices=VALID_LENGTH_GROUPS, default="longest")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-max-points", type=int, default=6000)
    parser.add_argument("--no-waveforms", action="store_true")
    parser.add_argument(
        "--interp",
        choices=INTERP_METHODS,
        default=DEFAULT_METHOD,
        help="B/H/T interpolation used to build the upsampled grid; dB/dt stays a forward "
             "difference. Use 'linear' to reproduce results from before the C^2 switch.",
    )
    return parser.parse_args()


def resolve_factors(args):
    if args.start_factor is not None or args.end_factor is not None:
        if args.start_factor is None or args.end_factor is None:
            raise ValueError("--start-factor and --end-factor must be used together")
        if args.start_factor < 1 or args.end_factor < args.start_factor:
            raise ValueError("invalid factor range")
        return list(range(args.start_factor, args.end_factor + 1))

    factors = sorted(dict.fromkeys(int(factor) for factor in args.factors))
    if any(factor < 1 for factor in factors):
        raise ValueError("upsampling factors must be >= 1")
    return factors


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


def setup_output_dir(path, length_group):
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ROOT / "results" / f"local_n87_{length_group}_upsample_exp16_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def select_length_group(lengths, length_group):
    if length_group == "longest":
        selected_len = int(lengths.max())
    elif length_group == "shortest":
        selected_len = int(lengths.min())
    else:
        raise ValueError(f"Unsupported length group: {length_group}")
    selected_row_indices = np.flatnonzero(lengths == selected_len).astype(int).tolist()
    if not selected_row_indices:
        raise RuntimeError(f"No rows selected for length group: {length_group}")
    return selected_len, selected_row_indices


def make_upsampled_batch(data, lengths, deltas, row_indices, factor, interp_method=DEFAULT_METHOD):
    if factor < 1:
        raise ValueError("factor must be >= 1")

    rows = []
    new_lengths = []
    new_deltas = []
    for row_idx in row_indices:
        real_len = int(lengths[row_idx])
        source = data[row_idx, :real_len, :]
        if factor == 1:
            sampled = source.astype(np.float32, copy=True)
        else:
            new_len = (real_len - 1) * factor + 1
            original_x = np.arange(real_len, dtype=np.float64)
            upsampled_x = np.linspace(0.0, float(real_len - 1), new_len, dtype=np.float64)
            sampled = np.empty((new_len, source.shape[1]), dtype=np.float32)
            for channel in (0, 2, source.shape[1] - 1):
                sampled[:, channel] = interpolate_channel(
                    original_x, source[:, channel], upsampled_x, interp_method
                ).astype(np.float32)

        dt_eff = float(deltas[row_idx]) / float(factor)
        b_norm = sampled[:, 0]
        d_b_dt = np.zeros_like(b_norm)
        if len(b_norm) > 1:
            d_b_dt[:-1] = (b_norm[1:] - b_norm[:-1]) / dt_eff
        sampled[:, 1] = d_b_dt

        rows.append(sampled)
        new_lengths.append(len(sampled))
        new_deltas.append(dt_eff)

    max_len = max(new_lengths)
    padded = np.zeros((len(rows), max_len, data.shape[2]), dtype=np.float32)
    for i, row in enumerate(rows):
        padded[i, : len(row), :] = row

    return padded, np.array(new_lengths, dtype=np.int64), np.array(new_deltas, dtype=np.float32)


def evaluate_batch_streaming(
    model,
    batch_data,
    batch_lengths,
    batch_deltas,
    warmup_steps,
    device,
    representative_position=None,
):
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    lengths_tensor = torch.from_numpy(batch_lengths).to(device=device, dtype=torch.long)
    dt = torch.from_numpy(batch_deltas).float().to(device).view(-1, 1)

    batch_size, seq_len, _feature_count = input_seq.shape
    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_size, model.hidden_size, device=device)
    sum_sq_err = torch.zeros(batch_size, device=device)
    sum_sq_tgt = torch.zeros(batch_size, device=device)
    count = torch.zeros(batch_size, device=device)

    rep_pred = None
    if representative_position is not None:
        rep_pred = np.empty(seq_len, dtype=np.float32)
        rep_pred[0] = float(h_curr[representative_position, 0].detach().cpu())

    with torch.inference_mode():
        for t in range(max(seq_len - 1, 0)):
            valid_step = t < (lengths_tensor - 1)
            if not bool(valid_step.any()):
                break

            input_t = input_seq[:, t, :]
            gt_t = h_gt[:, t, :]
            if t < warmup_steps:
                h_input = torch.where(valid_step.unsqueeze(1), gt_t, h_curr)
            else:
                h_input = h_curr

            h_next, hs_next = model.forward_step(input_t, h_input, hs, dt=dt)
            valid_step_exp = valid_step.unsqueeze(1)
            hs = torch.where(valid_step_exp, hs_next, hs)
            h_curr = torch.where(valid_step_exp, h_next, h_curr)

            pred_index = t + 1
            if pred_index > warmup_steps:
                valid_eval = pred_index < lengths_tensor
                if bool(valid_eval.any()):
                    target = h_gt[:, pred_index, 0]
                    pred = h_curr[:, 0]
                    mask = valid_eval.float()
                    sum_sq_err += ((pred - target) ** 2) * mask
                    sum_sq_tgt += (target**2) * mask
                    count += mask

            if rep_pred is not None:
                rep_pred[pred_index] = float(h_curr[representative_position, 0].detach().cpu())

    row_err = torch.full_like(sum_sq_err, float("nan"))
    valid = count > 0
    if valid.any():
        row_err[valid] = torch.sqrt(sum_sq_err[valid] / count[valid]) / torch.sqrt(sum_sq_tgt[valid] / count[valid])

    return row_err.detach().cpu().numpy().astype(np.float64), rep_pred


def evaluate_equal_length_batch_streaming(
    model,
    batch_data,
    batch_deltas,
    warmup_steps,
    device,
    representative_position=None,
):
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    dt = torch.from_numpy(batch_deltas).float().to(device).view(-1, 1)

    batch_size, seq_len, _feature_count = input_seq.shape
    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_size, model.hidden_size, device=device)
    sum_sq_err = torch.zeros(batch_size, device=device)
    sum_sq_tgt = torch.zeros(batch_size, device=device)
    count = max(seq_len - warmup_steps - 1, 0)

    rep_pred = None
    if representative_position is not None:
        rep_pred = np.empty(seq_len, dtype=np.float32)
        rep_pred[0] = float(h_curr[representative_position, 0].detach().cpu())

    with torch.inference_mode():
        for t in range(max(seq_len - 1, 0)):
            h_input = h_gt[:, t, :] if t < warmup_steps else h_curr
            h_curr, hs = model.forward_step(input_seq[:, t, :], h_input, hs, dt=dt)

            pred_index = t + 1
            if pred_index > warmup_steps:
                target = h_gt[:, pred_index, 0]
                pred = h_curr[:, 0]
                sum_sq_err += (pred - target) ** 2
                sum_sq_tgt += target**2

            if rep_pred is not None:
                rep_pred[pred_index] = float(h_curr[representative_position, 0].detach().cpu())

    row_err = torch.full_like(sum_sq_err, float("nan"))
    if count > 0:
        row_err = torch.sqrt(sum_sq_err / count) / torch.sqrt(sum_sq_tgt / count)

    return row_err.detach().cpu().numpy().astype(np.float64), rep_pred


def save_summary_csv(path, rows, selected_row_indices):
    row_fields = [f"row_err_{idx}" for idx in selected_row_indices]
    fieldnames = [
        "factor",
        "effective_length_min",
        "effective_length_max",
        "warmup_steps",
        "dt_eff_min",
        "dt_eff_max",
        "mean_err",
        "p95_err",
    ] + row_fields
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            for key in ["mean_err", "p95_err", "dt_eff_min", "dt_eff_max"]:
                if out[key] != "":
                    out[key] = f"{float(out[key]):.10g}"
            writer.writerow(out)


def save_error_curve(output_dir, rows, length_group):
    factors = [row["factor"] for row in rows]
    mean_err = [row["mean_err"] for row in rows]
    p95_err = [row["p95_err"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(factors, mean_err, marker="o", linewidth=2, label="Mean relative RMS")
    ax.plot(factors, p95_err, marker="s", linewidth=2, label="P95 relative RMS")
    ax.set_xlabel("Upsampling factor")
    ax.set_ylabel("Error")
    ax.set_title(f"N87 {length_group} internal test rows under upsampling")
    major_xticks = [factor for factor in factors if factor <= 5 or factor % 5 == 0]
    minor_xticks = [factor for factor in factors if factor not in set(major_xticks)]
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"upsample_error_curve.{suffix}")
    plt.close(fig)


def save_waveform_plot(output_dir, factor, h_true, h_pred, warmup_steps, row_idx, row_err, max_points):
    steps = np.arange(len(h_true))
    if max_points > 0 and len(steps) > max_points:
        stride = int(math.ceil(len(steps) / max_points))
        steps = steps[::stride]
        h_true = h_true[::stride]
        h_pred = h_pred[::stride]

    fig, ax = plt.subplots(figsize=(16, 4.2), dpi=160)
    ax.plot(steps, h_true, linewidth=1.6, label="Interpolated true H")
    ax.plot(steps, h_pred, linewidth=1.2, linestyle="--", label="Pred H")
    ax.axvline(warmup_steps, color="#b00020", linestyle=":", linewidth=1.2, label="Warmup end")
    ax.set_xlabel("Upsampled step")
    ax.set_ylabel("H")
    ax.set_title(f"Waveform, factor={factor}, row={row_idx}, err={row_err:.5f}")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"waveform_factor_{factor:02d}.png")
    plt.close(fig)


def relative_rms(pred, target, warmup_steps, length):
    start = min(int(warmup_steps) + 1, int(length))
    if start >= length:
        return float("nan")
    pred_slice = pred[start:length]
    target_slice = target[start:length]
    rms_err = np.sqrt(np.mean((pred_slice - target_slice) ** 2))
    rms_target = np.sqrt(np.mean(target_slice**2))
    return float(rms_err / rms_target) if rms_target > 0 else float("nan")


def main():
    args = parse_args()
    factors = resolve_factors(args)
    device = resolve_device(args.device)
    output_dir = setup_output_dir(args.output_dir, args.length_group)

    model, run_config, model_path = load_model(args.run_dir, device)
    data, lengths, deltas, norm_stats = load_origin_test(args.data_dir)

    selected_len, selected_row_indices = select_length_group(lengths, args.length_group)

    std_h = float(norm_stats["std_h"])
    base_warmup = int(run_config["train"]["eval_warmup_len"])

    summary_rows = []
    representative_row_idx = None
    representative_position = None

    for factor in factors:
        batch_data, batch_lengths, batch_deltas = make_upsampled_batch(
            data, lengths, deltas, selected_row_indices, factor, args.interp
        )
        warmup_steps = int(math.ceil(base_warmup * factor))
        if representative_position is None:
            representative_position = 0
            representative_row_idx = int(selected_row_indices[representative_position])

        evaluator = (
            evaluate_equal_length_batch_streaming
            if int(batch_lengths.min()) == int(batch_lengths.max())
            else evaluate_batch_streaming
        )
        if evaluator is evaluate_equal_length_batch_streaming:
            row_err, rep_pred_norm = evaluator(
                model,
                batch_data,
                batch_deltas,
                warmup_steps,
                device,
                representative_position=None if args.no_waveforms else representative_position,
            )
        else:
            row_err, rep_pred_norm = evaluator(
                model,
                batch_data,
                batch_lengths,
                batch_deltas,
                warmup_steps,
                device,
                representative_position=None if args.no_waveforms else representative_position,
            )
        valid = row_err[~np.isnan(row_err)]
        mean_err = float(np.mean(valid)) if valid.size else float("nan")
        p95_err = float(np.percentile(valid, 95)) if valid.size else float("nan")

        row = {
            "factor": factor,
            "effective_length_min": int(batch_lengths.min()),
            "effective_length_max": int(batch_lengths.max()),
            "warmup_steps": warmup_steps,
            "dt_eff_min": float(batch_deltas.min()),
            "dt_eff_max": float(batch_deltas.max()),
            "mean_err": mean_err,
            "p95_err": p95_err,
        }
        for row_idx, err in zip(selected_row_indices, row_err):
            row[f"row_err_{row_idx}"] = float(err)
        summary_rows.append(row)

        if not args.no_waveforms:
            length = int(batch_lengths[representative_position])
            h_true = batch_data[representative_position, :length, -1] * std_h
            h_pred = rep_pred_norm[:length] * std_h
            rep_err = relative_rms(rep_pred_norm, batch_data[representative_position, :, -1], warmup_steps, length)
            save_waveform_plot(
                output_dir,
                factor,
                h_true,
                h_pred,
                warmup_steps,
                representative_row_idx,
                rep_err,
                args.plot_max_points,
            )

        print(
            f"factor={factor:02d} length={int(batch_lengths.max())} warmup={warmup_steps} "
            f"mean={mean_err:.5f} p95={p95_err:.5f}",
            flush=True,
        )
        del batch_data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_summary_csv(output_dir / "upsample_error_summary.csv", summary_rows, selected_row_indices)
    save_error_curve(output_dir, summary_rows, args.length_group)

    metadata = {
        "run_dir": str(args.run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "interp_method": args.interp,
        "model_key": run_config["model_key"],
        "run_name": run_config["run_name"],
        "data_name": run_config["data_name"],
        "dt_norm_mode": run_config["dt_norm_mode"],
        "batch_size": run_config["train"]["batch_size"],
        "lr_12": run_config["train"]["lr_12"],
        "lr_3": run_config["train"]["lr_3"],
        "base_eval_warmup_len": base_warmup,
        "warmup_policy": "ceil(base_eval_warmup_len * upsample_factor)",
        "upsample_policy": "linear interpolation of normalized B, normalized H target, and temperature feature",
        "db_dt_policy": "recomputed after upsampling using dB_norm / (origin_dt / factor)",
        "error_definition": "per-row relative RMS on post-warmup points; mean and P95 over selected rows",
        "length_group": args.length_group,
        "selected_length": selected_len,
        "selected_row_count": len(selected_row_indices),
        "selected_row_indices": selected_row_indices,
        "representative_row_idx": representative_row_idx,
        "device": str(device),
        "factors": factors,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
