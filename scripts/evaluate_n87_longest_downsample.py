import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import get_model_entry
from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_cache import load_plot_cache, save_plot_cache
from figures.paper_plot_style import paper_matplotlib_rc
from rollout import rollout_sequence_original_grid, sequence_relative_rms


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "figures" / "output"
PAPER_FIGURE_DIR = PAPER_DIR
PAPER_PLOT_DATA_DIR = PAPER_DIR / "plot_data"
DEFAULT_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "rategru"
DEFAULT_DATA_DIR = ROOT / "data" / "N87_processed"
PAPER_WAVEFORM_FACTORS = (16, 32)
PAPER_WAVEFORM_STEM = "uniform_waveforms_16_32"
PAPER_WAVEFORM_REFERENCE_FACTOR = 16
PAPER_WAVEFORM_REFERENCE_WINDOW = (1, 1000)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the released N87 RateGRU checkpoint on the longest origin_test rows under uniform downsampling."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-factor", type=int, default=1)
    # The manuscript waveform contrasts 16x with 32x, so the default range has
    # to reach 32 or save_selected_waveform_grid has nothing to draw.
    parser.add_argument("--end-factor", type=int, default=max(PAPER_WAVEFORM_FACTORS))
    parser.add_argument("--factors", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-waveforms", action="store_true")
    parser.add_argument("--plot-only-cache", type=Path, default=None, help="Replot the paper waveform from a saved NPZ cache without model inference.")
    parser.add_argument("--normalized-bh-limits", action="store_true", help="Use shared H/std_h and B/std_b limits so zero aligns across both y axes.")
    parser.add_argument("--no-copy-to-paper", action="store_true", help="Do not copy the rendered manuscript waveform PDF into figures/output.")
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


def make_downsampled_batch(data, lengths, deltas, row_indices, factor):
    if factor < 1:
        raise ValueError("factor must be >= 1")

    rows = []
    new_lengths = []
    new_deltas = []
    for row_idx in row_indices:
        real_len = int(lengths[row_idx])
        source = data[row_idx, :real_len, :]
        sampled = source[::factor, :].astype(np.float32, copy=True)
        dt_eff = float(deltas[row_idx]) * factor

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


def evaluate_batch(
    model,
    batch_data,
    batch_lengths,
    batch_deltas,
    warmup_original_steps,
    original_stride,
    device,
):
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    dt = torch.from_numpy(batch_deltas).float().to(device).view(-1, 1)
    lengths_tensor = torch.from_numpy(batch_lengths).to(device)
    original_indices = (
        torch.arange(batch_data.shape[1], device=device, dtype=torch.long).unsqueeze(0)
        * int(original_stride)
    ).expand(batch_data.shape[0], -1)

    with torch.no_grad():
        pred, target_mask, eval_mask = rollout_sequence_original_grid(
            model,
            input_seq,
            h_gt,
            dt,
            original_indices,
            warmup_original_steps=warmup_original_steps,
            device=device,
            lengths=lengths_tensor,
        )
        _loss, row_err = sequence_relative_rms(pred, h_gt, eval_mask)

    return (
        row_err.detach().cpu().numpy().astype(np.float64),
        pred.detach().cpu().numpy().squeeze(-1),
        h_gt.detach().cpu().numpy().squeeze(-1),
    )


def relative_rms(pred, target, warmup_steps, length):
    start = min(int(warmup_steps) + 1, int(length))
    if start >= length:
        return float("nan")
    pred_slice = pred[start:length]
    target_slice = target[start:length]
    rms_err = np.sqrt(np.mean((pred_slice - target_slice) ** 2))
    rms_target = np.sqrt(np.mean(target_slice ** 2))
    return float(rms_err / rms_target) if rms_target > 0 else float("nan")


def setup_output_dir(path):
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ROOT / "results" / f"local_n87_longest_downsample_exp16_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def save_error_curve(output_dir, rows):
    factors = [row["factor"] for row in rows]
    mean_err = [row["mean_err"] for row in rows]
    p95_err = [row["p95_err"] for row in rows]
    xticks = [factor for factor in factors if factor <= 5 or factor in {10, 15, 20} or (factor > 20 and factor % 5 == 0)]
    minor_xticks = [factor for factor in factors if factor not in set(xticks)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.plot(factors, mean_err, marker="o", linewidth=2, label="Mean relative RMS")
    ax.plot(factors, p95_err, marker="s", linewidth=2, label="P95 relative RMS")
    ax.set_xlabel("Downsampling factor")
    ax.set_ylabel("Error")
    ax.set_title("N87 longest internal test rows under downsampling")
    ax.set_xticks(xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"downsample_error_curve.{suffix}")
    plt.close(fig)


def save_waveform_plot(output_dir, factor, h_true, h_pred, warmup_steps, row_idx, row_err):
    steps = np.arange(len(h_true))
    fig, ax = plt.subplots(figsize=(16, 4.2), dpi=160)
    ax.plot(steps, h_true, linewidth=1.6, label="True H")
    ax.plot(steps, h_pred, linewidth=1.2, linestyle="--", label="Pred H")
    ax.axvline(warmup_steps, color="#b00020", linestyle=":", linewidth=1.2, label="Warmup end")
    ax.set_xlabel("Downsampled step")
    ax.set_ylabel("H")
    ax.set_title(f"Waveform, factor={factor}, row={row_idx}, err={row_err:.5f}")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"waveform_factor_{factor:02d}.png")
    plt.close(fig)


def save_selected_waveform_grid(output_dir, per_factor_predictions, representative_position, representative_row_idx, std_b, std_h):
    missing = [factor for factor in PAPER_WAVEFORM_FACTORS if factor not in per_factor_predictions]
    if missing:
        raise RuntimeError(f"Cannot build paper waveform grid; missing factors: {missing}")

    plot_items = []
    h_values = []
    b_values = []
    for factor in PAPER_WAVEFORM_FACTORS:
        item = per_factor_predictions[factor]
        length = int(item["batch_lengths"][representative_position])
        b_true = item["batch_data"][representative_position, :length, 0] * std_b
        h_true = item["h_true_norm"][representative_position, :length] * std_h
        h_pred = item["pred_norm"][representative_position, :length] * std_h
        row_err = relative_rms(
            item["pred_norm"][representative_position],
            item["h_true_norm"][representative_position],
            int(item["warmup_steps"]),
            length,
        )
        plot_items.append(
            {
                "factor": factor,
                "steps": np.arange(length),
                "b_true": b_true,
                "h_true": h_true,
                "h_pred": h_pred,
                "row_err": row_err,
            }
        )
        h_values.extend([h_true, h_pred])
        b_values.append(b_true)

    save_uniform_waveform_cache(output_dir / f"{PAPER_WAVEFORM_STEM}_plot_data.npz", plot_items, representative_row_idx)
    PAPER_PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_uniform_waveform_cache(PAPER_PLOT_DATA_DIR / f"{PAPER_WAVEFORM_STEM}.npz", plot_items, representative_row_idx)
    render_selected_waveform_grid(output_dir, plot_items, representative_row_idx)


def padded_limits(values, pad_fraction=0.06):
    flat = np.concatenate([np.asarray(value).ravel() for value in values])
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high):
        pad = max(abs(low) * pad_fraction, 1.0)
    else:
        pad = (high - low) * pad_fraction
    return low - pad, high + pad


def normalized_bh_limits(h_values, b_values_mt, std_h: float, std_b: float):
    h_norm = [np.asarray(value, dtype=np.float64) / float(std_h) for value in h_values]
    b_norm = [np.asarray(value, dtype=np.float64) / (float(std_b) * 1000.0) for value in b_values_mt]
    z_low, z_high = padded_limits(h_norm + b_norm)
    return (z_low * float(std_h), z_high * float(std_h)), (
        z_low * float(std_b) * 1000.0,
        z_high * float(std_b) * 1000.0,
    )


def render_selected_waveform_grid(
    output_dir: Path,
    plot_items: list[dict[str, object]],
    representative_row_idx: int,
    *,
    normalized_bh_limits_enabled: bool = False,
    std_h: float | None = None,
    std_b: float | None = None,
    copy_to_paper: bool = True,
) -> None:
    ref_start, ref_end = PAPER_WAVEFORM_REFERENCE_WINDOW
    original_start = ref_start * PAPER_WAVEFORM_REFERENCE_FACTOR
    original_end = ref_end * PAPER_WAVEFORM_REFERENCE_FACTOR
    windowed_items = []
    for item in plot_items:
        factor = int(item["factor"])
        step_start = int(math.ceil(original_start / factor))
        step_end = int(math.floor(original_end / factor))
        steps = np.asarray(item["steps"])
        mask = (steps >= step_start) & (steps <= step_end)
        if not np.any(mask):
            raise RuntimeError(f"No waveform points remain for factor={factor} in the selected paper window.")
        clipped = dict(item)
        clipped["steps"] = steps[mask]
        clipped["b_true"] = np.asarray(item["b_true"])[mask]
        clipped["h_true"] = np.asarray(item["h_true"])[mask]
        clipped["h_pred"] = np.asarray(item["h_pred"])[mask]
        windowed_items.append(clipped)

    h_values = []
    b_values = []
    for item in windowed_items:
        h_values.extend([np.asarray(item["h_true"]), np.asarray(item["h_pred"])])
        b_values.append(1000.0 * np.asarray(item["b_true"]))
    if normalized_bh_limits_enabled:
        if std_h is None or std_b is None:
            raise ValueError("std_h and std_b are required for normalized H/B limits.")
        h_ylim, b_ylim = normalized_bh_limits(h_values, b_values, std_h, std_b)
    else:
        h_ylim = padded_limits(h_values)
        b_ylim = padded_limits(b_values)

    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        true_width = line_width(PAPER_WAVEFORM_STEM, 0.72)
        pred_width = line_width(PAPER_WAVEFORM_STEM, 0.6)
        b_width = line_width(PAPER_WAVEFORM_STEM, 0.42)
        grid_width = line_width(PAPER_WAVEFORM_STEM, 0.6)
        spine_width = line_width(PAPER_WAVEFORM_STEM, 0.8)
        tick_width = line_width(PAPER_WAVEFORM_STEM, 0.8)
        tick_length = marker_size(PAPER_WAVEFORM_STEM, 3.5)
        fig, axes = plt.subplots(2, 1, figsize=figure_size(PAPER_WAVEFORM_STEM), dpi=180, sharex=False)
        axes = axes.ravel()
        handles = None
        labels = None
        for ax, plot_item in zip(axes, windowed_items):
            true_line = ax.plot(plot_item["steps"], plot_item["h_true"], linewidth=true_width, color="#111111", label="True H")[0]
            pred_line = ax.plot(
                plot_item["steps"],
                plot_item["h_pred"],
                linewidth=pred_width,
                linestyle="-",
                color="#d62728",
                label="Pred H",
            )[0]
            b_ax = ax.twinx()
            b_line = b_ax.plot(
                plot_item["steps"],
                1000.0 * np.asarray(plot_item["b_true"]),
                linewidth=b_width,
                color="#6F768A",
                alpha=0.72,
                label="B input",
            )[0]
            ax.set_ylim(*h_ylim)
            b_ax.set_ylim(*b_ylim)
            b_ax.tick_params(axis="y", labelsize=8, labelcolor="#6F768A", pad=1, width=tick_width, length=tick_length)
            for spine in b_ax.spines.values():
                spine.set_linewidth(spine_width)
            if handles is None:
                handles = [true_line, pred_line, b_line]
                labels = [handle.get_label() for handle in handles]
            ax.set_title("")
            ax.text(
                0.5,
                0.02,
                f"{plot_item['factor']}\u00d7, row err = {100.0 * plot_item['row_err']:.2f}%",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(True, color="#e4e4e4", linewidth=grid_width)
            ax.tick_params(labelsize=8, pad=1, width=tick_width, length=tick_length)
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)

        fig.tight_layout(rect=[0.045, 0.095, 0.955, 0.875], pad=0.08, h_pad=0.24)
        if handles is not None and labels is not None:
            top_box = axes[0].get_position()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, top_box.y1 - 0.0287),
                fontsize=8,
            )
        bottom_box = axes[-1].get_position()
        fig.text(0.030, 0.51, "H (A/m)", ha="center", va="center", rotation=90, fontsize=8)
        fig.text(0.970, 0.51, "B (mT)", ha="center", va="center", rotation=90, color="#6F768A", fontsize=8)
        fig.text(0.5, 0.055, "Downsampled step", ha="center", va="center", fontsize=8)
        fig.text(bottom_box.x0, 0.055, f"row {representative_row_idx}", ha="left", va="center", fontsize=8)

        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                output_dir / f"{PAPER_WAVEFORM_STEM}.{suffix}",
                bbox_inches="tight",
                pad_inches=0.0,
            )
        if copy_to_paper:
            PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                output_dir / f"{PAPER_WAVEFORM_STEM}.pdf",
                PAPER_FIGURE_DIR / f"{PAPER_WAVEFORM_STEM}.pdf",
            )
        plt.close(fig)


def save_uniform_waveform_cache(path: Path, plot_items: list[dict[str, object]], representative_row_idx: int) -> None:
    arrays = {}
    metadata_items = []
    for idx, item in enumerate(plot_items):
        prefix = f"panel_{idx:02d}"
        arrays[f"{prefix}_steps"] = np.asarray(item["steps"], dtype=np.int64)
        arrays[f"{prefix}_b_true"] = np.asarray(item["b_true"], dtype=np.float32)
        arrays[f"{prefix}_h_true"] = np.asarray(item["h_true"], dtype=np.float32)
        arrays[f"{prefix}_h_pred"] = np.asarray(item["h_pred"], dtype=np.float32)
        metadata_items.append({"prefix": prefix, "factor": int(item["factor"]), "row_err": float(item["row_err"])})
    save_plot_cache(
        path,
        arrays=arrays,
        metadata={"stem": PAPER_WAVEFORM_STEM, "representative_row_idx": int(representative_row_idx), "items": metadata_items},
    )


def load_uniform_waveform_cache(path: Path) -> tuple[list[dict[str, object]], int]:
    arrays, metadata = load_plot_cache(path)
    plot_items = []
    for item in metadata["items"]:
        prefix = item["prefix"]
        plot_items.append(
            {
                "factor": int(item["factor"]),
                "row_err": float(item["row_err"]),
                "steps": arrays[f"{prefix}_steps"],
                "b_true": arrays[f"{prefix}_b_true"],
                "h_true": arrays[f"{prefix}_h_true"],
                "h_pred": arrays[f"{prefix}_h_pred"],
            }
        )
    return plot_items, int(metadata["representative_row_idx"])


def main():
    args = parse_args()
    if args.output_dir is None:
        output_dir = setup_output_dir(None)
    else:
        output_dir = setup_output_dir(args.output_dir)

    if args.plot_only_cache is not None:
        cache_path = args.plot_only_cache if args.plot_only_cache.is_absolute() else ROOT / args.plot_only_cache
        plot_items, representative_row_idx = load_uniform_waveform_cache(cache_path)
        norm_stats = load_json(DEFAULT_DATA_DIR / "norm_stats.json")
        render_selected_waveform_grid(
            output_dir,
            plot_items,
            representative_row_idx,
            normalized_bh_limits_enabled=args.normalized_bh_limits,
            std_h=float(norm_stats["std_h"]),
            std_b=float(norm_stats["std_b"]),
            copy_to_paper=not args.no_copy_to_paper,
        )
        print(f"Redrew {PAPER_WAVEFORM_STEM} from {cache_path}")
        return

    device = resolve_device(args.device)

    model, run_config, model_path = load_model(args.run_dir, device)
    data, lengths, deltas, norm_stats = load_origin_test(args.data_dir)

    max_len = int(lengths.max())
    selected_row_indices = np.flatnonzero(lengths == max_len).astype(int).tolist()
    if len(selected_row_indices) != 23 or max_len != 32016:
        raise RuntimeError(f"Expected 23 longest rows of length 32016, got {len(selected_row_indices)} rows length {max_len}.")

    std_h = float(norm_stats["std_h"])
    base_warmup = int(run_config["train"]["eval_warmup_len"])

    factors = list(args.factors) if args.factors is not None else list(range(args.start_factor, args.end_factor + 1))
    factors = sorted(dict.fromkeys(factors))
    if factors != list(range(1, max(PAPER_WAVEFORM_FACTORS) + 1)):
        print(f"Using custom factors: {factors}")
    representative_selection_factor = 1 if 1 in factors else factors[0]

    summary_rows = []
    per_factor_predictions = {}
    representative_row_idx = None
    representative_position = None

    for factor in factors:
        batch_data, batch_lengths, batch_deltas = make_downsampled_batch(
            data, lengths, deltas, selected_row_indices, factor
        )
        warmup_steps = int(math.ceil(base_warmup / factor))
        row_err, pred_norm, h_true_norm = evaluate_batch(
            model,
            batch_data,
            batch_lengths,
            batch_deltas,
            base_warmup,
            factor,
            device,
        )
        valid = row_err[~np.isnan(row_err)]
        mean_err = float(np.mean(valid)) if valid.size else float("nan")
        p95_err = float(np.percentile(valid, 95)) if valid.size else float("nan")

        if factor == representative_selection_factor:
            representative_position = int(np.nanargmax(row_err))
            representative_row_idx = int(selected_row_indices[representative_position])

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

        per_factor_predictions[factor] = {
            "batch_data": batch_data,
            "batch_lengths": batch_lengths,
            "warmup_steps": warmup_steps,
            "row_err": row_err,
            "pred_norm": pred_norm,
            "h_true_norm": h_true_norm,
        }
        print(
            f"factor={factor:02d} length={int(batch_lengths.max())} warmup={warmup_steps} "
            f"mean={mean_err:.5f} p95={p95_err:.5f}"
        )

    save_summary_csv(output_dir / "downsample_error_summary.csv", summary_rows, selected_row_indices)
    save_error_curve(output_dir, summary_rows)

    if representative_position is None or representative_row_idx is None:
        raise RuntimeError("Representative row was not selected.")

    if not args.no_waveforms:
        for factor in factors:
            item = per_factor_predictions[factor]
            length = int(item["batch_lengths"][representative_position])
            h_true = item["h_true_norm"][representative_position, :length] * std_h
            h_pred = item["pred_norm"][representative_position, :length] * std_h
            warmup_steps = int(item["warmup_steps"])
            row_err = relative_rms(
                item["pred_norm"][representative_position],
                item["h_true_norm"][representative_position],
                warmup_steps,
                length,
            )
            save_waveform_plot(
                output_dir,
                factor,
                h_true,
                h_pred,
                warmup_steps,
                representative_row_idx,
                row_err,
            )

        save_selected_waveform_grid(
            output_dir,
            per_factor_predictions,
            representative_position,
            representative_row_idx,
            float(norm_stats["std_b"]),
            std_h,
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
        "base_eval_warmup_len": base_warmup,
        "warmup_policy": (
            "use ground-truth H while the current input point's original-grid "
            "coordinate is smaller than base_eval_warmup_len"
        ),
        "error_start_policy": (
            "include outputs starting with the output produced by the first "
            "autoregressive H input"
        ),
        "downsample_policy": "regular stride points 0,k,2k,...; trailing remainder is not forced",
        "db_dt_policy": "recomputed after downsampling using dB_norm / (origin_dt * factor)",
        "error_definition": "per-row relative RMS on post-warmup points; mean and P95 over selected rows",
        "selected_length": max_len,
        "selected_row_count": len(selected_row_indices),
        "selected_row_indices": selected_row_indices,
        "representative_row_policy": (
            f"largest factor={representative_selection_factor} row error among selected longest rows"
        ),
        "representative_selection_factor": representative_selection_factor,
        "representative_row_idx": representative_row_idx,
        "waveforms_saved": not args.no_waveforms,
        "device": str(device),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
