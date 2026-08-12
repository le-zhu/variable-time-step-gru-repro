from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from figures.paper_figure_sizes import figure_size, line_width, marker_size
from figures.paper_plot_style import paper_matplotlib_rc

from scripts.evaluate_n87_longest_downsample import (
    ROOT,
    load_model,
    load_origin_test,
    make_downsampled_batch,
    resolve_device,
)
from scripts.evaluate_n87_longest_upsample import make_upsampled_batch


PAPER_FIGURE_DIR = ROOT / "figures" / "output"
DEFAULT_DTSCALED_RUN_DIR = ROOT / "checkpoints" / "n87_paper_models" / "dtscaledgru"
DEFAULT_DTSCALED_DATA_DIR = ROOT / "data" / "N87_processed"

DTSCALED_LONGEST_DOWNSAMPLE_METADATA = ROOT / "results" / "n87_longest_downsample_dtscaledgru" / "metadata.json"
DTSCALED_SHORTEST_UPSAMPLE_METADATA = ROOT / "results" / "n87_shortest_upsample_dtscaledgru" / "metadata.json"
EXPCAP_LONGEST_DOWNSAMPLE_METADATA = ROOT / "results" / "n87_longest_downsample_rategru" / "metadata.json"
EXPCAP_SHORTEST_UPSAMPLE_METADATA = ROOT / "results" / "n87_shortest_upsample_rategru" / "metadata.json"

DEFAULT_FACTORS = (1, 2, 5, 10, 20, 30, 40, 50)
DEFAULT_HIST_BINS = 5000
S_HIST_MIN = 1.0e-8
S_HIST_MAX = 1.0e4
EXPCAP_LINEAR_BOUNDARY = 0.8


@dataclass(frozen=True)
class CaseSpec:
    metadata_path: Path
    sampling: str
    sequence_group: str
    model_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DtScaledGRU and RateGRU update-coefficient statistics under "
            "longest-row downsampling and shortest-row upsampling."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("dtscaled", "expcap"),
        default=["dtscaled", "expcap"],
        help="Model families to evaluate. Use '--models expcap' to skip DtScaledGRU statistics.",
    )
    parser.add_argument("--factors", type=int, nargs="+", default=list(DEFAULT_FACTORS))
    parser.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--plot-only-summary",
        type=Path,
        default=None,
        help="Reuse an existing sampling_update_coefficient_summary.csv and only redraw figures.",
    )
    parser.add_argument(
        "--stat-flush-steps",
        type=int,
        default=128,
        help="Number of recurrent steps buffered before updating histograms.",
    )
    parser.add_argument(
        "--exact-effective-median",
        action="store_true",
        help=(
            "Store the bounded effective update coefficients for each case/factor in a temporary "
            "float32 file and compute their median with numpy.partition. Means and ratios are already exact."
        ),
    )
    parser.add_argument(
        "--median-work-dir",
        type=Path,
        default=ROOT / "results" / "tmp_exact_update_coeff_median",
        help="Temporary directory used by --exact-effective-median.",
    )
    parser.add_argument("--dtscaled-longest-downsample-metadata", type=Path, default=None)
    parser.add_argument("--dtscaled-shortest-upsample-metadata", type=Path, default=None)
    parser.add_argument("--expcap-longest-downsample-metadata", type=Path, default=None)
    parser.add_argument("--expcap-shortest-upsample-metadata", type=Path, default=None)
    parser.add_argument("--dtscaledgru-run-dir", type=Path, default=DEFAULT_DTSCALED_RUN_DIR)
    parser.add_argument("--dtscaledgru-data-dir", type=Path, default=DEFAULT_DTSCALED_DATA_DIR)
    parser.add_argument(
        "--sampling",
        choices=("both", "downsample", "upsample"),
        default="both",
        help="Limit evaluation to one sampling direction so the two directions can run in parallel.",
    )
    parser.add_argument("--no-paper-copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    expcap_longest_downsample_metadata = (
        args.expcap_longest_downsample_metadata
        if args.expcap_longest_downsample_metadata is not None
        else EXPCAP_LONGEST_DOWNSAMPLE_METADATA
    )
    expcap_shortest_upsample_metadata = (
        args.expcap_shortest_upsample_metadata
        if args.expcap_shortest_upsample_metadata is not None
        else EXPCAP_SHORTEST_UPSAMPLE_METADATA
    )

    selected = set(args.models)
    cases: list[CaseSpec] = []
    if "dtscaled" in selected:
        cases.extend(
            [
                CaseSpec(args.dtscaled_longest_downsample_metadata
                         or DTSCALED_LONGEST_DOWNSAMPLE_METADATA,
                         "downsample", "longest", "DtScaledGRU"),
                CaseSpec(args.dtscaled_shortest_upsample_metadata
                         or DTSCALED_SHORTEST_UPSAMPLE_METADATA,
                         "upsample", "shortest", "DtScaledGRU"),
            ]
        )
    if "expcap" in selected:
        cases.extend(
            [
                CaseSpec(expcap_longest_downsample_metadata, "downsample", "longest", "RateGRU"),
                CaseSpec(expcap_shortest_upsample_metadata, "upsample", "shortest", "RateGRU"),
            ]
        )
    return cases


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_output_dir(path: Path | None, overwrite: bool) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ROOT / "results" / f"local_n87_update_coeff_sampling_diagnostics_{stamp}"
    path = path if path.is_absolute() else ROOT / path
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def reciprocal(value: float) -> float:
    if value > 0.0 and np.isfinite(value):
        return 1.0 / value
    return float("nan")


def hist_median(hist: torch.Tensor, value_min: float, value_max: float) -> float:
    hist_cpu = hist.detach().cpu().numpy().astype(np.int64, copy=False)
    total = int(hist_cpu.sum())
    if total <= 0:
        return float("nan")
    target = 0.5 * total
    idx = int(np.searchsorted(np.cumsum(hist_cpu), target, side="left"))
    idx = min(max(idx, 0), len(hist_cpu) - 1)
    return float(value_min + (idx + 0.5) / len(hist_cpu) * (value_max - value_min))


def log_hist_median(hist: torch.Tensor, value_min: float, value_max: float) -> float:
    hist_cpu = hist.detach().cpu().numpy().astype(np.int64, copy=False)
    total = int(hist_cpu.sum())
    if total <= 0:
        return float("nan")
    target = 0.5 * total
    idx = int(np.searchsorted(np.cumsum(hist_cpu), target, side="left"))
    idx = min(max(idx, 0), len(hist_cpu) - 1)
    edges = np.geomspace(value_min, value_max, len(hist_cpu) + 1)
    return float(math.sqrt(edges[idx] * edges[idx + 1]))


class BufferedMetric:
    def __init__(
        self,
        *,
        hist_bins: int,
        device: torch.device,
        value_min: float,
        value_max: float,
        log_scale: bool = False,
    ) -> None:
        self.hist_bins = int(hist_bins)
        self.device = device
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.log_scale = bool(log_scale)
        self.hist = torch.zeros(self.hist_bins, dtype=torch.long, device=device)
        self.total = torch.zeros((), dtype=torch.float64, device=device)
        self.count = 0
        self.buffers: list[torch.Tensor] = []
        if self.log_scale:
            self.log_min = math.log(self.value_min)
            self.log_max = math.log(self.value_max)

    def add(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self.buffers.append(values.detach().reshape(-1).float())

    def flush(self) -> None:
        if not self.buffers:
            return
        values = torch.cat(self.buffers)
        self.buffers.clear()
        self.count += int(values.numel())
        self.total += values.double().sum()
        if self.log_scale:
            safe = torch.nan_to_num(values, nan=self.value_min, posinf=self.value_max, neginf=self.value_min)
            safe = torch.clamp(safe, self.value_min, torch.nextafter(torch.tensor(self.value_max, device=self.device), torch.tensor(0.0, device=self.device)))
            bins = torch.floor((torch.log(safe) - self.log_min) / (self.log_max - self.log_min) * self.hist_bins)
        else:
            safe = torch.nan_to_num(values, nan=self.value_min, posinf=self.value_max, neginf=self.value_min)
            safe = torch.clamp(safe, self.value_min, torch.nextafter(torch.tensor(self.value_max, device=self.device), torch.tensor(self.value_min, device=self.device)))
            bins = torch.floor((safe - self.value_min) / (self.value_max - self.value_min) * self.hist_bins)
        bins = torch.clamp(bins.long(), 0, self.hist_bins - 1)
        self.hist += torch.bincount(bins, minlength=self.hist_bins)

    def mean(self) -> float:
        self.flush()
        if self.count <= 0:
            return float("nan")
        return float((self.total / self.count).detach().cpu())

    def median(self) -> float:
        self.flush()
        if self.log_scale:
            return log_hist_median(self.hist, self.value_min, self.value_max)
        return hist_median(self.hist, self.value_min, self.value_max)


class ExactMedianBufferedMetric(BufferedMetric):
    def __init__(
        self,
        *,
        hist_bins: int,
        device: torch.device,
        value_min: float,
        value_max: float,
        temp_dir: Path,
        log_scale: bool = False,
    ) -> None:
        super().__init__(
            hist_bins=hist_bins,
            device=device,
            value_min=value_min,
            value_max=value_max,
            log_scale=log_scale,
        )
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="exact_effective_", suffix=".f32", dir=temp_dir)
        self.temp_path = Path(name)
        self._handle = os.fdopen(fd, "wb")
        self._median_value: float | None = None

    def flush(self) -> None:
        if not self.buffers:
            return
        values = torch.cat(self.buffers)
        self.buffers.clear()
        self.count += int(values.numel())
        self.total += values.double().sum()

        values_cpu = values.detach().cpu().numpy().astype(np.float32, copy=False)
        values_cpu.tofile(self._handle)

        if self.log_scale:
            safe = torch.nan_to_num(values, nan=self.value_min, posinf=self.value_max, neginf=self.value_min)
            safe = torch.clamp(
                safe,
                self.value_min,
                torch.nextafter(torch.tensor(self.value_max, device=self.device), torch.tensor(0.0, device=self.device)),
            )
            bins = torch.floor((torch.log(safe) - self.log_min) / (self.log_max - self.log_min) * self.hist_bins)
        else:
            safe = torch.nan_to_num(values, nan=self.value_min, posinf=self.value_max, neginf=self.value_min)
            safe = torch.clamp(
                safe,
                self.value_min,
                torch.nextafter(torch.tensor(self.value_max, device=self.device), torch.tensor(self.value_min, device=self.device)),
            )
            bins = torch.floor((safe - self.value_min) / (self.value_max - self.value_min) * self.hist_bins)
        bins = torch.clamp(bins.long(), 0, self.hist_bins - 1)
        self.hist += torch.bincount(bins, minlength=self.hist_bins)

    def median(self) -> float:
        self.flush()
        if self._median_value is not None:
            return self._median_value
        if self.count <= 0:
            return float("nan")
        self._handle.flush()
        self._handle.close()
        values = np.memmap(self.temp_path, dtype=np.float32, mode="r+", shape=(self.count,))
        kth_low = (self.count - 1) // 2
        kth_high = self.count // 2
        if kth_low == kth_high:
            values.partition(kth_low)
            median = float(values[kth_low])
        else:
            values.partition((kth_low, kth_high))
            median = 0.5 * (float(values[kth_low]) + float(values[kth_high]))
        values.flush()
        del values
        try:
            self.temp_path.unlink()
        except FileNotFoundError:
            pass
        self._median_value = median
        return median


class UpdateStats:
    def __init__(
        self,
        *,
        raw_metric: BufferedMetric,
        effective_metric: BufferedMetric,
        prelimit_metric: BufferedMetric | None = None,
        flush_steps: int = 128,
    ) -> None:
        self.raw_metric = raw_metric
        self.effective_metric = effective_metric
        self.prelimit_metric = prelimit_metric
        self.flush_steps = int(flush_steps)
        self.pending_steps = 0
        self.effective_gt_1_count = torch.zeros((), dtype=torch.long, device=raw_metric.device)
        self.effective_count = 0
        self.sdt_gt_0p8_count = torch.zeros((), dtype=torch.long, device=raw_metric.device)
        self.sdt_count = 0

    def add(self, raw: torch.Tensor, effective: torch.Tensor, prelimit: torch.Tensor | None = None) -> None:
        self.raw_metric.add(raw)
        self.effective_metric.add(effective)
        flat_effective = effective.detach().reshape(-1)
        self.effective_gt_1_count += torch.count_nonzero(flat_effective > 1.0)
        self.effective_count += int(flat_effective.numel())
        if prelimit is not None:
            if self.prelimit_metric is not None:
                self.prelimit_metric.add(prelimit)
            flat = prelimit.detach().reshape(-1)
            self.sdt_gt_0p8_count += torch.count_nonzero(flat > EXPCAP_LINEAR_BOUNDARY)
            self.sdt_count += int(flat.numel())
        self.pending_steps += 1
        if self.pending_steps >= self.flush_steps:
            self.flush()

    def flush(self) -> None:
        self.raw_metric.flush()
        self.effective_metric.flush()
        if self.prelimit_metric is not None:
            self.prelimit_metric.flush()
        self.pending_steps = 0

    def row(self) -> dict[str, float | int]:
        self.flush()
        raw_mean = self.raw_metric.mean()
        raw_median = self.raw_metric.median()
        effective_mean = self.effective_metric.mean()
        effective_median = self.effective_metric.median()
        prelimit_mean = self.prelimit_metric.mean() if self.prelimit_metric is not None else float("nan")
        prelimit_median = self.prelimit_metric.median() if self.prelimit_metric is not None else float("nan")
        return {
            "update_value_count": int(self.raw_metric.count),
            "raw_mean": raw_mean,
            "raw_median": raw_median,
            "effective_mean": effective_mean,
            "effective_median": effective_median,
            "prelimit_mean": prelimit_mean,
            "prelimit_median": prelimit_median,
            "reciprocal_raw_mean": reciprocal(raw_mean),
            "reciprocal_raw_median": reciprocal(raw_median),
            "reciprocal_effective_mean": reciprocal(effective_mean),
            "reciprocal_effective_median": reciprocal(effective_median),
            "effective_gt_1_ratio": (
                float((self.effective_gt_1_count.double() / self.effective_count).detach().cpu())
                if self.effective_count > 0
                else float("nan")
            ),
            "sdt_gt_0p8_ratio": (
                float((self.sdt_gt_0p8_count.double() / self.sdt_count).detach().cpu())
                if self.sdt_count > 0
                else float("nan")
            ),
        }


def maybe_exact_metric(
    *,
    exact_median: bool,
    hist_bins: int,
    device: torch.device,
    value_min: float,
    value_max: float,
    temp_dir: Path | None,
    log_scale: bool = False,
) -> BufferedMetric:
    if exact_median:
        if temp_dir is None:
            raise ValueError("temp_dir is required for exact median metrics")
        return ExactMedianBufferedMetric(
            hist_bins=hist_bins,
            device=device,
            value_min=value_min,
            value_max=value_max,
            temp_dir=temp_dir,
            log_scale=log_scale,
        )
    return BufferedMetric(
        hist_bins=hist_bins,
        device=device,
        value_min=value_min,
        value_max=value_max,
        log_scale=log_scale,
    )


def make_dtscaled_stats(
    hist_bins: int,
    device: torch.device,
    alpha_max: float,
    flush_steps: int,
    *,
    exact_effective_median: bool = False,
    median_work_dir: Path | None = None,
) -> UpdateStats:
    return UpdateStats(
        raw_metric=BufferedMetric(hist_bins=hist_bins, device=device, value_min=0.0, value_max=1.0),
        effective_metric=maybe_exact_metric(
            exact_median=exact_effective_median,
            hist_bins=hist_bins,
            device=device,
            value_min=0.0,
            value_max=max(float(alpha_max), 1.0e-12),
            temp_dir=median_work_dir,
        ),
        flush_steps=flush_steps,
    )


def make_expcap_stats(
    hist_bins: int,
    device: torch.device,
    sdt_max: float,
    flush_steps: int,
    *,
    exact_effective_median: bool = False,
    median_work_dir: Path | None = None,
) -> UpdateStats:
    return UpdateStats(
        raw_metric=BufferedMetric(
            hist_bins=hist_bins,
            device=device,
            value_min=S_HIST_MIN,
            value_max=S_HIST_MAX,
            log_scale=True,
        ),
        effective_metric=maybe_exact_metric(
            exact_median=exact_effective_median,
            hist_bins=hist_bins,
            device=device,
            value_min=0.0,
            value_max=1.0,
            temp_dir=median_work_dir,
        ),
        prelimit_metric=BufferedMetric(
            hist_bins=hist_bins,
            device=device,
            value_min=0.0,
            value_max=max(float(sdt_max), 1.0e-12),
        ),
        flush_steps=flush_steps,
    )


def dtscaled_step_with_stats(model, input_t: torch.Tensor, h_input: torch.Tensor, hs: torch.Tensor, dt: torch.Tensor):
    gru_input = torch.cat([input_t, h_input], dim=1)
    u_t = torch.sigmoid(model.gru_wz(gru_input) + model.gru_uz(hs))
    r_t = torch.sigmoid(model.gru_wr(gru_input) + model.gru_ur(hs))
    candidate = torch.tanh(model.gru_wh(gru_input) + model.gru_uh(r_t * hs))
    alpha_t = dt * u_t
    hs_next = hs + alpha_t * (candidate - hs)
    h_next = model.head(hs_next)
    return h_next, hs_next, u_t, alpha_t


def expcap(alpha_prelimit: torch.Tensor) -> torch.Tensor:
    return torch.where(
        alpha_prelimit <= EXPCAP_LINEAR_BOUNDARY,
        alpha_prelimit,
        1.0 - 0.2 * torch.exp(-(alpha_prelimit - EXPCAP_LINEAR_BOUNDARY) / 0.2),
    )


def expcap_step_with_stats(model, input_t: torch.Tensor, h_input: torch.Tensor, hs: torch.Tensor, dt: torch.Tensor):
    gru_input = torch.cat([input_t, h_input], dim=1)
    s_t = torch.nn.functional.softplus(model.gru_wz(gru_input) + model.gru_uz(hs))
    r_t = torch.sigmoid(model.gru_wr(gru_input) + model.gru_ur(hs))
    candidate = torch.tanh(model.gru_wh(gru_input) + model.gru_uh(r_t * hs))
    sdt = dt * s_t
    alpha_t = expcap(sdt)
    hs_next = hs + alpha_t * (candidate - hs)
    h_next = model.head(hs_next)
    return h_next, hs_next, s_t, alpha_t, sdt


def run_sampling_factor(
    *,
    model,
    model_key: str,
    batch_data: np.ndarray,
    batch_lengths: np.ndarray,
    batch_deltas: np.ndarray,
    warmup_original_steps: int,
    original_coordinate_scale: float,
    hist_bins: int,
    device: torch.device,
    flush_steps: int,
    exact_effective_median: bool = False,
    median_work_dir: Path | None = None,
) -> dict[str, float | int]:
    input_seq = torch.from_numpy(batch_data[:, :, :-1]).float().to(device)
    h_gt = torch.from_numpy(batch_data[:, :, -1:]).float().to(device)
    dt = torch.from_numpy(batch_deltas).float().to(device).view(-1, 1)
    lengths_tensor = torch.from_numpy(batch_lengths).to(device=device, dtype=torch.long)

    batch_n, seq_len, _ = input_seq.shape
    h_curr = h_gt[:, 0, :]
    hs = torch.zeros(batch_n, model.hidden_size, device=device)

    alpha_max = float(np.max(batch_deltas)) if model_key == "DtScaledGRU" else 1.0
    sdt_max = max(float(np.max(batch_deltas)) * S_HIST_MAX, 1.0)
    stats = (
        make_dtscaled_stats(
            hist_bins,
            device,
            alpha_max,
            flush_steps,
            exact_effective_median=exact_effective_median,
            median_work_dir=median_work_dir,
        )
        if model_key == "DtScaledGRU"
        else make_expcap_stats(
            hist_bins,
            device,
            sdt_max,
            flush_steps,
            exact_effective_median=exact_effective_median,
            median_work_dir=median_work_dir,
        )
    )

    with torch.inference_mode():
        if int(batch_lengths.min()) == int(batch_lengths.max()):
            for t in range(max(seq_len - 1, 0)):
                use_teacher = t * original_coordinate_scale < warmup_original_steps
                h_input = h_gt[:, t, :] if use_teacher else h_curr
                if model_key == "DtScaledGRU":
                    h_next, hs_next, raw, effective = dtscaled_step_with_stats(
                        model, input_seq[:, t, :], h_input, hs, dt
                    )
                    prelimit = None
                elif model_key == "SoftplusRateExpCapGRU":
                    h_next, hs_next, raw, effective, prelimit = expcap_step_with_stats(
                        model, input_seq[:, t, :], h_input, hs, dt
                    )
                else:
                    raise ValueError(f"Unsupported model_key: {model_key}")

                if not use_teacher:
                    stats.add(raw, effective, prelimit)

                hs = hs_next
                h_curr = h_next
        else:
            for t in range(max(seq_len - 1, 0)):
                valid_step = t < (lengths_tensor - 1)
                if not bool(valid_step.any()):
                    break

                use_teacher = t * original_coordinate_scale < warmup_original_steps
                h_input = torch.where(
                    valid_step.unsqueeze(1) & use_teacher,
                    h_gt[:, t, :],
                    h_curr,
                )
                if model_key == "DtScaledGRU":
                    h_next, hs_next, raw, effective = dtscaled_step_with_stats(
                        model, input_seq[:, t, :], h_input, hs, dt
                    )
                    prelimit = None
                elif model_key == "SoftplusRateExpCapGRU":
                    h_next, hs_next, raw, effective, prelimit = expcap_step_with_stats(
                        model, input_seq[:, t, :], h_input, hs, dt
                    )
                else:
                    raise ValueError(f"Unsupported model_key: {model_key}")

                if not use_teacher:
                    stats.add(
                        raw[valid_step],
                        effective[valid_step],
                        prelimit[valid_step] if prelimit is not None else None,
                    )

                valid_expanded = valid_step.unsqueeze(1)
                hs = torch.where(valid_expanded, hs_next, hs)
                h_curr = torch.where(valid_expanded, h_next, h_curr)

    del input_seq, h_gt, dt, lengths_tensor, hs, h_curr
    out = stats.row()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def selected_indices_from_metadata(metadata: dict[str, object]) -> list[int]:
    indices = metadata.get("selected_row_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("metadata does not contain selected_row_indices")
    return [int(idx) for idx in indices]


def build_case_rows(
    *,
    metadata_path: Path,
    sampling: str,
    sequence_group: str,
    factors: Iterable[int],
    hist_bins: int,
    device: torch.device,
    flush_steps: int,
    exact_effective_median: bool = False,
    median_work_dir: Path | None = None,
    run_dir_override: Path | None = None,
    data_dir_override: Path | None = None,
) -> list[dict[str, object]]:
    metadata = load_json(metadata_path)
    run_dir = run_dir_override.resolve() if run_dir_override is not None else Path(str(metadata["run_dir"]))
    data_dir = data_dir_override.resolve() if data_dir_override is not None else Path(str(metadata["data_dir"]))
    selected_indices = selected_indices_from_metadata(metadata)
    model, run_config, model_path = load_model(run_dir, device)
    data, lengths, deltas, _norm_stats = load_origin_test(data_dir)
    base_warmup = int(run_config["train"]["eval_warmup_len"])
    model_key = str(run_config["model_key"])
    model_label = "RateGRU" if model_key == "SoftplusRateExpCapGRU" else model_key

    rows: list[dict[str, object]] = []
    for factor in factors:
        if sampling == "downsample":
            batch_data, batch_lengths, batch_deltas = make_downsampled_batch(
                data, lengths, deltas, selected_indices, int(factor)
            )
            warmup_steps = int(math.ceil(base_warmup / int(factor)))
            original_coordinate_scale = float(factor)
        elif sampling == "upsample":
            batch_data, batch_lengths, batch_deltas = make_upsampled_batch(
                data, lengths, deltas, selected_indices, int(factor)
            )
            warmup_steps = int(math.ceil(base_warmup * int(factor)))
            original_coordinate_scale = 1.0 / float(factor)
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling}")

        print(
            f"[update-stats] {model_label} {sequence_group} {sampling} factor={factor} "
            f"length={int(batch_lengths.max())} warmup={warmup_steps}",
            flush=True,
        )
        stat_row = run_sampling_factor(
            model=model,
            model_key=model_key,
            batch_data=batch_data,
            batch_lengths=batch_lengths,
            batch_deltas=batch_deltas,
            warmup_original_steps=base_warmup,
            original_coordinate_scale=original_coordinate_scale,
            hist_bins=hist_bins,
            device=device,
            flush_steps=flush_steps,
            exact_effective_median=exact_effective_median,
            median_work_dir=median_work_dir,
        )
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "sampling": sampling,
                "sequence_group": sequence_group,
                "factor": int(factor),
                "effective_length_min": int(batch_lengths.min()),
                "effective_length_max": int(batch_lengths.max()),
                "warmup_steps": int(warmup_steps),
                "dt_eff_min": float(batch_deltas.min()),
                "dt_eff_max": float(batch_deltas.max()),
                "run_dir": str(run_dir),
                "model_path": str(model_path),
                "data_dir": str(data_dir),
                "source_metadata": str(metadata_path),
                **stat_row,
            }
        )
        del batch_data, batch_lengths, batch_deltas
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_key",
        "model_label",
        "sampling",
        "sequence_group",
        "factor",
        "effective_length_min",
        "effective_length_max",
        "warmup_steps",
        "dt_eff_min",
        "dt_eff_max",
        "update_value_count",
        "raw_mean",
        "raw_median",
        "effective_mean",
        "effective_median",
        "prelimit_mean",
        "prelimit_median",
        "effective_gt_1_ratio",
        "sdt_gt_0p8_ratio",
        "reciprocal_raw_mean",
        "reciprocal_raw_median",
        "reciprocal_effective_mean",
        "reciprocal_effective_median",
        "run_dir",
        "model_path",
        "data_dir",
        "source_metadata",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fieldnames:
                value = row.get(field, "")
                out[field] = format_float(value) if isinstance(value, float) else value
            writer.writerow(out)


def read_rows(path: Path) -> list[dict[str, object]]:
    int_fields = {
        "factor",
        "effective_length_min",
        "effective_length_max",
        "warmup_steps",
        "update_value_count",
    }
    float_fields = {
        "dt_eff_min",
        "dt_eff_max",
        "raw_mean",
        "raw_median",
        "effective_mean",
        "effective_median",
        "prelimit_mean",
        "prelimit_median",
        "effective_gt_1_ratio",
        "sdt_gt_0p8_ratio",
        "reciprocal_raw_mean",
        "reciprocal_raw_median",
        "reciprocal_effective_mean",
        "reciprocal_effective_median",
    }
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = dict(row)
            for field in int_fields:
                converted[field] = int(row[field])
            for field in float_fields:
                converted[field] = float(row[field]) if field in row and row[field] != "" else float("nan")
            rows.append(converted)
    return rows


def display_model_label(model_label: str) -> str:
    if model_label == "ExpCapRateGRU":
        return "RateGRU"
    return model_label


def rategru_effective_labels(*, reciprocal_mode: bool) -> tuple[str, str]:
    if reciprocal_mode:
        return (
            r"$1/\mathrm{mean}(\mathrm{expcap}(dt\,s_t))$",
            r"$1/\mathrm{median}(\mathrm{expcap}(dt\,s_t))$",
        )
    return (r"Mean $\mathrm{expcap}(dt\,s_t)$", r"Median $\mathrm{expcap}(dt\,s_t)$")


def combined_effective_labels(model_label: str, *, reciprocal_mode: bool) -> tuple[str, str, str]:
    if model_label == "DtScaledGRU":
        if reciprocal_mode:
            return (
                r"$1/[\mathrm{Mean}(dt\,u_t)]$",
                r"$1/[\mathrm{Median}(dt\,u_t)]$",
                "Value",
            )
        return (r"Mean $dt\,u_t$", r"Median $dt\,u_t$", "Value")

    if reciprocal_mode:
        return (
            r"$1/[\mathrm{Mean}(\mathrm{expcap}(dt\,s_t))]$",
            r"$1/[\mathrm{Median}(\mathrm{expcap}(dt\,s_t))]$",
            "Value",
        )
    return (
        r"Mean $\mathrm{expcap}(dt\,s_t)$",
        r"Median $\mathrm{expcap}(dt\,s_t)$",
        "Value",
    )


def filtered_rows(rows: list[dict[str, object]], model_label: str, sampling: str) -> list[dict[str, object]]:
    return sorted(
        [
            row
            for row in rows
            if display_model_label(str(row["model_label"])) == model_label and row["sampling"] == sampling
        ],
        key=lambda item: int(item["factor"]),
    )


def has_rows(rows: list[dict[str, object]], model_label: str, sampling: str) -> bool:
    return any(display_model_label(str(row["model_label"])) == model_label and row["sampling"] == sampling for row in rows)


def save_formats(fig, output_dir: Path, stem: str, copy_to_paper: bool) -> None:
    save_kwargs = {}
    if stem == "update_coeff_sampling_diagnostics_combined":
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.0}
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", **save_kwargs)
    if copy_to_paper and stem == "update_coeff_sampling_diagnostics_combined":
        PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIGURE_DIR / f"{stem}.pdf", **save_kwargs)


def configure_axes(ax, stem: str | None = None) -> None:
    major_width = line_width(stem, 0.8) if stem is not None else 0.8
    minor_width = line_width(stem, 0.55) if stem is not None else 0.55
    tick_length = marker_size(stem, 3.5) if stem is not None else 3.5
    ax.grid(True, which="major", color="#d9d9d9", linewidth=major_width)
    ax.grid(True, which="minor", axis="x", color="#e7e7e7", linewidth=minor_width)
    ax.tick_params(axis="both", labelsize=8, pad=1, width=major_width, length=tick_length)
    for spine in ax.spines.values():
        spine.set_linewidth(major_width)


def set_factor_axis(ax, factors: list[int], xlabel: str) -> None:
    ax.set_xlim(0.0, max(factors) + 3.0)
    if max(factors) >= 50:
        tick_factors = [factor for factor in factors if factor in {1, 10, 20, 30, 40, 50}]
    else:
        tick_factors = factors
    ax.set_xticks(tick_factors)
    labels = ax.set_xticklabels([str(factor) for factor in tick_factors], fontsize=8)
    for label in labels:
        label.set_ha("center")
    ax.set_xlabel(xlabel, fontsize=8, labelpad=1)
    ax.minorticks_off()


def align_zero_ticks(ax, ax2, right_top: float = 1.02, zero_fraction: float = 0.035) -> None:
    left_top = ax.get_ylim()[1]
    left_bottom = -zero_fraction * left_top / (1.0 - zero_fraction)
    right_bottom = -zero_fraction * right_top / (1.0 - zero_fraction)
    ax.set_ylim(left_bottom, left_top)
    ax2.set_ylim(right_bottom, right_top)


def use_unit_coefficient_ticks(ax) -> None:
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], 1.02))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))


def plot_dtscaled(rows: list[dict[str, object]], sampling: str, output_dir: Path, copy_to_paper: bool) -> None:
    data = filtered_rows(rows, "DtScaledGRU", sampling)
    if not data:
        return
    factors = [int(row["factor"]) for row in data]
    reciprocal_mode = sampling == "upsample"
    eff_mean_key = "reciprocal_effective_mean" if reciprocal_mode else "effective_mean"
    eff_median_key = "reciprocal_effective_median" if reciprocal_mode else "effective_median"
    if reciprocal_mode:
        eff_mean_label = r"$1/\mathrm{mean}(dt\,u_t)$"
        eff_median_label = r"$1/\mathrm{median}(dt\,u_t)$"
    else:
        eff_mean_label = r"Mean $dt\,u_t$"
        eff_median_label = r"Median $dt\,u_t$"

    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
        ax.plot(
            factors,
            [row[eff_mean_key] for row in data],
            marker="^",
            linewidth=1.9,
            label=eff_mean_label,
        )
        ax.plot(
            factors,
            [row[eff_median_key] for row in data],
            marker="D",
            linewidth=1.7,
            linestyle="--",
            label=eff_median_label,
        )
        if reciprocal_mode:
            ax.set_ylabel("Reciprocal coefficient")
            title = "DtScaledGRU shortest-row upsampling update-coefficient reciprocals"
            xlabel = "Upsampling factor"
            stem = "dtscaled_shortest_upsample_update_coeff_reciprocal"
        else:
            ax.set_ylabel("Coefficient value")
            title = "DtScaledGRU longest-row downsampling update coefficients"
            xlabel = "Downsampling factor"
            stem = "dtscaled_longest_downsample_update_coefficients"

        ax2 = ax.twinx()
        ratio = [float(row["effective_gt_1_ratio"]) for row in data]
        ax2.plot(
            factors,
            ratio,
            marker="v",
            linewidth=1.8,
            linestyle=":",
            color="#b21a35",
            label=r"$dt\,u_t>1$ ratio",
        )
        align_zero_ticks(ax, ax2)
        ax2.set_ylabel("Fraction of hidden-state updates")
        ax2.tick_params(axis="y", labelsize=9, colors="#b21a35", pad=5)

        set_factor_axis(ax, factors, xlabel)
        configure_axes(ax)
        ax.set_title(title)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        save_formats(fig, output_dir, stem, copy_to_paper)
        plt.close(fig)


def plot_expcap(
    rows: list[dict[str, object]],
    sampling: str,
    output_dir: Path,
    copy_to_paper: bool,
) -> None:
    data = filtered_rows(rows, "RateGRU", sampling)
    if not data:
        return
    factors = [int(row["factor"]) for row in data]
    reciprocal_mode = sampling == "upsample"
    eff_mean_key = "reciprocal_effective_mean" if reciprocal_mode else "effective_mean"
    eff_median_key = "reciprocal_effective_median" if reciprocal_mode else "effective_median"
    eff_mean_label, eff_median_label = rategru_effective_labels(reciprocal_mode=reciprocal_mode)

    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
        ax.plot(
            factors,
            [row[eff_mean_key] for row in data],
            marker="^",
            linewidth=1.9,
            label=eff_mean_label,
        )
        ax.plot(
            factors,
            [row[eff_median_key] for row in data],
            marker="D",
            linewidth=1.7,
            linestyle="--",
            label=eff_median_label,
        )
        if reciprocal_mode:
            ax.set_ylabel("Reciprocal coefficient")
            title = "RateGRU shortest-row upsampling update-coefficient reciprocals"
            xlabel = "Upsampling factor"
            stem = "expcap_shortest_upsample_update_coeff_reciprocal"
        else:
            ax.set_ylabel("Coefficient value")
            title = "RateGRU longest-row downsampling update coefficients"
            xlabel = "Downsampling factor"
            stem = "expcap_longest_downsample_update_coefficients"

        ax2 = ax.twinx()
        ratio = [float(row["sdt_gt_0p8_ratio"]) for row in data]
        ax2.plot(
            factors,
            ratio,
            marker="v",
            linewidth=1.8,
            linestyle=":",
            color="#b21a35",
            label=r"$dt\,s_t>0.8$ ratio",
        )
        align_zero_ticks(ax, ax2)
        ax2.set_ylabel("Fraction of hidden-state updates")
        ax2.tick_params(axis="y", labelsize=9, colors="#b21a35", pad=5)

        set_factor_axis(ax, factors, xlabel)
        configure_axes(ax)
        ax.set_title(title)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        save_formats(fig, output_dir, stem, copy_to_paper)
        plt.close(fig)


def plot_combined_update_coefficients(
    rows: list[dict[str, object]],
    output_dir: Path,
    copy_to_paper: bool,
) -> None:
    stem = "update_coeff_sampling_diagnostics_combined"
    with plt.rc_context(paper_matplotlib_rc(font_size=8)):
        model_labels = ("DtScaledGRU", "RateGRU")
        sampling_rows = (
            ("downsample", "50 kHz downsampling"),
            ("upsample", "800 kHz upsampling"),
        )
        if not any(has_rows(rows, model_label, sampling) for sampling, _ in sampling_rows for model_label in model_labels):
            return

        width, height = figure_size(stem)
        fig, axes = plt.subplots(2, 3, figsize=(width, height), dpi=300, squeeze=False)

        for row_idx, (sampling, row_label) in enumerate(sampling_rows):
            for col_idx, model_label in enumerate(model_labels):
                ax = axes[row_idx, col_idx]
                if not has_rows(rows, model_label, sampling):
                    ax.set_visible(False)
                    continue

                data = filtered_rows(rows, model_label, sampling)
                factors = [int(row["factor"]) for row in data]
                reciprocal_mode = sampling == "upsample"
                eff_mean_key = "reciprocal_effective_mean" if reciprocal_mode else "effective_mean"
                eff_median_key = "reciprocal_effective_median" if reciprocal_mode else "effective_median"
                xlabel = "Downsampling factor" if sampling == "downsample" else "Upsampling factor"
                mean_label, median_label, ylabel = combined_effective_labels(
                    model_label,
                    reciprocal_mode=reciprocal_mode,
                )

                if model_label == "DtScaledGRU":
                    ratio_key = "effective_gt_1_ratio"
                    ratio_label = r"Fraction $dt\,u_t>1$"
                else:
                    ratio_key = "sdt_gt_0p8_ratio"
                    ratio_label = r"Fraction $dt\,s_t>0.8$"

                ax.plot(
                    factors,
                    [row[eff_mean_key] for row in data],
                    marker="^",
                    linewidth=line_width(stem, 1.5),
                    markersize=marker_size(stem, 4.8),
                    color="#0072B2",
                    label=mean_label,
                )
                ax.plot(
                    factors,
                    [row[eff_median_key] for row in data],
                    marker="D",
                    linewidth=line_width(stem, 1.4),
                    markersize=marker_size(stem, 4.4),
                    linestyle="-",
                    color="#ff7f0e",
                    label=median_label,
                )
                if row_idx == 0:
                    ax.set_title(model_label, fontsize=8, pad=2)
                ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
                set_factor_axis(ax, factors, xlabel if row_idx == 1 else "")
                configure_axes(ax, stem)
                if row_idx == 1:
                    ax.tick_params(axis="y", labelsize=8, pad=5)
                    for tick in ax.get_yticklabels():
                        tick.set_rotation(90)
                        tick.set_ha("center")
                        tick.set_va("center")
                ax.set_box_aspect(1.0)

                ax2 = ax.twinx()
                ax2.plot(
                    factors,
                    [float(row[ratio_key]) for row in data],
                    marker="v",
                    linewidth=line_width(stem, 1.35),
                    markersize=marker_size(stem, 4.6),
                    linestyle=":",
                    color="#CC3311",
                    label=ratio_label,
                )
                if not reciprocal_mode:
                    use_unit_coefficient_ticks(ax)
                align_zero_ticks(ax, ax2)
                ax2.tick_params(
                    axis="y",
                    labelsize=8,
                    colors="#CC3311",
                    pad=0,
                    width=line_width(stem, 0.8),
                    length=marker_size(stem, 3.5),
                )
                ax2.set_ylabel(ratio_label, fontsize=8, color="#CC3311", labelpad=3)
                ax2.set_box_aspect(1.0)
                for spine in ax2.spines.values():
                    spine.set_linewidth(line_width(stem, 0.8))
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(
                    handles1 + handles2,
                    labels1 + labels2,
                    frameon=False,
                    fontsize=8,
                    loc="upper left",
                    handlelength=0.72,
                    handletextpad=0.25,
                    labelspacing=0.02,
                    borderaxespad=0.02,
                    ncols=1,
                )

        fig.tight_layout(w_pad=0.15, h_pad=0.18)
        fig.subplots_adjust(left=0.072, right=0.958, top=0.950, bottom=0.108, wspace=0.28, hspace=0.10)
        for row_idx, (_, row_label) in enumerate(sampling_rows):
            row_axes = [axes[row_idx, col_idx] for col_idx in range(len(model_labels)) if axes[row_idx, col_idx].get_visible()]
            if not row_axes:
                continue
            y0 = min(ax.get_position().y0 for ax in row_axes)
            y1 = max(ax.get_position().y1 for ax in row_axes)
            fig.text(
                0.006,
                0.5 * (y0 + y1),
                row_label,
                rotation=90,
                va="center",
                ha="left",
                fontsize=8,
            )

        save_formats(fig, output_dir, stem, copy_to_paper)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.hist_bins < 100:
        raise ValueError("--hist-bins should be at least 100.")
    if args.stat_flush_steps < 1:
        raise ValueError("--stat-flush-steps must be >= 1.")
    factors = sorted(dict.fromkeys(int(factor) for factor in args.factors))
    if any(factor < 1 for factor in factors):
        raise ValueError("all factors must be >= 1")

    if args.plot_only_summary is not None and args.output_dir is None:
        args.output_dir = args.plot_only_summary.parent
    output_dir = resolve_output_dir(args.output_dir, args.overwrite)
    device = resolve_device(args.device)

    if args.plot_only_summary is not None:
        all_rows = read_rows(args.plot_only_summary if args.plot_only_summary.is_absolute() else ROOT / args.plot_only_summary)
        copy_to_paper = not args.no_paper_copy
        plot_dtscaled(all_rows, "downsample", output_dir, copy_to_paper)
        plot_dtscaled(all_rows, "upsample", output_dir, copy_to_paper)
        plot_expcap(all_rows, "downsample", output_dir, copy_to_paper)
        plot_expcap(all_rows, "upsample", output_dir, copy_to_paper)
        plot_combined_update_coefficients(all_rows, output_dir, copy_to_paper)
        print(f"[update-stats] redrew figures from {args.plot_only_summary}", flush=True)
        return

    all_rows: list[dict[str, object]] = []
    case_specs = build_case_specs(args)
    if args.sampling != "both":
        case_specs = [case for case in case_specs if case.sampling == args.sampling]
    for case in case_specs:
        all_rows.extend(
            build_case_rows(
                metadata_path=case.metadata_path,
                sampling=case.sampling,
                sequence_group=case.sequence_group,
                factors=factors,
                hist_bins=args.hist_bins,
                device=device,
                flush_steps=args.stat_flush_steps,
                exact_effective_median=args.exact_effective_median,
                median_work_dir=args.median_work_dir if args.exact_effective_median else None,
                run_dir_override=(
                    args.dtscaledgru_run_dir if case.model_label == "DtScaledGRU" else None
                ),
                data_dir_override=(
                    args.dtscaledgru_data_dir if case.model_label == "DtScaledGRU" else None
                ),
            )
        )

    write_rows(output_dir / "sampling_update_coefficient_summary.csv", all_rows)
    copy_to_paper = not args.no_paper_copy
    plot_dtscaled(all_rows, "downsample", output_dir, copy_to_paper)
    plot_dtscaled(all_rows, "upsample", output_dir, copy_to_paper)
    plot_expcap(all_rows, "downsample", output_dir, copy_to_paper)
    plot_expcap(all_rows, "upsample", output_dir, copy_to_paper)
    plot_combined_update_coefficients(all_rows, output_dir, copy_to_paper)

    metadata = {
        "output_dir": str(output_dir.resolve()),
        "paper_figure_dir": str(PAPER_FIGURE_DIR.resolve()) if copy_to_paper else None,
        "factors": factors,
        "device": str(device),
        "models": list(args.models),
        "hist_bins": int(args.hist_bins),
        "stat_flush_steps": int(args.stat_flush_steps),
        "median_policy": (
            "Effective coefficient medians are exact numpy.partition medians over temporary float32 files."
            if args.exact_effective_median
            else (
                "Medians are estimated from streaming histograms. DtScaled u_t and all bounded effective "
                "coefficients use linear histograms; ExpCap s_t uses a log histogram over [1e-8, 1e4]."
            )
        ),
        "exact_effective_median": bool(args.exact_effective_median),
        "median_work_dir": str(args.median_work_dir.resolve()) if args.exact_effective_median else None,
        "mean_policy": "Means are exact streaming sums over non-warmup, non-padding hidden-state update values.",
        "warmup_policy": (
            "Use ground-truth H while the current input point's original-grid "
            "coordinate is smaller than the evaluation warmup length; collect "
            "statistics from the first autoregressive input onward."
        ),
        "dtscaled_ut_mapping": (
            "DtScaledGRU uses the explicit reset-before equations in models.py. The plotted u_t is "
            "sigmoid(W_u x_t + U_u h_t + b_u), so h' = h + dt * u_t * (candidate - h)."
        ),
        "dtscaled_extrapolation_ratio": "dt u_t > 1 is counted from the effective DtScaledGRU update coefficient.",
        "expcap_boundary_ratio": "dt s_t > 0.8 is counted before applying expcap(dt s_t).",
        "source_metadata": [str(case.metadata_path.resolve()) for case in case_specs],
        "figure_files": sorted(path.name for path in output_dir.glob("*.pdf")),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"[update-stats] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
