"""Input distortion caused by uniform downsampling, with and without antialiasing.

Produces the B and dB/dt input-distortion tables of the response letter for the
80 kHz (20016-point) and 50 kHz (32016-point) groups of a processed N87 split.
No model is involved; this is pure NumPy and runs on the CPU.

* Direct decimation keeps ``B[::r]``, the rule of ``process.py``.
* The antialiased reference applies a zero-phase ideal FFT low-pass with cutoff
  ``1 / (2 r dt0)`` to the whole sequence before decimation.
* B channel: relative RMS error of the decimated points linearly interpolated
  back onto the original grid.
* dB/dt channel: relative RMS error between the coarse forward difference and
  the original-grid forward difference at the same anchors.

Each value is the mean over the group's sequences, in percent, relative to the
original 16 MHz signal. The direct and antialiased columns are two separate
norms, not an additive decomposition.

    python -m review_response.input_distortion.quantify `
      --processed-data-dir runs\\SoftplusRateExpCapGRU\\data\\paper_n87_rategru_data
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FACTORS = (1, 2, 4, 6, 8, 9, 12, 16, 20, 24, 28, 32)
LENGTHS = {20016: 80.0, 32016: 50.0}


def lowpass(x, fc, fs):
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(x.size, 1.0 / fs) > fc] = 0.0
    return np.fft.irfft(spectrum, n=x.size)


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def quantify(data, lengths, dt0, factors=FACTORS, groups=LENGTHS):
    fs = 1.0 / dt0
    rows = []
    for length, nominal_khz in groups.items():
        index = np.flatnonzero(lengths == length)
        b_all = data[index][:, :length, 0].astype(np.float64)
        grid = np.arange(length)
        for factor in factors:
            fc = fs / (2.0 * factor)
            anchors = np.arange(0, length, factor)
            acc_b = np.zeros(2)
            acc_d = np.zeros(2)
            for series in b_all:
                filtered = lowpass(series, fc, fs)
                dec = series[::factor]
                dec_lp = filtered[::factor]
                rec = np.interp(grid, anchors, dec)
                rec_lp = np.interp(grid, anchors, dec_lp)
                acc_b += [rel_rms(rec_lp, series), rel_rms(rec, series)]

                fine_slope = np.diff(series) / dt0
                coarse_slope = np.diff(dec) / (dt0 * factor)
                coarse_slope_lp = np.diff(dec_lp) / (dt0 * factor)
                reference = fine_slope[: coarse_slope.size * factor : factor]
                acc_d += [rel_rms(coarse_slope_lp, reference), rel_rms(coarse_slope, reference)]
            count = float(len(b_all))
            b_terms = acc_b / count * 100.0
            d_terms = acc_d / count * 100.0
            rows.append({
                "length": length,
                "nominal_khz": nominal_khz,
                "factor": factor,
                "points_per_subcycle": length / 100.0 / factor,
                "b_total_pct": b_terms[1],
                "b_total_antialias_pct": b_terms[0],
                "dbdt_total_pct": d_terms[1],
                "dbdt_total_antialias_pct": d_terms[0],
            })
            print(
                f"{length} r={factor:<3} pts/sub={length / 100.0 / factor:6.2f}  "
                f"B direct={b_terms[1]:5.3f}% antialias={b_terms[0]:5.3f}%  "
                f"dB/dt direct={d_terms[1]:6.2f}% antialias={d_terms[0]:6.2f}%",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-data-dir", type=Path, required=True,
                        help="Processed N87 split (process.py output).")
    parser.add_argument("--split", default="origin_test", choices=["origin_test", "origin_val"])
    parser.add_argument("--factors", type=int, nargs="+", default=list(FACTORS))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "input_distortion")
    args = parser.parse_args()

    data_dir = args.processed_data_dir
    data = np.load(data_dir / f"{args.split}.npy", mmap_mode="r")
    lengths = np.load(data_dir / f"{args.split}_lengths.npy")
    dt0 = float(np.load(data_dir / f"{args.split}_delta_t.npy")[0])
    rows = quantify(data, lengths, dt0, factors=args.factors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "downsampling_distortion"
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".json").write_text(json.dumps({
        "split": args.split,
        "base_delta_t": dt0,
        "group_sequences": {str(length): int((lengths == length).sum()) for length in LENGTHS},
        "note": ("*_antialias_pct: the same error after an ideal zero-phase low-pass with cutoff "
                 "1/(2 r dt0) before decimation"),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
