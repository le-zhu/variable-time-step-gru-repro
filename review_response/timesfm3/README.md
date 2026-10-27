# TimesFM-3 Zero-Shot Comparison

Code for the comparison with TimesFM-3 in the response to Reviewer 2, Comment 4.
It is not part of the manuscript.

## What is compared

- Model: `google/timesfm-3.0-pytorch` at revision
  `43046b85ec22d584a13f8098c2ed39c889e129c2`, 330,710,976 parameters, used
  zero-shot: no training, fine-tuning, or tuning on this task.
- Data: the N87 held-out split on its original 16 MHz grid, 1261 sequences, the
  split of experiment `n87_rategru`.
- Target H; past-and-future covariates B and dB/dt (dB/dt z-normalized per
  sequence). Temperature is constant within each sequence and is not passed.
- Sequences longer than the 15,360-point context limit keep their last 15,360
  points (80 of 1261). The forecast horizon is produced in one call.
- History fraction f = 0.1, ..., 0.9: the first f of each (truncated) sequence
  supplies the measured H, the rest is predicted and scored with the
  per-sequence relative RMS error.
- RateGRU: the released `n87:rategru` checkpoint under the same rule, receiving
  the measured H for the first f of the points and then running freely.

The scored interval differs from the manuscript tables (fixed 100-point
warmup), so these numbers are not comparable with them. `protocol.py` holds the
rules both scripts share.

## Licenses

The TimesFM source code is Apache-2.0. The TimesFM-3 weights are distributed
under the TimesFM Non-Commercial License v1.0 and are not included here;
`download_model.py` fetches them from Hugging Face. This repository's MIT
license does not cover either. The forecasts and errors in
`reference_results/` are model outputs.

## Setup

Use the repository environment (`requirements.txt`, which provides torch) and
add the pinned TimesFM package, ideally in a separate virtual environment that
reuses the installed torch:

```powershell
python -m venv --system-site-packages review_response\timesfm3\.venv
review_response\timesfm3\.venv\Scripts\python.exe -m pip install -r review_response\timesfm3\requirements-timesfm.txt
review_response\timesfm3\.venv\Scripts\python.exe -m review_response.timesfm3.download_model
```

`download_model.py` checks every file against `model_source.json`;
`source_files_sha256.json` lists the hashes of the three TimesFM source files
the comparison depends on.

## Run

Both scripts read the processed N87 split written by
`scripts/run_paper_experiment.py --experiment n87_rategru`, here assumed at
`runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data`.

```powershell
# TimesFM-3 columns: 20 to 30 minutes on an 8 GB laptop GPU (RTX 5060), batch 16
review_response\timesfm3\.venv\Scripts\python.exe -m review_response.timesfm3.timesfm3_context_sweep `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data

# RateGRU columns: a few minutes, CPU or GPU
python -m review_response.timesfm3.rategru_context_sweep `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data
```

Results are written to `results/timesfm3/`. `--max-length-groups 1` runs
TimesFM-3 on the 80 longest sequences only, as a quick installation check.

## Reference results

`reference_results/` holds the outputs behind the table in the response letter,
regenerated with the commands above from a fresh install of
`requirements-timesfm.txt`; they agree with the original run to the last digit
for TimesFM-3 and within 1e-10 for RateGRU.
