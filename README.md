# Variable-Time-Step GRU for Magnetic Hysteresis

> Reproducibility repository for the variable-time-step GRU manuscript.

## Scope

This repository contains the infrastructure-independent implementation, the
selected paper configurations, and the released model weights for the
variable-time-step GRU manuscript. Raw data and the manuscript itself are not
included.

The paper-to-code model names are:

| Paper name | Code key |
| --- | --- |
| FixedGRU | `StandardGRU` |
| DtScaledGRU | `DtScaledGRU` |
| RateGRU | `SoftplusRateExpCapGRU` |
| Hidden32 | `SoftplusRateExpCapGRUHidden32` |
| No-H | `SoftplusRateExpCapGRUNoH` |
| HardCutRateGRU | `SoftplusRateHardcutGRU` |
| GlobalExpRateGRU | `SoftplusRateGlobalExpGRU` |

Every model row of every results table in the manuscript has released weights.
DtScaledGRU and RateGRU appear twice, once per time-step normalization scheme,
so the checkpoint folders carry a `_max_step` suffix for the Max-step variant.

## Current contents

- Core model definitions, preprocessing, training, rollout, and evaluation
- The checked single-run configurations for every released paper model
- Eleven N87 checkpoints: the five of the controlled comparison and structural
  ablation, and the six remaining rows of the normalization/limiter ablation
- Ten material-specific RateGRU checkpoints, each with its `run_config.json`
- Checkpoint checksums and strict loading tests
- Sanitized training logs for the paper-reported hyperparameter searches
- Scripts for every reported metric: held-out and resampled sequence error,
  core-loss error, and the sampling-interval experiments
- Generators for all eleven manuscript figures

## Layout

```text
*.py                  core model, preprocessing, training, rollout, evaluation
paper_experiments.py  the selected configuration of every released model
checkpoints/          released weights, run configurations, and checksums
scripts/              command entry points: train, verify, evaluate
figures/              manuscript figure generators and their shared style
sweep_logs/           sanitized per-run training logs of the two paper searches
tests/                loading, configuration, and archive-consistency checks
data/                 format description only; no data ships here
```

## Not included

- Training or evaluation data
- Server controllers, credentials, logs, or local machine paths
- The manuscript source or PDF

## Platform and cost

The repository is about 5 MB unpacked and 2.5 MB compressed. It is pure Python
and runs on Windows, Linux, and macOS; every command below was run on Windows 11
with Python 3.12.0 and the pinned versions in `requirements.txt`, which are also
the versions in which the released checkpoints were loaded and verified. A CUDA
GPU is used
when PyTorch reports one and the CPU is used otherwise, so no GPU is required.
Recomputing one reported metric takes under a minute; the verification commands
below take seconds. Retraining a configuration runs the full three-stage
schedule of 50 + 50 + 150 epochs and is the only step that wants a GPU.

## Data source

The measurements are the MagNetX transient hysteresis dataset (Kwon et al.,
APEC 2025, doi:10.1109/APEC48143.2025.10977252). The same measurements are also
distributed with the MagNet Challenge 2025
(<https://github.com/minjiechen/magnetchallenge-2>) under a different directory
layout, so check the file naming below before pointing a script at a download.

## Expected data layout

Supply one directory per material. Each sample set is represented by three CSV
files with matching identifiers:

```text
N87_001_B.csv
N87_001_H.csv
N87_001_T.csv
N87_002_B.csv
N87_002_H.csv
N87_002_T.csv
```

See `data/README.md` for the row format and split behavior.

## Reproduce one selected configuration

```powershell
python -m pip install -r requirements.txt
python scripts/run_paper_experiment.py `
  --experiment n87_rategru `
  --input-dir C:\path\to\raw\N87
```

Use `python scripts/run_paper_experiment.py --help` to list all eleven N87 and
ten material-specific experiment identifiers. Outputs are written below `runs/`.

The entry point fixes the sequence split seed, preprocessing mode, three-stage
schedule, selected learning rates, and gradient-clipping threshold, and writes
the resolved configuration into the run directory before training starts.

This command retrains the already-selected configuration. It does not rerun the
historical hyperparameter sweep that selected that configuration.

## Verify released weights

```powershell
python scripts/verify_checkpoints.py
python -m pytest -q
```

The paper sweep evidence is indexed by `sweep_logs/manifest.csv`. It is split
into `controlled_n87/`, the common-protocol search behind the model comparison,
and `ten_materials/`, the wider per-material search, because the manuscript
states a different scan rule for each. Together they hold 228 per-run training
logs and sanitized configurations; no server controller output or machine path
is included. `sweep_logs/COVERAGE.md` maps both folders onto the search rules
stated in the manuscript.

## Recompute a reported metric

Every command below takes a released checkpoint and a processed split, so none
of them retrains anything.

```powershell
# Held-out sequence error
python scripts/evaluate_checkpoint.py --checkpoint n87:rategru `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data

# Held-out and resampled sequence error, the "Test" and "ResTest" columns
python scripts/evaluate_resampled_holdout.py --checkpoint material:3C94 `
  --processed-data-dir <processed split for 3C94>

# Core-loss error, the "epsilon_W" columns
python scripts/evaluate_core_loss.py --checkpoint material:3C94 `
  --processed-data-dir <processed split for 3C94>
```

## Sampling-interval experiments

These reproduce the nonuniform-grid and downsampling results. They read the
released N87 checkpoints and a processed N87 split, which the commands below
expect at `data/N87_processed`.

```powershell
# Nonuniform grids, four seeds
python -m scripts.evaluate_n87_longest_random_interval `
  --max-step 10 --seeds 42 43 44 45 --output-dir results\n87_random_interval

# Downsampling of the longest sequences
python -m scripts.evaluate_n87_longest_downsample `
  --output-dir results\n87_longest_downsample_rategru

# Frequency-group sampling diagnostics; add --copy-to-paper for the figure
python -m scripts.evaluate_n87_frequency_groups
```

The first two finish in minutes. The third sweeps every frequency group under
both up- and downsampling for two models and took about an hour on one GPU.

## Figures

`figures/README.md` maps every manuscript figure to the script that draws it and
gives the commands. Five figures are schematics and need nothing but the
repository:

```powershell
python -m figures.plot_model_overview
python -m figures.plot_data_processing_pipeline
python -m figures.plot_three_stage_training
python -m figures.plot_limiter_function_comparison
python -m figures.plot_update_regimes
```

The rest read a released checkpoint, a processed split, or the CSV written by a
sampling-interval run. Two of them are drawn by the sampling-interval scripts in
`scripts/`, because the pass that computes the reported numbers also renders the
figure. All output goes to `figures/output/`, which is not tracked.

## License

MIT, in `LICENSE`. It covers everything released here: the code, the released
checkpoints, and the archived training logs.

## Contact

Yenan Chen (yenanc@zju.edu.cn), corresponding author, College of Electrical
Engineering, Zhejiang University, Hangzhou 310027, China.
