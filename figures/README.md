# Figures

Every figure in the manuscript has a generator here. Two of them are produced by
the Section VI evaluation scripts, because the same pass that computes the
reported numbers also draws the figure; they are listed below with their real
location.

The generators in this directory write to `figures/output/`, which is not
tracked. The two in `scripts/` write their full result set to a run directory
under `results/` and place only the manuscript PDF in `figures/output/`:
`evaluate_n87_longest_downsample.py` does so by default and takes
`--no-copy-to-paper` to suppress it, while `evaluate_n87_frequency_groups.py`
needs `--copy-to-paper` to do it at all.

| Fig. | Manuscript file | Generator | Needs |
| --- | --- | --- | --- |
| 1 | `transient_prediction_task.pdf` | `figures/plot_transient_task_overview.py` | processed N87 |
| 2 | `model_overview.pdf` | `figures/plot_model_overview.py` | nothing |
| 3 | `update_regimes.pdf` | `figures/plot_update_regimes.py` | nothing |
| 4 | `limiter_function_comparison.pdf` | `figures/plot_limiter_function_comparison.py` | nothing |
| 5 | `data_processing_pipeline.pdf` | `figures/plot_data_processing_pipeline.py` | nothing |
| 6 | `three_stage_training.pdf` | `figures/plot_three_stage_training.py` | nothing |
| 7 | `no_h_initial_sync_comparison.pdf` | `figures/plot_no_h_initial_sync_waveform.py` | RateGRU and RateGRU-NF (`no_h`) checkpoints, processed N87 |
| 8 | `n87_sampling_downsampling_low2_with_dtinput.pdf` | `scripts/evaluate_n87_frequency_groups.py` | DtScaledGRU, RateGRU, and DtInputGRU checkpoints, processed N87 |
| 9 | `uniform_waveforms_16_32.pdf` | `scripts/evaluate_n87_longest_downsample.py` | RateGRU checkpoint, processed N87 |
| 10 | `update_coeff_downsampling_diagnostics.pdf` | `figures/plot_update_coeff_downsampling_single_column.py` | the summary CSV below |
| 11 | `random_interval_representative_zoom_points_1to9_window750_meanlike.pdf` | `figures/plot_random_interval_representative_zoom.py` | RateGRU checkpoint, processed N87 |
| 12 | `restest_step_reduction.pdf` | `figures/plot_restest_step_reduction.py` | the ten evaluation outputs below |
| 13 | `ten_material_error_comparison.pdf` | `figures/plot_ten_material_comparison.py` | nothing |

The six figures that need nothing take no arguments:

```powershell
python -m figures.plot_model_overview
python -m figures.plot_update_regimes
python -m figures.plot_limiter_function_comparison
python -m figures.plot_data_processing_pipeline
python -m figures.plot_three_stage_training
python -m figures.plot_ten_material_comparison
```

The comparison figure carries no data dependency because every value in it is
transcribed from a published table; its docstring names the three sources.

The cost figure reads the same evaluation that produces the Test and ResTest
columns of the cross-material table. Run it once per material into its own
subdirectory, then point the figure at the parent:

```powershell
python scripts/evaluate_resampled_holdout.py --checkpoint material:N87 `
  --processed-data-dir <processed split for N87> `
  --out-dir results\restest\N87
python -m figures.plot_restest_step_reduction --results-dir results\restest
```

Step totals are summed from each `resampled_holdout_row_errors.csv`; the plotted
errors are read from each `resampled_holdout_summary.json` rather than
recomputed, so the P95 is the statistic the evaluator reported.

The per-frequency-group Test and ResTest means quoted with Table IV (for
example DtInputGRU 10.104% vs 5.453% and RateGRU 5.767% vs 5.480% in the 50 kHz
group) regroup the same per-sequence CSV by raw sequence length; no model is
rerun:

```powershell
foreach ($m in "rategru", "dtinputgru") {
  python scripts/evaluate_resampled_holdout.py --checkpoint n87:$m `
    --processed-data-dir <processed split for N87> `
    --out-dir results\table4_by_length\$m
}
python scripts/summarize_holdout_by_length.py `
  results\table4_by_length\rategru results\table4_by_length\dtinputgru
```

The table is printed and written to
`results\table4_by_length\per_group_mean_error.csv`.

The three model-dependent waveform and sampling figures below read the processed
N87 split (`std_db_dt`, the one the released N87 checkpoints were trained on)
from `data/N87_processed` unless `--data-dir` points elsewhere. Their defaults
or the arguments shown reproduce the revised manuscript figures.

Fig. 7, RateGRU against RateGRU-NF. The defaults rank the longest validation
sequences and draw rank 6 (`origin_val` row 324); RateGRU-NF is rolled out
without teacher forcing and scored after the same 100-step warmup as RateGRU:

```powershell
python -m figures.plot_no_h_initial_sync_waveform --output-dir results\no_h_initial_sync
```

Fig. 8, low-frequency downsampling for DtScaledGRU, RateGRU, and DtInputGRU.
`--low2-only` evaluates just the 80 and 50 kHz downsampling cases the figure
draws; without it the script also runs the upsampling and training-range cases
of the full diagnostics for DtScaledGRU and RateGRU:

```powershell
python -m scripts.evaluate_n87_frequency_groups --low2-only --copy-to-paper
```

Fig. 11, the representative random-interval waveform. The row whose seed-45
error is closest to the seed mean is chosen first, then the 750-sample window
in that row whose local error is closest to the same mean (row 1253, samples
21601 to 22351); the annotation is the error of the whole row:

```powershell
python -m figures.plot_random_interval_representative_zoom `
  --max-step 9 --seed 45 --auto-window-span 750 --meanlike-row `
  --stem random_interval_representative_zoom_points_1to9_window750_meanlike `
  --copy-to-paper
```

The update-coefficient figure is drawn in two steps. The diagnostics pass
computes the per-factor statistics of `dt * u_t` and `expcap(dt * s_t)` and
writes `sampling_update_coefficient_summary.csv`; the second command renders the
single-column panel pair that the manuscript shows:

```powershell
python -m figures.plot_update_coeff_sampling_diagnostics `
  --output-dir results\n87_update_coefficients
python -m figures.plot_update_coeff_downsampling_single_column `
  --summary results\n87_update_coefficients\sampling_update_coefficient_summary.csv
```

The diagnostics pass draws its own multi-panel overview as well. That overview
is a working figure and does not appear in the manuscript; it is kept because
the same pass writes the summary CSV.

`no_h_waveform_common.py` is not a generator. It holds the sequence loading and
rollout helpers that `plot_no_h_initial_sync_waveform.py` imports.

`paper_figure_sizes.py`, `paper_plot_style.py`, and `paper_plot_cache.py` are
shared: they fix the per-figure page geometry, the Matplotlib style, and the
cache format that lets a waveform figure be redrawn without rerunning the model.
