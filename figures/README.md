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

| Manuscript file | Generator | Needs |
| --- | --- | --- |
| `transient_prediction_task.pdf` | `figures/plot_transient_task_overview.py` | processed N87 |
| `model_overview.pdf` | `figures/plot_model_overview.py` | nothing |
| `data_processing_pipeline.pdf` | `figures/plot_data_processing_pipeline.py` | nothing |
| `three_stage_training.pdf` | `figures/plot_three_stage_training.py` | nothing |
| `update_regimes.pdf` | `figures/plot_update_regimes.py` | nothing |
| `limiter_function_comparison.pdf` | `figures/plot_limiter_function_comparison.py` | nothing |
| `no_h_initial_sync_comparison.pdf` | `figures/plot_no_h_initial_sync_waveform.py` | RateGRU and No-H checkpoints, processed N87 |
| `n87_sampling_downsampling_low2.pdf` | `scripts/evaluate_n87_frequency_groups.py` | DtScaledGRU and RateGRU checkpoints, processed N87 |
| `uniform_waveforms_16_32.pdf` | `scripts/evaluate_n87_longest_downsample.py` | RateGRU checkpoint, processed N87 |
| `update_coeff_downsampling_diagnostics.pdf` | `figures/plot_update_coeff_downsampling_single_column.py` | the summary CSV below |
| `random_interval_representative_zoom_points_1to10_window750_meanlike.pdf` | `figures/plot_random_interval_representative_zoom.py` | RateGRU checkpoint, processed N87 |
| `restest_step_reduction.pdf` | `figures/plot_restest_step_reduction.py` | the ten evaluation outputs below |
| `ten_material_error_comparison.pdf` | `figures/plot_ten_material_comparison.py` | nothing |

The six figures that need nothing take no arguments:

```powershell
python -m figures.plot_model_overview
python -m figures.plot_data_processing_pipeline
python -m figures.plot_three_stage_training
python -m figures.plot_update_regimes
python -m figures.plot_limiter_function_comparison
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
