# Random nonuniform grids on ten materials

Extends the manuscript's N87 nonuniform-grid table to the ten released
per-material RateGRU checkpoints (`material:<M>` in `checkpoints/manifest.json`).
The per-material evaluation is `scripts/evaluate_random_interval_groups.py`;
this folder runs it on every material, aggregates the cells, and draws the
response-letter figure.

Protocol: every frequency group whose training downsampling factor is above 1
(320, 200, 125, 80, and 50 kHz; factors 2, 3, 4, 6, 9), random steps from 1 to
that factor, seeds 42-45, a 100-point warmup on the original grid, and the
original measured grid as the comparison column. Ten materials give 48
material x group cells (N49 has three groups).

```powershell
# One processed split per material, e.g. from run_paper_experiment.py
python -m review_response.random_interval_materials.run_materials `
  --data-root runs\SoftplusRateExpCapGRU\data
python -m review_response.random_interval_materials.aggregate `
  --summary results\random_interval_materials\random_interval_materials_summary.csv
python -m review_response.random_interval_materials.plot `
  --summary results\random_interval_materials\random_interval_materials_summary.csv
```

`--dir-pattern` names the per-material sub-folder (default
`paper_material_{material}_data`); `--processed-data-dir M=PATH` overrides one
material. The run took 25 minutes on a shared laptop GPU (RTX 5060, 8 GB) and
also runs on the CPU, more slowly.

`aggregate.py` applies the letter's conventions: sequence-weighted means of the
cell errors, weighted random over weighted original for the ratios, the
sequence-weighted mean of the cell P95 values, and plain means of the cell
ratios per training factor.

`reference_results/` holds the summary CSV and the aggregate JSON of the
released checkpoints, for comparison without a GPU.
