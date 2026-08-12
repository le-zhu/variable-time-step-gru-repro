# Sweep Coverage Against the Manuscript

The manuscript states two different search rules, one for the controlled model
comparison on N87 and a wider one for the per-material training of the ten
materials. The archive keeps them apart, so each folder holds exactly its own
scan range:

```text
controlled_n87/   the common-protocol search behind the model comparison
ten_materials/    the wider per-material search, one folder per material
manifest.csv      one row per run in either folder
```

N87's RateGRU RMS-slope runs at `lr12 = 0.001` are part of both searches, so
they appear in both folders, in the same way the N87 RateGRU checkpoint is
released under both checkpoint collections. Those five runs are the only
duplication in the archive; `manifest.csv` lists 228 rows for 223 distinct runs.

## N87 controlled search

| Experiment group | Available phase-3 lr3 values |
| --- | --- |
| FixedGRU, Max-step | 0.001, 0.002, 0.004, 0.006, 0.008 |
| DtScaledGRU, Max-step | 0.001, 0.002, 0.004, 0.006, 0.008 |
| DtScaledGRU, RMS-slope | 0.002, 0.004, 0.006, 0.008 |
| RateGRU, Max-step | 0.002, 0.004, 0.006, 0.008, 0.010 |
| RateGRU, RMS-slope | 0.002, 0.004, 0.006, 0.008 |
| HardCutRateGRU, Max-step | 0.002, 0.004, 0.006, 0.008 |
| HardCutRateGRU, RMS-slope | 0.002, 0.004, 0.006, 0.008, 0.010 |
| GlobalExpRateGRU, Max-step | 0.002, 0.004, 0.006, 0.008 |
| GlobalExpRateGRU, RMS-slope | 0.002, 0.004, 0.006, 0.008, 0.010 |
| Hidden32 | 0.002, 0.004, 0.006, 0.008 |
| No-H | 0.002, 0.004, 0.006, 0.008 |

The RateGRU Max-step `lr3=0.010` boundary extension was run after the rest of
that group, under the same paper protocol and from the same phase-1/2
checkpoint. Its validation error did not improve on the selected `lr3=0.008`
result.

## Per-material search

The manuscript scans `lr3` over {0.002, 0.004, 0.006, 0.008} at `lr12 = 0.001`
and, in addition, at the two values of `lr12` in {0.0005, 0.001, 0.0015, 0.002}
that give the lowest first-stage training loss at epoch 50. `ten_materials/`
holds that search for every one of the ten materials. Each of them contributes:

- four phase-1/2 screening logs per material, one for each `lr12` candidate,
  which are the evidence for the epoch-50 ranking;
- the four phase-3 runs of the `lr12 = 0.001` row;
- the phase-3 runs of the retained `lr12` values. For eight materials the
  ranking retains {0.0015, 0.002}, so both rows are present. For 3E6 it retains
  {0.001, 0.002} and for N87 {0.0015, 0.001}, so in those two cases one retained
  row is the `lr12 = 0.001` row itself and only the second row is additional.

Boundary extensions outside {0.002, 0.004, 0.006, 0.008} are present where they
were run: upward to `lr3 = 0.010` and `0.015`, downward to `0.001` and `0.0005`.
None of them produced a selected configuration.

Two cells of the `lr12 = 0.001` row, `lr3 = 0.002` and `lr3 = 0.004`, were run
later than the rest for eight of the ten materials; the `original_run_name`
column identifies which runs those are. Selection was carried out once, over the
completed grid, and `checkpoints/manifest.json` records its outcome for every
material.

## What the logs record

Each `run_*_log.csv` is a verbatim copy of the log the training run wrote, so it
carries every epoch of the three stages. The last line of a phase-3 log is a
single end-of-training evaluation on the held-out test set, which the training
entry point performs automatically for every run. Those numbers took no part in
any selection: the manuscript selects on validation error, and `manifest.csv`
therefore indexes the validation columns only. The screening logs stop at epoch
50 and carry no validation error, because the epoch-50 first-stage training loss
is what ranks the `lr12` candidates.

Eight of the 228 runs diverge to a non-finite loss during phase 3 and are ended
by the early-stopping rule; the archive keeps them as they ran. Seven of the
eight sit at `lr3 = 0.008` or `0.010`, the upper end of the scanned range. In
seven cases early stopping restores the best weights from before the divergence
and the run still reports a validation error; the exception is the `78` run at
`lr12 = 0.002`,
`lr3 = 0.002`, which diverges before any checkpoint is recorded and is listed
with an empty validation error. None of the eight is a selected configuration,
so no released checkpoint comes from a diverged run.

The gradient-clipping threshold is 20 for every archived run, matching the
single value the manuscript reports and the value recorded in every released
`run_config.json`. The archive covers the search the manuscript describes, so
runs at other thresholds, which belong to no reported comparison, are not part
of it.

Counting the whole archive, `ten_materials/` contributes four phase-1/2
screening logs for each of the ten materials, and every group of
`controlled_n87/` contributes one phase-1/2 log for each configuration it
compares.
