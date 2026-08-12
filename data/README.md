# Data

No dataset is included. Users provide the raw B, H, and temperature CSV files
and run the public preprocessing/training entry point.

The measurements used in the paper are the MagNetX transient hysteresis dataset
(Kwon et al., APEC 2025, doi:10.1109/APEC48143.2025.10977252). The same
measurements are also distributed with the MagNet Challenge 2025
(<https://github.com/minjiechen/magnetchallenge-2>), but under a different
directory layout, so rename or re-lay-out the files to match the pattern below
before running the preprocessor.

## File naming

For each identifier, all three files must exist in the same directory:

```text
<material>_<id>_B.csv
<material>_<id>_H.csv
<material>_<id>_T.csv
```

Examples are `N87_001_B.csv`, `N87_001_H.csv`, and `N87_001_T.csv`.
Identifiers are matched as strings; only complete B/H/T triplets are used.

## CSV row format

- One row is one measured sequence.
- The B and H rows contain the time-ordered samples as numeric CSV cells.
- Matching B and H rows must have the same nonzero length.
- The matching T row supplies the scalar temperature in its first numeric cell.
- Header or nonnumeric cells are ignored by the parser.

The order and number of B, H, and T rows in a triplet must correspond. The
preprocessor uses the minimum row count among the three files.

## Split and preprocessing

- The split is performed at sequence level with seed 42.
- Validation and held-out test fractions are each 10%; the rest is training.
- Normalization statistics are computed from the training split only.
- The resampled training and validation length is at least 1000 and at most
  4000 for all released configurations except FixedGRU, whose maximum is 40000.
  The lower bound never removes a MagNetX sequence: the seven frequency groups
  have 2016 to 32016 original points, and the stride rule leaves 2016, 3216,
  2504, 2672, 3204, 3336, and 3558 points respectively.
- `std_db_dt` and `max_dt_to_1` are implemented in `process.py`; the selected
  mode for every checkpoint is recorded in `checkpoints/manifest.json`.

Generated arrays and normalization statistics are written below `runs/` and
are intentionally ignored by Git.

## Processed split expected by the evaluation and figure scripts

`scripts/evaluate_*.py` take the processed folder explicitly through
`--processed-data-dir`, so any location works. The sampling-interval scripts in
`scripts/evaluate_n87_*.py` and the waveform figure generators default to
`data/N87_processed`; either place the processed N87 folder there or pass
`--data-dir`.

A processed folder is one produced by `process.py`, that is, the
`origin_test*.npy` arrays together with `norm_stats.json` and
`process_config.json`. Predicted waveforms are never stored: they are large and
every script regenerates them from a checkpoint and this split.
