# Ten-Material RateGRU Models

This folder contains the selected RateGRU checkpoint for each material in the
paper's ten-material table:

`N87`, `N49`, `3C94`, `3E6`, `78`, `3F4`, `77`, `N30`, `N27`, and `3C90`.

Every file loads into the `SoftplusRateExpCapGRU` class. Material-specific
learning rates, held-out errors, resampled held-out errors, core-loss errors,
and checksums are recorded in `../manifest.json`; the gradient-clipping
threshold is 20 for all ten. Each folder also holds a `run_config.json` giving
the full checked configuration, which the evaluation and figure scripts read.
