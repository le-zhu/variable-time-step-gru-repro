# N87 Paper Models

This folder holds the eleven N87 checkpoints. The five below are the controlled
comparison and the structural ablation, for which the paper reports both
validation and held-out test errors.

| Folder | Paper name | Code key | Validation mean / P95 (%) | Test mean / P95 (%) |
| --- | --- | --- | ---: | ---: |
| `fixedgru` | FixedGRU | `StandardGRU` | 7.303 / 14.865 | 7.143 / 13.863 |
| `dtscaledgru` | DtScaledGRU | `DtScaledGRU` | 4.886 / 7.222 | 4.747 / 7.284 |
| `rategru` | RateGRU | `SoftplusRateExpCapGRU` | 3.503 / 5.167 | 3.389 / 5.457 |
| `hidden32` | Hidden32 | `SoftplusRateExpCapGRUHidden32` | 4.169 / 6.184 | 4.028 / 6.191 |
| `no_h` | No-H | `SoftplusRateExpCapGRUNoH` | 4.821 / 8.323 | 4.825 / 8.611 |

The percentages above are the paper values. Machine-readable values and
checksums are recorded in `../manifest.json`.

The remaining six folders complete the joint ablation of time-step
normalization and limiting functions, and carry a `_max_step` or `_rms_slope`
suffix because that table reports both normalization schemes for each limiter.
The paper selects on validation error alone at that stage, so `../manifest.json`
records validation mean and P95 for them and no test columns.
