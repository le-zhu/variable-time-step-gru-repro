# N87 Paper Models

This folder holds the thirteen N87 checkpoints. The seven below are the
controlled comparison and the structural ablation, for which the paper reports
both validation and held-out test errors.

| Folder | Paper name | Code key | Validation mean / P95 (%) | Test mean / P95 (%) |
| --- | --- | --- | ---: | ---: |
| `fixedgru` | FixedGRU | `StandardGRU` | 7.303 / 14.865 | 7.143 / 13.863 |
| `dtinputgru` | DtInputGRU | `DtInputFixedGRU` | 4.759 / 8.968 | 4.686 / 8.882 |
| `dtscaledgru` | DtScaledGRU | `DtScaledGRU` | 4.886 / 7.222 | 4.747 / 7.284 |
| `rategru` | RateGRU | `SoftplusRateExpCapGRU` | 3.503 / 5.167 | 3.389 / 5.457 |
| `rategru_orig` | RateGRU-Orig | `SoftplusRateExpCapGRU` | 7.734 / 16.225 | 7.544 / 15.716 |
| `hidden32` | RateGRU-32 | `SoftplusRateExpCapGRUHidden32` | 4.169 / 6.184 | 4.028 / 6.191 |
| `no_h` | RateGRU-NF | `SoftplusRateExpCapGRUNoH` | 4.821 / 8.323 | 4.825 / 8.611 |

The percentages above are the paper values. Machine-readable values and
checksums are recorded in `../manifest.json`.

DtInputGRU and RateGRU-Orig were added in the revision. DtInputGRU is trained
on the same resampled grid as RateGRU. RateGRU-Orig is RateGRU trained on the
original grid only, like FixedGRU, so its processed split is built with
`max_len = 40000` (experiment `n87_rategru_orig`) and, as for FixedGRU, the
resampled held-out columns are N/A. The ablation models were renamed in the
revision (Hidden32 to RateGRU-32, No-H to RateGRU-NF); the folder names are
unchanged.

The remaining six folders complete the joint ablation of time-step
normalization and limiting functions, and carry a `_max_step` or `_rms_slope`
suffix because that table reports both normalization schemes for each limiter.
The paper selects on validation error alone at that stage, so `../manifest.json`
records validation mean and P95 for them and no test columns.
