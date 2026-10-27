# Warmup-length sensitivity

Rolls the released N87 RateGRU (`n87:rategru`) out on the processed N87 test
split with no warmup, the measured initial H only, and warmups of 10, 50, 100,
500, and 1000 points, scoring every setting on the same two fixed windows
(after point 100 and after point 1000). This is the warmup table of the
response letter, summarized in Section VI of the revised manuscript; nothing is
retrained.

```powershell
python -m review_response.warmup.sensitivity `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data
# The validation-split trend quoted in the letter
python -m review_response.warmup.sensitivity --split origin_val `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data
```

By default the whole split is rolled out in one batch, as for the reported
numbers: a peak of about 1.3 GB of GPU memory and five to six minutes on a
laptop GPU per split.
`--batch-size` lowers the memory at the cost of possible last-digit differences.
Writes `results/warmup/rategru_rms_<split>.json` (errors as fractions).
