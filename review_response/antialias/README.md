# Direct versus antialiased downsampling

Feeds the 23 sequences of the N87 50 kHz test group to the released RateGRU,
DtScaledGRU, and DtInputGRU checkpoints (`n87:rategru`, `n87:dtscaledgru`,
`n87:dtinputgru`) at downsampling factors 1, 9, 16, and 32, once directly
decimated and once after an ideal zero-phase low-pass on B. This is the
antialiasing table of the response letter; the direct column is the protocol of
the manuscript's downsampling figure.

```powershell
python -m review_response.antialias.evaluate `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data
```

About six minutes on a laptop GPU, dominated by the 32016-step rollouts at
factor 1; the CPU also works, more slowly. Writes
`results/antialias/antialias_downsampling.csv` (errors as fractions).
