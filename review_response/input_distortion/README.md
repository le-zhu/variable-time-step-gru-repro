# Input distortion of uniform downsampling

Quantifies how much uniform decimation distorts the two model inputs, B and
dB/dt, for the N87 80 kHz and 50 kHz test groups, with direct decimation and
with an ideal zero-phase antialiasing low-pass before decimation. These are the
B-channel and dB/dt-channel distortion tables of the response letter. No model
is involved.

```powershell
python -m review_response.input_distortion.quantify `
  --processed-data-dir runs\SoftplusRateExpCapGRU\data\paper_n87_rategru_data
```

Pure NumPy; a few seconds on the CPU. Writes `downsampling_distortion.csv` and
`.json` to `results/input_distortion/`. The letter prints a subset of the
factors (1, 2, 4, 6, 9, 12, 16, 24, 32).
