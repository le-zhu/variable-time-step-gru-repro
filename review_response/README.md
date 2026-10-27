# Analyses Reported in the Response to Reviewers

This folder holds the code for results that appear only in the response to the
reviewers of the first revision. None of them is part of the manuscript, and
none of them retrains a model: every script reads a released checkpoint from
`checkpoints/` and a processed split written by `scripts/run_paper_experiment.py`.

| Response item | Folder | Needs | Run time |
| --- | --- | --- | --- |
| Reviewer 2, Comment 4: zero-shot TimesFM-3 against RateGRU | `timesfm3/` | TimesFM-3 package and weights (see its README), `n87:rategru`, N87 split | 20 to 30 min on an 8 GB GPU |
| Reviewer 3, Comment 5: input distortion caused by downsampling | `input_distortion/` | N87 split | seconds, CPU |
| Reviewer 3, Comment 5: direct versus anti-aliased downsampling at 50 kHz | `antialias/` | `n87:rategru`, `n87:dtscaledgru`, `n87:dtinputgru`, N87 split | a few minutes |
| Reviewer 3, Comment 6: warmup-length sensitivity | `warmup/` | `n87:rategru`, N87 split | about 5 min per split |
| Reviewer 3, Comment 9: random nonuniform grids on ten materials | `random_interval_materials/` | the ten `material:<M>` checkpoints and their splits | about 25 min on a GPU |

Each folder has a README with the exact command and protocol. The N87 split is
the one of experiment `n87_rategru`; all scripts take its location through
`--processed-data-dir`. Outputs go to `results/`, which is not tracked.

The N87 table of nonuniform grids in the revised manuscript (Table V) is a
manuscript result and is produced by `scripts/evaluate_random_interval_groups.py`;
`random_interval_materials/` runs the same evaluator on all ten materials.
