"""Write a sanitized ``run_config.json`` next to every released checkpoint.

The released weights are plain ``state_dict`` files, so on their own they do not
say which model class, normalization, or learning rates produced them. This
writes the checked configuration from ``paper_experiments.py`` beside each one,
in the same schema ``workflow.py`` uses for a locally trained run, with the raw
input directory left empty because no data ships with this repository.

The evaluation scripts under ``scripts/`` read these files, so re-run this after
changing ``paper_experiments.py``. ``tests/test_paper_experiments.py`` checks
that the two stay consistent.
"""

import argparse
import json
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paper_experiments import COMMON_TRAINING, PAPER_EXPERIMENTS  # noqa: E402


# Keyed by experiment identifier, not by paper name: the joint ablation reports
# DtScaledGRU and RateGRU under both normalization schemes, so a paper name can
# denote two different released checkpoints.
N87_CHECKPOINT_DIRS = {
    "n87_fixedgru": "fixedgru",
    "n87_dtscaledgru": "dtscaledgru",
    "n87_rategru": "rategru",
    "n87_dtinputgru": "dtinputgru",
    "n87_rategru_orig": "rategru_orig",
    "n87_hidden32": "hidden32",
    "n87_no_h": "no_h",
    "n87_dtscaledgru_max_step": "dtscaledgru_max_step",
    "n87_rategru_max_step": "rategru_max_step",
    "n87_hardcut_max_step": "hardcut_max_step",
    "n87_hardcut_rms_slope": "hardcut_rms_slope",
    "n87_globalexp_max_step": "globalexp_max_step",
    "n87_globalexp_rms_slope": "globalexp_rms_slope",
}


def checkpoint_dir_for(experiment_id, spec):
    """Map an experiment identifier onto its released checkpoint directory."""
    root = REPOSITORY_ROOT / "checkpoints"
    if experiment_id.startswith("material_"):
        return root / "ten_material_rategru" / spec["material"]
    return root / "n87_paper_models" / N87_CHECKPOINT_DIRS[experiment_id]


def build_run_config(experiment_id, spec):
    material = re.escape(spec["material"])
    return {
        "model_key": spec["model_key"],
        "dt_norm_mode": spec["dt_norm_mode"],
        "run_name": f"paper_{experiment_id}_seed42",
        "data_name": f"paper_{experiment_id}_data",
        "tasks": {"process": False, "pretrain": False, "finetune": False,
                  "test": True, "draw": False},
        "process": {
            "file_pattern_b": rf"{material}_(\d+)_B\.csv",
            "file_pattern_h": rf"{material}_(\d+)_H\.csv",
            "file_pattern_t": rf"{material}_(\d+)_T\.csv",
            # Empty by design: no raw data ships with this repository.
            "input_dir": "",
            "base_delta_t": 1.0,
            "min_len": 1000,
            "max_len": spec["max_len"],
            "validation_split": 0.1,
            "test_split": 0.1,
            "split_seed": 42,
        },
        "train": {**COMMON_TRAINING,
                  "grad_clip_value": spec["grad_clip_value"],
                  "lr_12": spec["lr_12"],
                  "lr_3": spec["lr_3"],
                  "device": "auto"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report drift instead of writing.")
    args = parser.parse_args()

    stale = []
    for experiment_id, spec in PAPER_EXPERIMENTS.items():
        target = checkpoint_dir_for(experiment_id, spec) / "run_config.json"
        payload = json.dumps(build_run_config(experiment_id, spec), indent=2) + "\n"
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current == payload:
            continue
        if args.check:
            stale.append(str(target.relative_to(REPOSITORY_ROOT)))
        else:
            target.write_text(payload, encoding="utf-8")
            print(f"wrote {target.relative_to(REPOSITORY_ROOT)}")
    if args.check:
        if stale:
            print("Out of date:\n  " + "\n  ".join(stale))
            raise SystemExit(1)
        print(f"All {len(PAPER_EXPERIMENTS)} checkpoint run configurations are up to date.")


if __name__ == "__main__":
    main()
