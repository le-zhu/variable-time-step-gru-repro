"""Run one checked paper configuration without any server infrastructure."""

import argparse
import os
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from config import make_workflow_settings  # noqa: E402
from paper_experiments import COMMON_TRAINING, PAPER_EXPERIMENTS, get_paper_experiment  # noqa: E402
from workflow import run_workflow  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate one selected paper model from raw B/H/T CSV files."
    )
    parser.add_argument("--experiment", choices=PAPER_EXPERIMENTS, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Folder containing <material>_<id>_B.csv, _H.csv, and _T.csv files; required unless --reuse-processed is used.",
    )
    parser.add_argument("--run-name", help="Output folder name under runs/<model key>/.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device string.")
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Reuse the experiment's existing processed-data folder instead of processing CSV files.",
    )
    parser.add_argument("--no-test", action="store_true", help="Skip the held-out test evaluation.")
    parser.add_argument("--draw", action="store_true", help="Save held-out prediction plots.")
    return parser


def make_settings(args):
    spec = get_paper_experiment(args.experiment)
    if args.input_dir is None:
        if not args.reuse_processed:
            raise ValueError("--input-dir is required unless --reuse-processed is used.")
        input_dir = REPOSITORY_ROOT / "data" / "not_used"
    else:
        input_dir = args.input_dir.expanduser().resolve()
        if not args.reuse_processed and not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

    material = spec["material"]
    material_pattern = re.escape(material)
    process = {
        "file_pattern_b": rf"{material_pattern}_(\d+)_B\.csv",
        "file_pattern_h": rf"{material_pattern}_(\d+)_H\.csv",
        "file_pattern_t": rf"{material_pattern}_(\d+)_T\.csv",
        "input_dir": str(input_dir),
        "base_delta_t": 1.0,
        "min_len": 1000,
        "max_len": spec["max_len"],
        "validation_split": 0.1,
        "test_split": 0.1,
        "split_seed": 42,
    }
    train = {
        **COMMON_TRAINING,
        "grad_clip_value": spec["grad_clip_value"],
        "lr_12": spec["lr_12"],
        "lr_3": spec["lr_3"],
        "device": args.device,
    }
    tasks = {
        "process": not args.reuse_processed,
        "pretrain": True,
        "finetune": True,
        "test": not args.no_test,
        "draw": args.draw,
    }
    run_name = args.run_name or f"paper_{args.experiment}_seed42"
    data_name = f"paper_{args.experiment}_data"
    return make_workflow_settings(
        model_key=spec["model_key"],
        dt_norm_mode=spec["dt_norm_mode"],
        run_name=run_name,
        data_name=data_name,
        tasks=tasks,
        process=process,
        train=train,
    )


def main():
    args = build_parser().parse_args()
    os.chdir(REPOSITORY_ROOT)
    settings = make_settings(args)
    if Path(settings.run_dir).exists():
        raise FileExistsError(
            f"Run directory already exists: {settings.run_dir}. Use a new --run-name."
        )
    run_workflow(settings)


if __name__ == "__main__":
    main()
