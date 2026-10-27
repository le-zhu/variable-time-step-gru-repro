"""Random nonuniform grids for the ten released per-material RateGRU checkpoints.

Runs ``scripts.evaluate_random_interval_groups`` on every material with its
released checkpoint (``material:<M>``) and the default protocol: random step
capped at each frequency group's training downsampling factor, seeds 42-45,
a 100-point warmup, and the original measured grid as the comparison column.
The combined CSV is the input of ``aggregate.py`` and ``plot.py``.

Each material needs its own processed split (``process.py`` output, std_db_dt,
split seed 42). Point ``--data-root`` at a folder holding one sub-folder per
material, named by ``--dir-pattern``; the default pattern is the folder that
``scripts/run_paper_experiment.py --experiment material_<M>`` writes. Single
materials can be redirected with ``--processed-data-dir M=PATH``.

    python -m review_response.random_interval_materials.run_materials `
      --data-root runs\\SoftplusRateExpCapGRU\\data
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_random_interval_groups import (  # noqa: E402
    DEFAULT_SEEDS, DEFAULT_WARMUP, evaluate_groups, load_split, write_csv,
)
from scripts.paper_eval import (  # noqa: E402
    check_normalization, load_checkpoint_entries, load_model, resolve_device,
)


MATERIALS = ("N87", "N49", "3C94", "3E6", "78", "3F4", "77", "N30", "N27", "3C90")
SUMMARY_NAME = "random_interval_materials_summary.csv"


def parse_overrides(values):
    overrides = {}
    for value in values or []:
        material, sep, path = value.partition("=")
        if not sep or material not in MATERIALS:
            raise ValueError(f"Expected M=PATH with M in {MATERIALS}, got {value!r}")
        overrides[material] = Path(path)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Folder holding one processed split per material.")
    parser.add_argument("--dir-pattern", default="paper_material_{material}_data",
                        help="Sub-folder name per material; {material} is substituted.")
    parser.add_argument("--processed-data-dir", action="append", metavar="M=PATH",
                        help="Explicit processed split for one material; repeatable.")
    parser.add_argument("--materials", nargs="+", default=list(MATERIALS), choices=MATERIALS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--original-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "random_interval_materials")
    args = parser.parse_args()

    overrides = parse_overrides(args.processed_data_dir)
    data_dirs = {}
    for material in args.materials:
        if material in overrides:
            data_dirs[material] = overrides[material]
        elif args.data_root is not None:
            data_dirs[material] = args.data_root / args.dir_pattern.format(material=material)
        else:
            parser.error(f"No processed split for {material}: give --data-root or --processed-data-dir")
        if not (data_dirs[material] / "origin_test.npy").is_file():
            raise FileNotFoundError(f"{material}: no origin_test.npy in {data_dirs[material]}")

    entries = load_checkpoint_entries()
    device = resolve_device(args.device)
    summary, row_records, metadata = [], [], {}
    for material in args.materials:
        key = f"material:{material}"
        entry = entries[key]
        check_normalization(data_dirs[material], entry)
        data, lengths, deltas, process_config = load_split(data_dirs[material], "origin_test")
        model = load_model(entry, device)
        started = time.perf_counter()
        rows, groups, records = evaluate_groups(
            model, data, lengths, deltas, material=material,
            train_max_len=int(process_config["max_len"]), seeds=args.seeds, warmup=args.warmup,
            device=device, batch_size=args.batch_size, original_batch_size=args.original_batch_size,
            log=lambda message: print(message, flush=True),
        )
        summary += rows
        row_records += records
        metadata[material] = {
            "checkpoint": key,
            "checkpoint_path": entry["path"],
            "checkpoint_sha256": entry["sha256"],
            "test_row_count": int(len(lengths)),
            "processed_data": {k: process_config.get(k) for k in (
                "dt_norm_mode", "min_len", "max_len", "validation_split", "test_split", "split_seed")},
            "seconds": round(time.perf_counter() - started, 1),
            "groups": groups,
        }
        del model, data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / SUMMARY_NAME, summary)
    write_csv(args.output_dir / "random_interval_materials_row_errors.csv", row_records)
    (args.output_dir / "random_interval_materials_metadata.json").write_text(json.dumps({
        "protocol": {
            "seeds": args.seeds,
            "warmup_original_steps": args.warmup,
            "step_cap": "group training downsampling factor ceil(length / max_len)",
            "groups": "training factor > 1 (320, 200, 125, 80, 50 kHz)",
            "comparison": "original measured grid (stride 1)",
        },
        "device": str(device),
        "materials": metadata,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output_dir / SUMMARY_NAME}", flush=True)


if __name__ == "__main__":
    main()
