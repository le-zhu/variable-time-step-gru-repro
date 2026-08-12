"""Evaluate one released checkpoint on a matching processed held-out split."""

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from config import TrainSettings  # noqa: E402
from models import get_model_entry  # noqa: E402
from scripts.paper_eval import load_checkpoint_entries  # noqa: E402
import train  # noqa: E402


def main():
    entries = load_checkpoint_entries()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=entries, required=True)
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        required=True,
        help="Folder produced by process.py for the same material, split, and dt normalization.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    entry = entries[args.checkpoint]
    checkpoint_path = REPOSITORY_ROOT / "checkpoints" / entry["path"]
    data_dir = args.processed_data_dir.expanduser().resolve()
    process_config_path = data_dir / "process_config.json"
    if not process_config_path.is_file():
        raise FileNotFoundError(f"Processed-data configuration not found: {process_config_path}")
    process_config = json.loads(process_config_path.read_text(encoding="utf-8"))
    expected_norm = entry.get("dt_normalization", "std_db_dt")
    if process_config.get("dt_norm_mode") != expected_norm:
        raise ValueError(
            "Processed data uses dt_norm_mode="
            f"{process_config.get('dt_norm_mode')!r}; checkpoint expects {expected_norm!r}."
        )
    model_entry = get_model_entry(entry["model_key"])
    cfg = train.build_train_config(
        run_dir=str(checkpoint_path.parent),
        data_dir=str(data_dir),
        model_class=model_entry.model_class,
        train_settings=TrainSettings(device=args.device),
    )
    train.set_reproducibility(cfg.seed)
    model = train.create_model(cfg)
    state = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    test_dataset, test_indices = train.prepare_test_data(cfg)
    mean_error, stats = train.evaluate_full_length_loss(
        model, test_dataset, test_indices, cfg, desc="[Released checkpoint]"
    )
    # The evaluator returns a fraction; the manuscript prints percent.
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Held-out mean relative error: {mean_error * 100:.3f}%")
    print(f"Held-out P95 relative error: {stats['p95_row_err'] * 100:.3f}%")


if __name__ == "__main__":
    main()
