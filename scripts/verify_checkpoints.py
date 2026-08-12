"""Verify released checkpoint checksums and strict model compatibility."""

import hashlib
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models import get_model_entry  # noqa: E402


def main():
    checkpoint_root = REPOSITORY_ROOT / "checkpoints"
    manifest = json.loads((checkpoint_root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["n87_paper_models"] + manifest["ten_material_rategru"]
    for entry in entries:
        path = checkpoint_root / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        model = get_model_entry(entry["model_key"]).model_class()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"OK  {entry['path']}")
    print(f"Verified {len(entries)} manifest entries.")


if __name__ == "__main__":
    main()
