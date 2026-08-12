from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


METADATA_KEY = "__metadata_json__"


def save_plot_cache(path: str | Path, *, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload[METADATA_KEY] = np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    np.savez_compressed(target, **payload)


def load_plot_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != METADATA_KEY}
        metadata_raw = str(np.asarray(data[METADATA_KEY]).item())
    return arrays, json.loads(metadata_raw)
