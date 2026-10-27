"""Download the pinned TimesFM-3 checkpoint and verify every file.

The weights are distributed by Google under the TimesFM Non-Commercial License
and are therefore not part of this repository. This script fetches the exact
revision used for the review response from Hugging Face and checks each file
against ``model_source.json``: Git blob hashes for the small files and the LFS
SHA-256 for ``model.safetensors`` (about 1.3 GB). Files that already verify are
skipped, so the script can also be used to check an existing download.

    python -m review_response.timesfm3.download_model
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def file_matches(path, meta):
    if not path.is_file() or path.stat().st_size != meta["size"]:
        return False
    if "lfs" in meta:
        digest, expected = hashlib.sha256(), meta["lfs"]["sha256"]
    else:
        # Git blob hash: sha1 over "blob <size>\0" followed by the content.
        digest, expected = hashlib.sha1(), meta["blobId"]
        digest.update(f"blob {meta['size']}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=HERE / "model")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check an existing download without contacting the network.")
    args = parser.parse_args()

    source = json.loads((HERE / "model_source.json").read_text(encoding="utf-8"))
    if source["repo_id"] != REPO_ID or source["revision"] != REVISION:
        raise ValueError("model_source.json does not describe the pinned revision")

    args.dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for meta in source["files"]:
        name = meta["rfilename"]
        if Path(name).name != name:
            raise ValueError(f"Unexpected checkpoint path: {name}")
        target = args.dest / name
        if file_matches(target, meta):
            print(f"verified {name}", flush=True)
            continue
        if args.verify_only:
            failed.append(name)
            continue
        from huggingface_hub import hf_hub_download

        print(f"downloading {name} ({meta['size']:,} bytes)", flush=True)
        hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=args.dest)
        if not file_matches(target, meta):
            failed.append(name)
        else:
            print(f"verified {name}", flush=True)
    if failed:
        raise SystemExit(f"Size or hash mismatch: {', '.join(failed)}")
    print(f"All {len(source['files'])} files of {REPO_ID}@{REVISION[:8]} verified in {args.dest}")


if __name__ == "__main__":
    main()
