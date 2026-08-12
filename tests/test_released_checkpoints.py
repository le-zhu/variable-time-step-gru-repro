import hashlib
import json
from pathlib import Path
import unittest

import torch

from models import get_model_entry
from scripts.paper_eval import load_checkpoint_entries


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ReleasedCheckpointTests(unittest.TestCase):
    def test_manifest_checkpoints_match_checksum_and_model(self):
        checkpoint_root = REPOSITORY_ROOT / "checkpoints"
        manifest = json.loads((checkpoint_root / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest["n87_paper_models"] + manifest["ten_material_rategru"]
        self.assertEqual(len(entries), 21)

        for entry in entries:
            with self.subTest(path=entry["path"]):
                path = checkpoint_root / entry["path"]
                self.assertTrue(path.is_file())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, entry["sha256"])
                model = get_model_entry(entry["model_key"]).model_class()
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)

    def test_every_released_model_uses_the_reported_clipping_threshold(self):
        # The manuscript reports a single gradient-clipping threshold of 20.
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["n87_paper_models"] + manifest["ten_material_rategru"]
        self.assertEqual({entry["gradient_clip_norm"] for entry in entries}, {20.0})

    def test_ten_material_entries_carry_every_reported_column(self):
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        required = {"validation_mean", "test_mean", "test_p95",
                    "resampled_heldout_mean", "resampled_heldout_p95",
                    "core_loss_mean", "core_loss_p95"}
        for entry in manifest["ten_material_rategru"]:
            with self.subTest(material=entry["material"]):
                self.assertTrue(required <= set(entry))

    def test_both_collections_name_their_metrics_the_same_way(self):
        """The scripts cross-check a recomputed value against the manifest by
        field name, so a metric that changes name between the two collections
        would silently skip that check for one of them."""
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        metrics = {"validation_mean", "validation_p95", "test_mean", "test_p95",
                   "resampled_heldout_mean", "resampled_heldout_p95",
                   "core_loss_mean", "core_loss_p95"}
        seen = {collection: metrics & {key for entry in manifest[collection] for key in entry}
                for collection in ("n87_paper_models", "ten_material_rategru")}
        shared = seen["n87_paper_models"] & seen["ten_material_rategru"]
        self.assertTrue({"test_mean", "test_p95", "validation_mean"} <= shared)

    def test_n87_entries_carry_every_reported_column(self):
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        # The five main-comparison models are also tested on the held-out set;
        # the joint-ablation rows are reported on the validation set only.
        tested = {"n87_paper_models/fixedgru/model.pth",
                  "n87_paper_models/dtscaledgru/model.pth",
                  "n87_paper_models/rategru/model.pth",
                  "n87_paper_models/hidden32/model.pth",
                  "n87_paper_models/no_h/model.pth"}
        for entry in manifest["n87_paper_models"]:
            with self.subTest(path=entry["path"]):
                self.assertTrue({"validation_mean", "validation_p95"} <= set(entry))
                if entry["path"] not in tested:
                    continue
                self.assertTrue({"test_mean", "test_p95", "resampled_heldout_mean",
                                 "resampled_heldout_p95"} <= set(entry))
                # The manuscript prints N/A for FixedGRU: its resampled length
                # cap equals the raw length, so resampling is a no-op there.
                expected_null = entry["paper_name"] == "FixedGRU"
                self.assertEqual(entry["resampled_heldout_mean"] is None, expected_null)

    def test_every_released_checkpoint_has_its_own_evaluation_key(self):
        """Four models are released under both normalizations, so a key built
        from the paper name alone would let one shadow another and evaluate the
        wrong weights without saying so."""
        entries = load_checkpoint_entries()
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        released = {entry["path"] for collection in
                    ("n87_paper_models", "ten_material_rategru")
                    for entry in manifest[collection]}
        self.assertEqual({entry["path"] for entry in entries.values()}, released)
        self.assertEqual(len(entries), len(released))

    def test_joint_ablation_rows_are_released(self):
        """All eight rows of the normalization/limiter ablation must have weights."""
        manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )
        released = {(entry["paper_name"], entry["dt_normalization"])
                    for entry in manifest["n87_paper_models"]}
        for paper_name in ("DtScaledGRU", "RateGRU", "HardCutRateGRU", "GlobalExpRateGRU"):
            for norm in ("max_dt_to_1", "std_db_dt"):
                with self.subTest(model=paper_name, normalization=norm):
                    self.assertIn((paper_name, norm), released)


if __name__ == "__main__":
    unittest.main()
