import csv
import hashlib
import json
from pathlib import Path
import unittest

from paper_experiments import PAPER_EXPERIMENTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = REPOSITORY_ROOT / "sweep_logs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_rows():
    with (SWEEP_ROOT / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class SweepLogTests(unittest.TestCase):
    def test_manifest_covers_all_logs_and_sanitized_configs(self):
        rows = manifest_rows()
        self.assertEqual(len(rows), 228)
        self.assertEqual(len(list(SWEEP_ROOT.rglob("run_*_log.csv"))), len(rows))
        self.assertEqual(len(list(SWEEP_ROOT.rglob("run_*_config.json"))), len(rows))
        self.assertEqual(len({(row["group"], row["run_id"]) for row in rows}), len(rows))

        forbidden = ("/root/", "C:\\Users\\", "autodl", "private_key", "access_token")
        for row in rows:
            with self.subTest(group=row["group"], run_id=row["run_id"]):
                log_path = SWEEP_ROOT / row["log_path"]
                config_path = SWEEP_ROOT / row["group"] / f"{row['run_id']}_config.json"
                self.assertEqual(sha256(log_path), row["log_sha256"])
                self.assertEqual(sha256(config_path), row["config_sha256"])
                config_text = config_path.read_text(encoding="utf-8")
                self.assertFalse(any(text.lower() in config_text.lower() for text in forbidden))
                config = json.loads(config_text)
                self.assertNotIn("input_dir", config.get("process", {}))
                self.assertNotIn("device", config.get("train", {}))
                train = config["train"]
                self.assertEqual(train["phase1_teacher_forcing_epochs"], 50)
                self.assertEqual(train["phase2_curriculum_transition_epochs"], 50)
                self.assertEqual(train["phase3_finetuning_epochs"], 150)
                # The manuscript reports a single clipping threshold, so every
                # archived run of both searches must carry that value.
                self.assertEqual(train["grad_clip_value"], 20.0)

    def test_the_two_searches_share_only_the_documented_runs(self):
        """N87's RateGRU RMS-slope runs at lr12 = 0.001 belong to the controlled
        comparison and to the per-material search, so they are the only logs the
        two folders have in common."""
        rows = manifest_rows()

        by_hash = {}
        for row in rows:
            by_hash.setdefault(row["log_sha256"], []).append(row)
        shared = [group for group in by_hash.values() if len(group) > 1]

        self.assertEqual(len(shared), 5)
        for group in shared:
            with self.subTest(run=group[0]["original_run_name"]):
                self.assertEqual(len(group), 2)
                self.assertEqual(
                    {row["group"] for row in group},
                    {"controlled_n87/rategru_rms_slope", "ten_materials/N87"},
                )
                self.assertEqual(float(group[0]["lr_12"]), 0.001)

        for folder, expected in (("controlled_n87/", 60), ("ten_materials/", 168)):
            with self.subTest(folder=folder):
                counted = sum(1 for row in rows if row["group"].startswith(folder))
                self.assertEqual(counted, expected)

    def test_the_index_carries_the_selection_criterion_only(self):
        """The manuscript selects on validation error. Each log additionally
        records one automatic end-of-training held-out evaluation, so the index
        must not offer held-out metrics next to the criterion."""
        columns = set(manifest_rows()[0])
        self.assertTrue({"final_val_error", "final_val_p95"} <= columns)
        self.assertFalse(any("test" in column for column in columns))

    def test_no_released_checkpoint_comes_from_a_diverged_run(self):
        """Eight archived runs diverge to a non-finite loss. They are kept as
        they ran, so the archive has to show that none of them was selected."""
        released = {
            (value["material"], value["model_key"], value["dt_norm_mode"],
             value["lr_12"], value["lr_3"])
            for value in PAPER_EXPERIMENTS.values()
        }

        diverged = set()
        for row in manifest_rows():
            log = (SWEEP_ROOT / row["log_path"]).read_text(encoding="utf-8")
            if "nan" in log.lower() or "inf," in log.lower():
                diverged.add((
                    row["material"], row["model_key"], row["dt_norm_mode"],
                    float(row["lr_12"]), float(row["lr_3"]),
                ))

        self.assertTrue(diverged, "expected the archive to keep its diverged runs")
        self.assertEqual(diverged & released, set())

    def test_every_material_search_covers_the_lr12_anchor_row(self):
        """The manuscript scans lr3 over four values at lr12 = 0.001 for all ten
        materials; the archive must show that row in full."""
        rows = manifest_rows()
        anchor = {0.002, 0.004, 0.006, 0.008}
        materials = {
            "3F4", "N87", "N27", "N49", "78", "77", "3C94", "3C90", "N30", "3E6",
        }
        for material in sorted(materials):
            with self.subTest(material=material):
                scanned = {
                    float(row["lr_3"])
                    for row in rows
                    if row["material"] == material
                    and row["stage"] == "phase3"
                    and row["model_key"] == "SoftplusRateExpCapGRU"
                    and row["dt_norm_mode"] == "std_db_dt"
                    and float(row["lr_12"]) == 0.001
                }
                self.assertTrue(anchor <= scanned, f"missing {sorted(anchor - scanned)}")


if __name__ == "__main__":
    unittest.main()
