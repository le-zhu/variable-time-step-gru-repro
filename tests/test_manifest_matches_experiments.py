import json
from pathlib import Path
import unittest

from paper_experiments import PAPER_EXPERIMENTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ManifestConsistencyTests(unittest.TestCase):
    """`paper_experiments.py` drives retraining and `checkpoints/manifest.json`
    describes the released weights. They must state the same configuration, or a
    reader would retrain something other than what was released."""

    def setUp(self):
        self.manifest = json.loads(
            (REPOSITORY_ROOT / "checkpoints" / "manifest.json").read_text(encoding="utf-8")
        )

    def _assert_matches(self, entry, name, experiment):
        self.assertEqual(experiment["model_key"], entry["model_key"])
        self.assertEqual(experiment["lr_12"], entry["lr_phase12"])
        self.assertEqual(experiment["lr_3"], entry["lr_phase3"])
        self.assertEqual(experiment["grad_clip_value"], entry["gradient_clip_norm"])

    def test_n87_models_match(self):
        # Keyed by name and normalization: the joint ablation reports
        # DtScaledGRU and RateGRU under both normalization schemes.
        by_name = {(value["paper_name"], value["dt_norm_mode"]): (key, value)
                   for key, value in PAPER_EXPERIMENTS.items()
                   if value["material"] == "N87"}
        for entry in self.manifest["n87_paper_models"]:
            key = (entry["paper_name"], entry["dt_normalization"])
            with self.subTest(model=key):
                self.assertIn(key, by_name)
                name, experiment = by_name[key]
                self._assert_matches(entry, name, experiment)

    def test_ten_materials_match(self):
        by_material = {value["material"]: (key, value)
                       for key, value in PAPER_EXPERIMENTS.items()
                       if key.startswith("material_")}
        for entry in self.manifest["ten_material_rategru"]:
            with self.subTest(material=entry["material"]):
                self.assertIn(entry["material"], by_material)
                name, experiment = by_material[entry["material"]]
                self._assert_matches(entry, name, experiment)


if __name__ == "__main__":
    unittest.main()
