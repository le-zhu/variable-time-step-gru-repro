from argparse import Namespace
import json
from pathlib import Path
import re
import tempfile
import unittest

from paper_experiments import PAPER_EXPERIMENTS
from scripts.run_paper_experiment import make_settings


class PaperExperimentTests(unittest.TestCase):
    def test_every_selected_configuration_builds(self):
        self.assertEqual(len(PAPER_EXPERIMENTS), 23)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            for experiment_id, spec in PAPER_EXPERIMENTS.items():
                with self.subTest(experiment=experiment_id):
                    args = Namespace(
                        experiment=experiment_id,
                        input_dir=input_dir,
                        run_name=f"test_{experiment_id}",
                        device="cpu",
                        reuse_processed=False,
                        no_test=False,
                        draw=False,
                    )
                    settings = make_settings(args)
                    self.assertEqual(settings.model_key, spec["model_key"])
                    self.assertEqual(settings.dt_norm_mode, spec["dt_norm_mode"])
                    self.assertEqual(settings.train.lr_12, spec["lr_12"])
                    self.assertEqual(settings.train.lr_3, spec["lr_3"])
                    self.assertEqual(settings.train.grad_clip_value, spec["grad_clip_value"])
                    filename = f"{spec['material']}_001_B.csv"
                    self.assertIsNotNone(re.match(settings.process.file_pattern_b, filename))

    def test_released_checkpoint_run_configs_match_paper_experiments(self):
        """The evaluation and figure scripts read the run_config.json written
        beside each checkpoint, so it must not drift from the checked settings."""
        from scripts.write_checkpoint_run_configs import build_run_config, checkpoint_dir_for

        for experiment_id, spec in PAPER_EXPERIMENTS.items():
            with self.subTest(experiment=experiment_id):
                path = checkpoint_dir_for(experiment_id, spec) / "run_config.json"
                self.assertTrue(path.is_file(), f"missing {path}")
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")),
                    build_run_config(experiment_id, spec),
                )


if __name__ == "__main__":
    unittest.main()
