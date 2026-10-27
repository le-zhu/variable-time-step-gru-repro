import json
from pathlib import Path
import unittest

import numpy as np
import torch

from models import SoftplusRateExpCapGRU
from process import compute_resample_stride
from review_response.antialias.evaluate import antialiased_copy, lowpass
from review_response.input_distortion.quantify import quantify
from review_response.random_interval_materials.aggregate import aggregate, load_cells
from scripts.evaluate_random_interval_groups import (
    evaluate_groups, frequency_khz, stack_seed_grids,
)


REFERENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "review_response" / "random_interval_materials" / "reference_results"
)


def synthetic_split(lengths, seed=0):
    rng = np.random.default_rng(seed)
    padded = max(lengths)
    data = np.zeros((len(lengths), padded, 4), dtype=np.float32)
    for row, length in enumerate(lengths):
        phase = np.linspace(0.0, 2.0 * np.pi * 100.0, length)
        data[row, :length, 0] = np.sin(phase)
        data[row, :length, 2] = rng.normal()
        data[row, :length, 3] = np.cos(phase)
    return data, np.array(lengths, dtype=np.int64), np.full(len(lengths), 0.01, dtype=np.float32)


class RandomIntervalProtocolTests(unittest.TestCase):
    def test_frequency_groups_and_training_factors(self):
        expected = {2016: (800, 1), 3216: (500, 1), 5008: (320, 2), 8015: (200, 3),
                    12816: (125, 4), 20016: (80, 6), 32016: (50, 9)}
        for length, (khz, factor) in expected.items():
            self.assertEqual(frequency_khz(length), khz)
            self.assertEqual(compute_resample_stride(length, 4000), factor)

    def test_random_grids_stay_inside_the_step_cap(self):
        data, lengths, deltas = synthetic_split([500, 500, 480])
        _stacked, grid_lengths, dt, indices, per_seed = stack_seed_grids(
            data, lengths, deltas, [0, 1, 2], seeds=[42, 43], max_step=3
        )
        self.assertEqual(len(grid_lengths), 6)
        for seed_indices in per_seed:
            for row, row_indices in enumerate(seed_indices):
                steps = np.diff(row_indices)
                self.assertEqual(row_indices[0], 0)
                self.assertTrue(np.all((steps >= 1) & (steps <= 3)))
                # The trailing partial interval is dropped, never padded.
                self.assertLess(row_indices[-1], lengths[row])
                self.assertGreaterEqual(row_indices[-1], lengths[row] - 3)
        self.assertTrue(np.all(dt[0, : grid_lengths[0] - 1] > 0))

    def test_unresampled_groups_are_skipped(self):
        data, lengths, deltas = synthetic_split([300, 300, 900])
        torch.manual_seed(0)
        model = SoftplusRateExpCapGRU().eval()
        summary, _groups, rows = evaluate_groups(
            model, data, lengths, deltas, material="X", train_max_len=400, seeds=[42, 43],
            warmup=10, device=torch.device("cpu"), log=lambda _message: None,
        )
        self.assertEqual([row["original_length"] for row in summary], [900])
        self.assertEqual(summary[0]["max_step"], 3)
        self.assertEqual(len(rows), 2)


class ReviewAggregateTests(unittest.TestCase):
    def test_reference_aggregate_matches_reference_summary(self):
        summary = REFERENCE_DIR / "random_interval_materials_summary.csv"
        reference = json.loads((REFERENCE_DIR / "aggregate.json").read_text(encoding="utf-8"))
        result = aggregate(load_cells(summary))
        self.assertEqual(result["total"]["cells"], 48)
        self.assertEqual(result["total"]["sequences"], 6729)
        self.assertEqual(json.loads(json.dumps(result)), reference)


class DistortionTests(unittest.TestCase):
    def test_factor_one_is_distortion_free(self):
        data, lengths, _deltas = synthetic_split([20016])
        rows = quantify(data, lengths, 1.0, factors=[1, 4], groups={20016: 80.0})
        self.assertEqual(rows[0]["b_total_pct"], 0.0)
        self.assertEqual(rows[0]["dbdt_total_pct"], 0.0)
        self.assertGreater(rows[1]["b_total_pct"], 0.0)

    def test_antialias_filters_only_b(self):
        data, lengths, _deltas = synthetic_split([1000])
        self.assertIs(antialiased_copy(data, lengths, 1), data)
        filtered = antialiased_copy(data, lengths, 4)
        np.testing.assert_array_equal(filtered[..., 1:], data[..., 1:])
        np.testing.assert_allclose(lowpass(np.ones(64), 4), np.ones(64), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
