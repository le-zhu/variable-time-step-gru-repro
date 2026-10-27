import unittest

from scripts.summarize_holdout_by_length import group_by_length, nominal_frequency_khz


class HoldoutByLengthTests(unittest.TestCase):
    def test_magnetx_lengths_map_to_nominal_frequencies(self):
        expected = {2016: 800.0, 3216: 500.0, 5008: 320.0, 8015: 200.0, 12816: 125.0, 20016: 80.0, 32016: 50.0}
        for length, khz in expected.items():
            with self.subTest(length=length):
                self.assertEqual(nominal_frequency_khz(length), khz)

    def test_group_means_are_per_length_percentages(self):
        rows = [
            {"raw_length": "32016", "stride": "9", "holdout_error": "0.10", "resampled_holdout_error": "0.05"},
            {"raw_length": "32016", "stride": "9", "holdout_error": "0.12", "resampled_holdout_error": "0.07"},
            {"raw_length": "2016", "stride": "1", "holdout_error": "0.03", "resampled_holdout_error": "0.03"},
        ]
        table = group_by_length(rows, "DtInputGRU")
        self.assertEqual([row["raw_length"] for row in table], [2016, 32016])
        longest = table[1]
        self.assertEqual(longest["n_rows"], 2)
        self.assertEqual(longest["resample_stride"], "9")
        self.assertAlmostEqual(longest["test_mean_pct"], 11.0)
        self.assertAlmostEqual(longest["resampled_test_mean_pct"], 6.0)
        self.assertAlmostEqual(longest["difference_pct_points"], 5.0)


if __name__ == "__main__":
    unittest.main()
