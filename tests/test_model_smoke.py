import unittest

import torch

from models import DtScaledGRU, SoftplusRateExpCapGRU, StandardGRU


class ModelSmokeTests(unittest.TestCase):
    def test_paper_comparison_models_produce_expected_shapes(self):
        batch_size = 2
        model_input = torch.zeros(batch_size, 3)
        previous_h = torch.zeros(batch_size, 1)
        dt = torch.ones(batch_size, 1)

        for model_class in (StandardGRU, DtScaledGRU, SoftplusRateExpCapGRU):
            with self.subTest(model=model_class.__name__):
                model = model_class()
                hidden_state = torch.zeros(batch_size, model.hidden_size)
                next_h, next_hidden = model.forward_step(
                    model_input,
                    previous_h,
                    hidden_state,
                    dt,
                )
                self.assertEqual(next_h.shape, (batch_size, 1))
                self.assertEqual(next_hidden.shape, (batch_size, model.hidden_size))
                self.assertTrue(torch.isfinite(next_h).all())
                self.assertTrue(torch.isfinite(next_hidden).all())


if __name__ == "__main__":
    unittest.main()
