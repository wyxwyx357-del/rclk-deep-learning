"""Scientific boundary checks for independent segments and padding."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.batches import pad_patient_runs, rcl_runs
from rclk_dl.tcn import PatientTCN


class TCNTests(unittest.TestCase):
    def test_fully_missing_family_splits_runs(self):
        valid = np.ones((5, 38), dtype=bool)
        valid[2, 16:24] = False
        record = {'time_s': np.arange(5),
                  'values': {'R': np.ones((5, 2, 8)), 'C': np.ones((5, 1, 8)),
                             'L': np.ones((5, 2, 7))},
                  'valid': {'R': valid[:, :16].reshape(5, 2, 8),
                            'C': valid[:, 16:24].reshape(5, 1, 8),
                            'L': valid[:, 24:].reshape(5, 2, 7)}}
        runs = rcl_runs(record)
        self.assertEqual([value.shape[1] for value, _ in runs], [2, 2])

    def test_padding_and_other_patient_do_not_change_logit(self):
        torch.manual_seed(7)
        model = PatientTCN(static_features=2, width=8).eval()
        a = np.ones((38, 4), dtype=np.float32)
        b = np.ones((38, 10), dtype=np.float32) * 3
        single = pad_patient_runs([[(a, np.ones_like(a, dtype=bool))]], [[0.2, 0.4]])
        mixed = pad_patient_runs([[(a, np.ones_like(a, dtype=bool))],
                                  [(b, np.ones_like(b, dtype=bool))]],
                                 [[0.2, 0.4], [0.1, 0.3]])
        with torch.no_grad():
            alone = model(*single)[0]
            padded = model(*mixed)[0]
        torch.testing.assert_close(alone, padded, rtol=0, atol=1e-6)

    def test_one_training_step_has_finite_loss(self):
        torch.manual_seed(7)
        model = PatientTCN(static_features=2, width=8).train()
        x = np.ones((38, 8), dtype=np.float32)
        batch = pad_patient_runs([[(x, np.ones_like(x, dtype=bool))],
                                  [(x * 2, np.ones_like(x, dtype=bool))]],
                                 [[0.2, 0.4], [0.1, 0.3]])
        logits = model(*batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
