"""Synthetic sequence and pooling checks; no patient data or fitting."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.batches import pad_patient_runs, rcl_runs
from rclk_dl.frame_mlp import CHANNELS, PatientFrameMLP


class FrameMLPInputTests(unittest.TestCase):
    def test_family_gap_separates_runs_and_preserves_signed_values(self):
        r = np.ones((3, 2, 8), dtype=np.float32)
        r[0, 0, 0], r[2, 0, 0] = -2, 3
        c = np.ones((3, 1, 8), dtype=np.float32)
        c[1] = np.nan
        l = np.ones((3, 2, 7), dtype=np.float32)
        record = {'time_s': np.arange(3) / 35,
                  'values': {'R': r, 'C': c, 'L': l},
                  'valid': {'R': np.isfinite(r), 'C': np.isfinite(c),
                            'L': np.isfinite(l)}}
        runs = rcl_runs(record)
        self.assertEqual([value.shape for value, _ in runs], [(CHANNELS, 1)] * 2)
        self.assertEqual([value[0, 0] for value, _ in runs], [-2, 3])

    def test_order_and_right_padding_do_not_change_patient_output(self):
        torch.manual_seed(1)
        model = PatientFrameMLP(static_features=0).eval()
        short = np.zeros((CHANNELS, 2), dtype=np.float32)
        short[0] = [-2, 3]
        long = np.ones((CHANNELS, 4), dtype=np.float32)
        valid_short = np.ones_like(short, dtype=bool)
        valid_long = np.ones_like(long, dtype=bool)
        alone = pad_patient_runs([[(short, valid_short)]], np.empty((1, 0), np.float32))
        together = pad_patient_runs([[(short, valid_short)], [(long, valid_long)]],
                                    np.empty((2, 0), np.float32))
        reversed_run = pad_patient_runs([[(short[:, ::-1], valid_short[:, ::-1])]],
                                        np.empty((1, 0), np.float32))
        self.assertEqual(together[2][0].tolist(), [True, True, False, False])
        with torch.no_grad():
            single = model(*alone)[0]
            batched = model(*together)[0]
            reversed_output = model(*reversed_run)[0]
        self.assertTrue(torch.allclose(single, batched, atol=1e-6))
        self.assertTrue(torch.allclose(single, reversed_output, atol=1e-6))

    def test_local_missing_mask_is_distinct_from_observed_zero(self):
        model = PatientFrameMLP(static_features=0).eval()
        with torch.no_grad():
            model.frame.weight.zero_()
            model.frame.bias.zero_()
            model.frame.weight[0, CHANNELS] = 1
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.weight[0, 0] = 1
        values = np.zeros((CHANNELS, 1), dtype=np.float32)
        observed = np.ones_like(values, dtype=bool)
        missing = observed.copy()
        missing[0, 0] = False
        present_batch = pad_patient_runs([[(values, observed)]], np.empty((1, 0), np.float32))
        missing_batch = pad_patient_runs([[(values, missing)]], np.empty((1, 0), np.float32))
        with torch.no_grad():
            present = model(*present_batch)[0]
            absent = model(*missing_batch)[0]
        self.assertGreater(present.item(), absent.item())


if __name__ == '__main__':
    unittest.main()

