"""Synthetic input-contract checks; no patient data or model fitting."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.flat_mlp import CHANNELS, INPUT_FEATURES, MAX_FRAMES, flatten_record


class FlatMLPInputTests(unittest.TestCase):
    def test_signed_order_missing_family_and_padding(self):
        r = np.zeros((3, 2, 8), dtype=np.float32)
        r[0, 0, 0], r[1, 0, 0], r[2, 0, 0] = -2, 3, 4
        c = np.ones((3, 1, 8), dtype=np.float32)
        l = np.ones((3, 2, 7), dtype=np.float32)
        c[2] = np.nan
        record = {'time_s': np.arange(3) / 35,
                  'values': {'R': r, 'C': c, 'L': l},
                  'valid': {'R': np.isfinite(r), 'C': np.isfinite(c),
                            'L': np.isfinite(l)}}
        flat = flatten_record(record, np.zeros(CHANNELS), np.ones(CHANNELS))
        self.assertEqual(flat.shape, (INPUT_FEATURES,))
        grid = flat.reshape(MAX_FRAMES, CHANNELS * 2)
        self.assertEqual(grid[0, 0], -2)
        self.assertEqual(grid[1, 0], 3)
        self.assertEqual(grid[0, CHANNELS], 1)
        self.assertEqual(grid[2].sum(), 0)  # C gap breaks TCN history.
        self.assertEqual(grid[3:].sum(), 0)  # Right padding is never a measurement.

    def test_longer_patient_is_rejected_instead_of_truncated(self):
        count = MAX_FRAMES + 1
        r = np.ones((count, 2, 8), dtype=np.float32)
        c = np.ones((count, 1, 8), dtype=np.float32)
        l = np.ones((count, 2, 7), dtype=np.float32)
        record = {'time_s': np.arange(count) / 35,
                  'values': {'R': r, 'C': c, 'L': l},
                  'valid': {'R': np.isfinite(r), 'C': np.isfinite(c),
                            'L': np.isfinite(l)}}
        with self.assertRaisesRegex(ValueError, 'do not truncate'):
            flatten_record(record, np.zeros(CHANNELS), np.ones(CHANNELS))


if __name__ == '__main__':
    unittest.main()
