"""The long patient's frame count must not dominate sequence scaling."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.scaling import fit_family_scaler, scale_patient_runs


class ScalingTests(unittest.TestCase):
    def test_equal_patient_weight_and_missing_placeholder(self):
        short = np.ones((38, 2), dtype=np.float32)
        long = np.ones((38, 20), dtype=np.float32) * 3
        mask = np.ones_like(short, dtype=bool)
        mask[0, 0] = False
        runs = [[(short, mask)], [(long, np.ones_like(long, dtype=bool))]]
        means, scales = fit_family_scaler(runs)
        np.testing.assert_allclose(means, 2.0)
        np.testing.assert_allclose(scales, 1.0)
        scaled = scale_patient_runs(runs, means, scales)
        self.assertEqual(float(scaled[0][0][0][0, 0]), 0.0)
        self.assertFalse(scaled[0][0][1][0, 0])


if __name__ == '__main__':
    unittest.main()
