"""Boundary checks for the two interpolation steps."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.inputs import _spatial, _temporal, _timeline


class InputInterpolationTests(unittest.TestCase):
    def test_spatial_does_not_bridge_missing_neighbor(self):
        values = np.array([[[1.0, np.nan, 3.0]]], dtype=np.float32)
        out, seen, _ = _spatial(values, [0, 0.5, 1], [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_allclose(out[0, 0, [0, 4]], [1, 3])
        self.assertTrue(np.isnan(out[0, 0, 1:4]).all())
        self.assertTrue(seen[0, 0, 0] and seen[0, 0, 4])

    def test_temporal_does_not_bridge_masked_frame(self):
        source = np.array([[[1.0]], [[np.nan]], [[3.0]]], dtype=np.float32)
        seen = np.isfinite(source)
        _, left, right, weight, exact = _timeline(3, 24)
        out, valid, observed = _temporal(source, seen, left, right, weight, exact)
        self.assertFalse(valid[(left == 1) | (right == 1)].any())
        self.assertFalse(observed[~exact].any())
        self.assertEqual(float(out[0, 0, 0]), 1.0)


if __name__ == '__main__':
    unittest.main()

