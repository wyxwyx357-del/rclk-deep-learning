"""Boundary checks for the two interpolation steps."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.inputs import _spatial, _temporal, _timeline, read_case


class InputInterpolationTests(unittest.TestCase):
    def test_integer_topology_flags_are_rejected_before_indexing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            core = Path(folder) / 'case' / '02_追踪质控与形变'
            radial = core / '10_rsr_fourway'
            anatomy = core / '11_anatomical_deformation'
            evidence = core / '05_radial_tracking_corrected'
            for path in (radial, anatomy, evidence):
                path.mkdir(parents=True)
            x = np.linspace(0, 1, 8)
            np.savez(radial / 'case_rsr_fourway_v1.npz',
                     current_qc_radial_strain_rate_s=np.ones((3, 2, 8)),
                     normalized_cervix_to_fundus_position=x, fps=35,
                     formal_qc_lineage_id='test')
            np.savez(anatomy / 'case_anatomical_deformation_v1.npz',
                     qc_cavity_width_strain_rate_s=np.ones((3, 8)),
                     qc_longitudinal_wall_strain_rate_s=np.ones((3, 2, 7)),
                     normalized_cervix_to_fundus_position=x, fps=35,
                     formal_qc_lineage_id='test')
            np.savez(evidence / 'case_p3_image_boundary_evidence_v2.npz',
                     p3_topology_risk_frame=np.array([0, 1, 0], dtype=np.uint8))
            with self.assertRaisesRegex(TypeError, 'must be boolean'):
                read_case('case', folder)

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

