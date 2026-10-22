"""Patient-level isolation and paired evaluation checks."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.evaluation import evaluate_oof
from rclk_dl.baselines import logistic_fold
from rclk_dl.study import CATEGORICAL, K_SHAPE, NUMERIC
from rclk_dl.training import fit_tcn, select_config


class ExperimentTests(unittest.TestCase):
    def test_outer_test_label_cannot_change_prediction(self):
        torch.set_num_threads(1)
        value = np.ones((38, 6), dtype=np.float32)
        mask = np.ones_like(value, dtype=bool)
        prepared = {'runs': {i: [(value * (j + 1), mask)]
                             for j, i in enumerate(('a', 'b', 'c'))},
                    'static': {i: np.array([j], dtype=np.float32)
                               for j, i in enumerate(('a', 'b', 'c'))},
                    'labels': {'a': 0, 'b': 1, 'c': 0},
                    'mean': [0.] * 38, 'static_feature_count': 1}
        first, _ = fit_tcn(prepared, ['a', 'b'], ['c'], (8, 0.001),
                           20261022, torch.device('cpu'), fixed_epochs=1)
        prepared['labels']['c'] = 1
        second, _ = fit_tcn(prepared, ['a', 'b'], ['c'], (8, 0.001),
                            20261022, torch.device('cpu'), fixed_epochs=1)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)

    def test_tie_rule_prefers_small_width_then_larger_decay(self):
        rows = [{'width': width, 'weight_decay': decay, 'auc': score}
                for width, decay, score in ((8, .001, .701), (8, .01, .700),
                                            (16, .001, .704), (16, .01, .703))]
        chosen, _ = select_config(rows)
        self.assertEqual(chosen, (8, .01))

    def test_paired_bootstrap_requires_one_oof_per_patient_repeat(self):
        frame = pd.DataFrame({'outcome': [0, 0, 1, 1]}, index=list('abcd'))
        rows = [{'audit_id': i, 'outcome': int(frame.loc[i, 'outcome']),
                 'repeat': r, 'model': m, 'probability': float(p)}
                for r in range(3) for m, values in
                (('M1', [.1, .3, .6, .8]), ('M3', [.1, .2, .7, .9]))
                for i, p in zip('abcd', values)]
        result = evaluate_oof(rows, frame, set('abcd'), ('M1', 'M3'), (('M3', 'M1'),))
        self.assertEqual(result['patient_n'], 4)
        self.assertEqual(result['bootstrap_draws'], 10000)
        with self.assertRaises(ValueError):
            evaluate_oof(rows[:-1], frame, set('abcd'), ('M1', 'M3'), (('M3', 'M1'),))

    def test_logistic_grid_uses_patient_inner_folds(self):
        ids = [f'synthetic_{j}' for j in range(30)]
        frame = pd.DataFrame(index=ids)
        for name in NUMERIC + K_SHAPE:
            frame[name] = np.linspace(0, 1, 30)
        for name in CATEGORICAL:
            frame[name] = 'synthetic_category'
        frame['outcome'] = np.arange(30) % 2
        fake_study = type('FakeStudy', (), {'frame': frame})()
        train, test = ids[:20], ids[20:]
        inner = [(train[:4 * j] + train[4 * j + 4:], train[4 * j:4 * j + 4])
                 for j in range(5)]
        probability, details = logistic_fold(fake_study, 'M1', train, test, inner)
        self.assertEqual(len(probability), len(test))
        self.assertTrue(np.isfinite(probability).all())
        self.assertEqual(len(details['grid_scores']), 16)


if __name__ == '__main__':
    unittest.main()
