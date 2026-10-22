"""Validate the frozen 295-patient inputs without fitting a model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rclk_dl.inputs import TARGET_HZ, model_channels, read_case


FEATURES = {
    'R': 'rsr_abs_median',
    'C': 'cavity_width_strain_rate_abs_median',
    'L': 'longitudinal_wall_strain_rate_abs_median',
    'K': 'wall_curvature_change_rate_mm_inv_s_abs_median',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Final patient-output root')
    parser.add_argument('--cohort', type=Path, required=True, help='Frozen 295-patient input CSV')
    parser.add_argument('--formal', type=Path, required=True, help='Frozen formal F01-F20 CSV')
    parser.add_argument('--out', type=Path, default=Path('outputs/input_acceptance.csv'))
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort, usecols=['audit_id', 'numeric_id'])
    formal = pd.read_csv(args.formal)
    formal['numeric_id'] = formal.case_id.str.extract(r'_(\d+)_')[0].astype(int)
    patients = cohort.merge(formal, on='numeric_id', validate='one_to_one')
    assert len(patients) == 295 and patients.audit_id.is_unique
    rows = []
    for row in patients.to_dict('records'):
        have_k = pd.notna(row[FEATURES['K']])
        record = read_case(row['case_id'], args.root, include_k=have_k)
        t = record['time_s']
        assert t[0] == 0 and np.all(np.diff(t) > 0)
        assert np.allclose(t, np.arange(len(t)) / TARGET_HZ, atol=1e-9)
        assert t[-1] <= (record['native_frames'] - 1) / record['fps'] + 1e-9
        left, right, risk = record['time_left_frame'], record['time_right_frame'], record['topology_risk']
        assert np.all(right - left <= 1) and np.all(left <= right)
        bad = risk[left] | risk[right]
        values, valid = model_channels(record, include_k=have_k)
        assert np.array_equal(np.isfinite(values), valid)
        assert not valid[bad].any(), row['case_id']
        for name, field in FEATURES.items():
            if name in record['values']:
                assert abs(record['formal_medians'][name] - float(row[field])) < 4e-9
                assert not (record['observed'][name] & ~record['valid'][name]).any()
                assert not record['observed'][name][~record['time_observed']].any()
        rows.append({'audit_id': row['audit_id'], 'frames_original': record['native_frames'],
                     'frames_grid': len(t), 'fps': record['fps'], 'positions': record['native_positions'],
                     'K_available': bool(have_k), 'topology_frames': int(risk.sum()),
                     'interpolated_time_fraction': float((~record['time_observed']).mean())})
    result = pd.DataFrame(rows)
    assert result.K_available.sum() == 288
    assert (result.topology_frames > 0).sum() == 14 and result.topology_frames.sum() == 1171
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(json.dumps({'status': 'passed', 'patients': len(result), 'K': int(result.K_available.sum()),
                      'topology_frames': int(result.topology_frames.sum()),
                      'median_interpolated_time_fraction': float(result.interpolated_time_fraction.median()),
                      'output': str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

