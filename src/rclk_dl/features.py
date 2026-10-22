"""Frozen simple temporal statistics and quality covariates."""

from pathlib import Path

import numpy as np

from .inputs import _one


GROUPS = (('R_anterior', 'R', 0), ('R_posterior', 'R', 1),
          ('C', 'C', 0), ('L_anterior', 'L', 0), ('L_posterior', 'L', 1))
M2_STAT_NAMES = ('abs_median', 'abs_p95', 'positive_fraction', 'lag_1s_corr',
                 'position_abs_median_iqr', 'neighbor_corr_median')
M2_NAMES = tuple(f'{group}_{stat}' for group, _, _ in GROUPS for stat in M2_STAT_NAMES)
MQ_NAMES = ('video_duration_s', 'fps', 'source_positions', 'R_invalid_fraction',
            'C_invalid_fraction', 'L_invalid_fraction', 'translation_speed_median_px_s',
            'rotation_speed_abs_median_deg_s', 'fb_error_p95_time_median_px',
            'pcc_time_median', 'qc_grade2_frame_fraction', 'grid_interpolation_fraction')


def _median_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if values.size else np.nan


def _corr(x, y, minimum_pairs=70):
    if len(x) < minimum_pairs or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def simple_temporal_features(record):
    """Thirty prespecified M2 features from signed 35 Hz R/C/L arrays."""
    result = []
    lag = 35
    for _, name, wall in GROUPS:
        x = record['values'][name][:, wall, :]
        valid = record['valid'][name][:, wall, :]
        observed = x[valid].astype(np.float64)
        if observed.size:
            magnitude = np.abs(observed)
            first = [float(np.median(magnitude)), float(np.percentile(magnitude, 95)),
                     float(np.mean(observed > 0))]
        else:
            first = [np.nan] * 3
        lag_corr = []
        medians = []
        neighbor_corr = []
        for j in range(x.shape[1]):
            one = x[:, j]
            good = valid[:, j]
            if good.any():
                medians.append(float(np.median(np.abs(one[good]))))
            if len(one) > lag:
                # A pair counts only when every intervening 35 Hz sample is valid.
                cumulative = np.r_[0, np.cumsum(good.astype(np.int32))]
                uninterrupted = cumulative[lag + 1:] - cumulative[:-(lag + 1)] == lag + 1
                lag_corr.append(_corr(one[:-lag][uninterrupted], one[lag:][uninterrupted]))
            if j + 1 < x.shape[1]:
                pair = good & valid[:, j + 1]
                neighbor_corr.append(_corr(one[pair], x[:, j + 1][pair]))
        medians = np.asarray(medians)
        iqr = float(np.percentile(medians, 75) - np.percentile(medians, 25)) if len(medians) >= 2 else np.nan
        result.extend(first + [_median_or_nan(np.asarray(lag_corr)[np.isfinite(lag_corr)]),
                               iqr, _median_or_nan(np.asarray(neighbor_corr)[np.isfinite(neighbor_corr)])])
    assert len(result) == len(M2_NAMES)
    return dict(zip(M2_NAMES, result))


def quality_features(record, root):
    """Twelve frozen MQ variables; internal motion is not probe-motion truth."""
    case_id = record['case_id']
    folder = Path(root) / case_id / '02_追踪质控与形变' / '07_artifact_qc'
    with np.load(_one(folder, '*_artifact_qc_v2_1.npz'), allow_pickle=False) as qc:
        fields = {key: qc[key] for key in ('global_translation_step_px',
                  'global_rotation_step_deg', 'pair_fb_error_p95_px',
                  'pair_mean_pcc', 'artifact_grade')}
    frames, fps = record['native_frames'], record['fps']
    if any(len(value) != frames for value in fields.values()):
        raise ValueError(f'QC frame count mismatch for {case_id}')
    def finite_median(array):
        return _median_or_nan(array[np.isfinite(array)])
    values = [frames / fps, fps, record['native_positions']]
    values.extend(1.0 - float(record['valid'][name].mean()) for name in ('R', 'C', 'L'))
    values.extend([
        finite_median(fields['global_translation_step_px'] * fps),
        finite_median(np.abs(fields['global_rotation_step_deg']) * fps),
        finite_median(fields['pair_fb_error_p95_px']),
        finite_median(fields['pair_mean_pcc']),
        float(np.mean(fields['artifact_grade'] == 2)),
        float(np.mean(~record['time_observed'])),
    ])
    assert len(values) == len(MQ_NAMES)
    return dict(zip(MQ_NAMES, values))
