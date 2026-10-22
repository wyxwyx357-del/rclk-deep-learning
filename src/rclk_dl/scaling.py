"""Training-patient-only scaling for signed R/C/L sequence values."""

import numpy as np


FAMILIES = {'R': slice(0, 16), 'C': slice(16, 24), 'L': slice(24, 38)}
K_FAMILY = slice(38, 54)


def fit_family_scaler(patient_runs):
    """Give every training patient equal weight in each family's moments."""
    if not patient_runs:
        raise ValueError('no training patients')
    channels = patient_runs[0][0][0].shape[0]
    if channels not in (38, 54):
        raise ValueError('expected 38 RCL or 54 RCLK channels')
    means, scales = np.zeros(channels, dtype=np.float32), np.ones(channels, dtype=np.float32)
    families = (*FAMILIES.values(), K_FAMILY) if channels == 54 else tuple(FAMILIES.values())
    for family in families:
        moments = []
        for runs in patient_runs:
            selected = [value[family][valid[family]] for value, valid in runs]
            observed = np.concatenate(selected) if selected else np.empty(0)
            if observed.size == 0 or not np.isfinite(observed).all():
                raise ValueError('each training patient needs finite family observations')
            observed = observed.astype(np.float64)
            moments.append((observed.mean(), np.mean(observed ** 2)))
        center = float(np.mean([m[0] for m in moments]))
        variance = max(0.0, float(np.mean([m[1] for m in moments])) - center ** 2)
        scale = max(variance ** 0.5, 1e-8)
        means[family], scales[family] = center, scale
    return means, scales


def scale_patient_runs(patient_runs, means, scales):
    """Use fitted training moments; missing values remain masked."""
    return [[(np.where(valid, (value - means[:, None]) / scales[:, None], 0)
              .astype(np.float32), valid)
             for value, valid in runs] for runs in patient_runs]
