"""Preserve anatomical family gaps when preparing independent TCN runs."""

import numpy as np
import torch

from .inputs import model_channels


RCL_FAMILIES = (slice(0, 16), slice(16, 24), slice(24, 38))


def rcl_runs(record, include_k=False):
    """Return runs where every R/C/L family has at least one valid channel.

    A frame with a fully missing family breaks the convolution history. Values
    from that frame are not used by M3; their absence remains available to the
    separately specified quality control model.
    """
    values, valid = model_channels(record, include_k=include_k)
    usable = np.logical_and.reduce([valid[:, family].any(axis=1)
                                    for family in RCL_FAMILIES])
    changes = np.diff(np.r_[False, usable, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [(values[start:end].T.copy(), valid[start:end].T.copy())
            for start, end in zip(starts, ends)]


def pad_patient_runs(patient_runs, static):
    """Right-pad independent runs and retain their patient membership.

    ``patient_runs`` contains one list of (channel x time values, mask) pairs
    per patient. Inputs must already be standardized on the training fold.
    """
    static = np.asarray(static, dtype=np.float32)
    if static.ndim != 2 or static.shape[0] != len(patient_runs):
        raise ValueError('static rows must match patients')
    if not np.isfinite(static).all():
        raise ValueError('static features must be finite')
    lengths = [value.shape[1] for runs in patient_runs for value, _ in runs]
    if not lengths or any(not runs for runs in patient_runs):
        raise ValueError('every patient needs at least one run')
    channels = patient_runs[0][0][0].shape[0]
    if channels not in (38, 54):
        raise ValueError('expected 38 RCL or 54 RCLK channels')
    count, max_length = len(lengths), max(lengths)
    values = np.zeros((count, channels, max_length), dtype=np.float32)
    mask = np.zeros((count, channels, max_length), dtype=bool)
    time_mask = np.zeros((count, max_length), dtype=bool)
    patient_index = np.empty(count, dtype=np.int64)
    k = 0
    for patient, runs in enumerate(patient_runs):
        for value, valid in runs:
            if value.shape != valid.shape or value.shape[0] != channels or value.shape[1] == 0:
                raise ValueError('run must have matching [channels, time] arrays')
            size = value.shape[1]
            values[k, :, :size] = np.where(valid, value, 0)
            mask[k, :, :size] = valid
            time_mask[k, :size] = True
            patient_index[k] = patient
            k += 1
    return tuple(torch.from_numpy(x) for x in
                 (values, mask, time_mask, patient_index, static))
