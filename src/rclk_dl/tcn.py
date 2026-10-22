"""Small causal TCN for independent R/C/L runs and patient-level pooling.

All scaling and static-feature preprocessing must be fitted on training patients.
One row in the sequence batch is one continuous run, never a whole patient
with missing intervals silently replaced by zeros.
"""

import torch
from torch import nn
from torch.nn import functional as F


DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


class CausalConv1d(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout=0.2):
        super().__init__()
        self.first = CausalConv1d(width, dilation)
        self.second = CausalConv1d(width, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_mask):
        y = self.dropout(F.relu(self.first(x)))
        y = self.dropout(F.relu(self.second(y)))
        return F.relu(x + y) * time_mask[:, None, :]


class PatientTCN(nn.Module):
    """Sequence values and validity masks have shape [runs, 38, time].

    ``time_mask`` marks real samples rather than right padding. ``patient_index``
    maps each independent run to a row of ``static``. Each patient needs at
    least one run. The output is one uncalibrated logit per patient.
    """

    def __init__(self, static_features, width=8, dropout=0.2, measurement_channels=38):
        super().__init__()
        if width not in (8, 16):
            raise ValueError('Frozen protocol only allows widths 8 and 16')
        if static_features < 0:
            raise ValueError('static_features must be nonnegative')
        if measurement_channels not in (38, 54):
            raise ValueError('expected 38 RCL or 54 RCLK channels')
        self.measurement_channels = measurement_channels
        self.projection = nn.Conv1d(2 * measurement_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, dilation, dropout) for dilation in DILATIONS)
        self.head = nn.Linear(2 * width + static_features, 1)

    def forward(self, values, measurement_mask, time_mask, patient_index, static):
        if values.ndim != 3 or values.shape[1] != self.measurement_channels:
            raise ValueError('values must have shape [runs, channels, time]')
        if (measurement_mask.shape != values.shape or
                time_mask.shape != (values.shape[0], values.shape[2])):
            raise ValueError('mask shapes do not match values')
        if measurement_mask.dtype != torch.bool or time_mask.dtype != torch.bool:
            raise ValueError('measurement_mask and time_mask must be boolean')
        if static.ndim != 2 or patient_index.shape != (values.shape[0],):
            raise ValueError('static or patient_index has the wrong shape')
        if static.shape[1] != self.head.in_features - 2 * self.projection.out_channels:
            raise ValueError('static feature count does not match the model')
        if values.shape[0] == 0 or static.shape[0] == 0:
            raise ValueError('every batch needs patients and runs')
        if not torch.isfinite(values[measurement_mask]).all() or not torch.isfinite(static).all():
            raise ValueError('observed values and prepared static features must be finite')
        if (measurement_mask & ~time_mask[:, None, :]).any():
            raise ValueError('padding cannot contain valid measurements')
        if (patient_index < 0).any() or (patient_index >= static.shape[0]).any():
            raise ValueError('patient_index is out of bounds')
        if not torch.all(torch.bincount(patient_index, minlength=static.shape[0]) > 0):
            raise ValueError('each patient must have at least one run')

        x = torch.cat((torch.where(measurement_mask, values, 0.0),
                       measurement_mask.to(values.dtype)), dim=1)
        x = self.projection(x) * time_mask[:, None, :]
        for block in self.blocks:
            x = block(x, time_mask)

        counts = time_mask.sum(dim=1)
        if not torch.all(counts > 0):
            raise ValueError('each run must have at least one real sample')
        patient_count = static.shape[0]
        sums = x.new_zeros((patient_count, x.shape[1]))
        sums.index_add_(0, patient_index, (x * time_mask[:, None, :]).sum(dim=2))
        patient_lengths = x.new_zeros(patient_count)
        patient_lengths.index_add_(0, patient_index, counts.to(x.dtype))
        means = sums / patient_lengths[:, None]

        run_max = x.masked_fill(~time_mask[:, None, :], -torch.inf).amax(dim=2)
        maxima = torch.stack([
            run_max[patient_index == i].amax(dim=0) for i in range(patient_count)
        ])
        return self.head(torch.cat((means, maxima, static), dim=1)).squeeze(1)
