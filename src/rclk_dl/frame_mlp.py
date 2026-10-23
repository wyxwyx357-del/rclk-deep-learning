"""Order-invariant, framewise R/C/L control for the 260-patient TCN run."""

import copy

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from .training import SequenceBank, _batch, prepare_fold


CHANNELS = 38
WIDTH = 8
WEIGHT_DECAY = 0.01


class PatientFrameMLP(nn.Module):
    """Apply one 76-to-8 map to every usable frame, then pool per patient."""

    def __init__(self, static_features):
        super().__init__()
        self.frame = nn.Linear(2 * CHANNELS, WIDTH)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(2 * WIDTH + static_features, 1)

    def forward(self, values, measurement_mask, time_mask, patient_index, static):
        if values.ndim != 3 or values.shape[1] != CHANNELS:
            raise ValueError('values must have shape [runs, 38, time]')
        if measurement_mask.shape != values.shape or time_mask.shape != (values.shape[0], values.shape[2]):
            raise ValueError('mask shapes do not match values')
        if measurement_mask.dtype != torch.bool or time_mask.dtype != torch.bool:
            raise ValueError('masks must be boolean')
        if (measurement_mask & ~time_mask[:, None, :]).any():
            raise ValueError('padding cannot contain valid measurements')
        if static.ndim != 2 or static.shape[1] != self.head.in_features - 2 * WIDTH:
            raise ValueError('static features do not match the model')
        if patient_index.shape != (values.shape[0],) or static.shape[0] == 0:
            raise ValueError('patient_index or static rows do not match runs')
        if (patient_index < 0).any() or (patient_index >= static.shape[0]).any():
            raise ValueError('patient_index is out of bounds')
        if not torch.all(torch.bincount(patient_index, minlength=static.shape[0]) > 0):
            raise ValueError('each patient needs a run')
        if not torch.all(time_mask.any(dim=1)):
            raise ValueError('each run needs a usable frame')

        frame_input = torch.cat((torch.where(measurement_mask, values, 0.0),
                                 measurement_mask.to(values.dtype)), dim=1).transpose(1, 2)
        features = self.dropout(F.relu(self.frame(frame_input)))
        real = time_mask.unsqueeze(2)
        patient_count = static.shape[0]
        sums = features.new_zeros((patient_count, WIDTH))
        sums.index_add_(0, patient_index, (features * real).sum(dim=1))
        counts = features.new_zeros(patient_count)
        counts.index_add_(0, patient_index, time_mask.sum(dim=1).to(features.dtype))
        means = sums / counts[:, None]
        run_max = features.masked_fill(~real, -torch.inf).amax(dim=1)
        maxima = torch.stack([run_max[patient_index == i].amax(dim=0)
                              for i in range(patient_count)])
        return self.head(torch.cat((means, maxima, static), dim=1)).squeeze(1)


def _predict(model, prepared, ids, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            group = ids[start:start + 16]
            probabilities.extend(torch.sigmoid(model(*_batch(prepared, group, device))).cpu().numpy())
    return np.asarray(probabilities, dtype=np.float64)


def fit_frame_mlp(prepared, train_ids, eval_ids, seed, device, fixed_epochs=None):
    if fixed_epochs is not None and not 1 <= fixed_epochs <= 100:
        raise ValueError('invalid fixed epoch count')
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = PatientFrameMLP(prepared['static_feature_count']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    lowest_loss, patience_reference, stale = np.inf, np.inf, 0
    best_state, best_epoch = None, None
    for epoch in range(1, (fixed_epochs or 100) + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        for start in range(0, len(order), 16):
            group = [train_ids[j] for j in order[start:start + 16]]
            labels = torch.as_tensor([prepared['labels'][i] for i in group],
                                     dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(*_batch(prepared, group, device)), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite frame MLP training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for start in range(0, len(eval_ids), 16):
                    group = eval_ids[start:start + 16]
                    labels = torch.as_tensor([prepared['labels'][i] for i in group],
                                             dtype=torch.float32, device=device)
                    total += float(F.binary_cross_entropy_with_logits(
                        model(*_batch(prepared, group, device)), labels, reduction='sum'))
            validation_loss = total / len(eval_ids)
            if not np.isfinite(validation_loss):
                raise ValueError('nonfinite frame MLP validation loss')
            if validation_loss < lowest_loss:
                lowest_loss, best_epoch = validation_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
            if validation_loss < patience_reference - 0.0001:
                patience_reference, stale = validation_loss, 0
            else:
                stale += 1
            if stale >= 12:
                break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    return _predict(model, prepared, eval_ids, device), {
        'best_epoch': best_epoch if fixed_epochs is None else fixed_epochs,
        'epochs_run': epoch,
        'validation_log_loss': float(lowest_loss) if fixed_epochs is None else None,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'seed': seed,
    }


def frame_mlp_fold(study, split, allowed_ids, device):
    train_ids, test_ids, inner = study.fold_ids(split, allowed_ids)
    bank = SequenceBank(study)
    inner_rows = []
    for inner_index, (inner_train, inner_val) in enumerate(inner):
        prepared = prepare_fold(bank, inner_train, inner_val)
        seed = 20261022 + 1000 * split['repeat'] + 100 * split['fold'] + inner_index
        probability, details = fit_frame_mlp(prepared, inner_train, inner_val, seed, device)
        inner_rows.append({'inner_fold': inner_index,
                           'auc': float(roc_auc_score(study.frame.loc[inner_val, 'outcome'],
                                                      probability)), **details})
    refit_epochs = max(1, int(np.floor(np.median([row['best_epoch']
                                                 for row in inner_rows]) + 0.5)))
    prepared = prepare_fold(bank, train_ids, test_ids)
    predictions, refits = [], []
    for base in (20261022, 20261023, 20261024):
        seed = base + 1000 * split['repeat'] + 100 * split['fold']
        probability, details = fit_frame_mlp(prepared, train_ids, test_ids,
                                             seed, device, fixed_epochs=refit_epochs)
        predictions.append(probability)
        refits.append(details)
    return np.mean(predictions, axis=0), {
        'inner_fits': inner_rows, 'outer_refit_epochs': refit_epochs,
        'outer_refits': refits, 'sequence_mean': prepared['mean'],
        'sequence_scale': prepared['scale'], 'train_n': len(train_ids),
        'test_n': len(test_ids),
    }

