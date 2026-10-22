"""Frozen nested-CV fitting for RCL/RCLK causal TCNs and controls."""

import copy
import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from .baselines import static_features
from .batches import pad_patient_runs, rcl_runs
from .scaling import fit_family_scaler, scale_patient_runs
from .tcn import PatientTCN


CONFIGS = tuple((width, decay) for width in (8, 16) for decay in (0.001, 0.01))
SPATIAL_GROUPS = (slice(0, 8), slice(8, 16), slice(16, 24),
                  slice(24, 31), slice(31, 38))


def _control_runs(runs, control, audit_id):
    if control == 'none':
        return runs
    changed = []
    for index, (value, valid) in enumerate(runs):
        value, valid = value.copy(), valid.copy()
        if control == 'shuffle':
            digest = hashlib.sha256(f'20261022:{audit_id}:{index}'.encode()).digest()
            permutation = np.random.default_rng(int.from_bytes(digest[:8], 'little')).permutation(value.shape[1])
            value, valid = value[:, permutation], valid[:, permutation]
        elif control == 'spatial_mean':
            for group in SPATIAL_GROUPS:
                count = valid[group].sum(axis=0)
                mean = np.divide(np.where(valid[group], value[group], 0).sum(axis=0), count,
                                 out=np.zeros(value.shape[1], dtype=np.float32), where=count > 0)
                value[group] = np.where(valid[group], mean[None, :], np.nan)
        elif control == 'mask_quality':
            value[:] = 0
        else:
            raise ValueError(f'unknown control {control}')
        changed.append((value, valid))
    return changed


class SequenceBank:
    def __init__(self, study, include_k=False, static_k=False, control='none'):
        if control != 'none' and include_k:
            raise ValueError('mechanism controls are specified for primary RCL only')
        self.study, self.include_k, self.static_k, self.control = study, include_k, static_k, control
        self._runs = {}

    def get(self, audit_id):
        if audit_id not in self._runs:
            record = self.study.record(audit_id, include_k=self.include_k)
            runs = rcl_runs(record, include_k=self.include_k)
            if not runs:
                raise ValueError(f'no usable RCL run for {audit_id}')
            self._runs[audit_id] = _control_runs(runs, self.control, audit_id)
        return self._runs[audit_id]


def prepare_fold(bank, train_ids, other_ids):
    """Fit both transforms on training patients, then transform held-out IDs."""
    study = bank.study
    include_quality = bank.control == 'mask_quality'
    arrays, prep = static_features(study, train_ids, [other_ids],
                                   include_k=bank.static_k,
                                   include_quality=include_quality)
    train_runs = [bank.get(i) for i in train_ids]
    mean, scale = fit_family_scaler(train_runs)
    ids = list(train_ids) + list(other_ids)
    runs = {i: scaled for i, scaled in zip(ids, scale_patient_runs(
        train_runs + [bank.get(i) for i in other_ids], mean, scale))}
    static = {i: vector for i, vector in zip(ids, np.concatenate(arrays, axis=0))}
    return {'runs': runs, 'static': static, 'labels': study.frame.outcome.to_dict(),
            'mean': mean.tolist(), 'scale': scale.tolist(),
            'static_feature_count': arrays[0].shape[1],
            'static_numeric_center': prep.named_transformers_['numeric'].named_steps['scale'].mean_.tolist(),
            'static_numeric_scale': prep.named_transformers_['numeric'].named_steps['scale'].scale_.tolist()}


def _batch(prepared, ids, device):
    arrays = pad_patient_runs([prepared['runs'][i] for i in ids],
                              np.stack([prepared['static'][i] for i in ids]))
    return tuple(x.to(device) for x in arrays)


def _predict(model, prepared, ids, device, batch_patients=16):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_patients):
            group = ids[start:start + batch_patients]
            probabilities.extend(torch.sigmoid(model(*_batch(prepared, group, device))).cpu().numpy())
    return np.asarray(probabilities, dtype=np.float64)


def _validation_loss(model, prepared, ids, device, batch_patients=16):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_patients):
            group = ids[start:start + batch_patients]
            labels = torch.as_tensor([prepared['labels'][i] for i in group],
                                     dtype=torch.float32, device=device)
            total += float(F.binary_cross_entropy_with_logits(
                model(*_batch(prepared, group, device)), labels, reduction='sum'))
    return total / len(ids)


def fit_tcn(prepared, train_ids, eval_ids, config, seed, device,
            fixed_epochs=None):
    """Inner fits use early stopping; outer fits use fixed selected epochs."""
    width, weight_decay = config
    if config not in CONFIGS:
        raise ValueError('configuration outside frozen four-group grid')
    if fixed_epochs is not None and not 1 <= fixed_epochs <= 100:
        raise ValueError('invalid fixed epoch count')
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = PatientTCN(static_features=prepared['static_feature_count'],
                       width=width, measurement_channels=len(prepared['mean'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    lowest_loss, patience_reference, stale = np.inf, np.inf, 0
    best_state, best_epoch = None, None
    epochs = fixed_epochs or 100
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_ids))
        for start in range(0, len(order), 16):
            group = [train_ids[j] for j in order[start:start + 16]]
            label = torch.as_tensor([prepared['labels'][i] for i in group],
                                    dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(*_batch(prepared, group, device)), label)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite training loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError('nonfinite TCN gradient')
            optimizer.step()
        if fixed_epochs is None:
            validation_loss = _validation_loss(model, prepared, eval_ids, device)
            if not np.isfinite(validation_loss):
                raise ValueError('nonfinite validation loss')
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
    else:
        best_epoch = fixed_epochs
    probability = _predict(model, prepared, eval_ids, device)
    return probability, {'best_epoch': best_epoch, 'validation_log_loss':
                         float(lowest_loss) if fixed_epochs is None else None,
                         'parameter_count': sum(p.numel() for p in model.parameters()),
                         'seed': seed, 'epochs_run': epoch}


def select_config(grid_rows):
    scores = {(width, decay): np.mean([row['auc'] for row in grid_rows
                                      if (row['width'], row['weight_decay']) == (width, decay)])
              for width, decay in CONFIGS}
    maximum = max(scores.values())
    eligible = [config for config in CONFIGS if maximum - scores[config] <= 0.005]
    return sorted(eligible, key=lambda c: (c[0], -c[1]))[0], scores


def tcn_fold(study, split, allowed_ids, device, include_k=False,
             static_k=False,
             control='none'):
    train_ids, test_ids, inner = study.fold_ids(split, allowed_ids)
    bank = SequenceBank(study, include_k=include_k, static_k=static_k,
                        control=control)
    grid_rows = []
    for inner_index, (inner_train, inner_val) in enumerate(inner):
        prepared = prepare_fold(bank, inner_train, inner_val)
        for config in CONFIGS:
            seed = 20261022 + 1000 * split['repeat'] + 100 * split['fold'] + inner_index
            probabilities, details = fit_tcn(prepared, inner_train, inner_val,
                                             config, seed, device)
            grid_rows.append({'inner_fold': inner_index, 'width': config[0],
                              'weight_decay': config[1],
                              'auc': float(roc_auc_score(study.frame.loc[inner_val, 'outcome'],
                                                         probabilities)), **details})
    chosen, mean_scores = select_config(grid_rows)
    best_epochs = [row['best_epoch'] for row in grid_rows
                   if (row['width'], row['weight_decay']) == chosen]
    refit_epochs = max(1, int(np.floor(np.median(best_epochs) + 0.5)))
    prepared = prepare_fold(bank, train_ids, test_ids)
    predictions, refits = [], []
    for base in (20261022, 20261023, 20261024):
        seed = base + 1000 * split['repeat'] + 100 * split['fold']
        probability, details = fit_tcn(prepared, train_ids, test_ids,
                                       chosen, seed, device, fixed_epochs=refit_epochs)
        predictions.append(probability)
        refits.append(details)
    probability = np.mean(predictions, axis=0)
    return probability, {
        'chosen_width': chosen[0], 'chosen_weight_decay': chosen[1],
        'mean_inner_auc_by_config': [
            {'width': key[0], 'weight_decay': key[1], 'mean_auc': float(value)}
            for key, value in mean_scores.items()],
        'inner_fits': grid_rows, 'outer_refit_epochs': refit_epochs,
        'outer_refits': refits,
        'sequence_mean': prepared['mean'], 'sequence_scale': prepared['scale'],
        'static_numeric_center': prepared['static_numeric_center'],
        'static_numeric_scale': prepared['static_numeric_scale'],
        'train_n': len(train_ids), 'test_n': len(test_ids),
    }
