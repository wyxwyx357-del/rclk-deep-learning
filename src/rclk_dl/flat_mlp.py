"""Fixed-length R/C/L MLP control for the completed 260-patient TCN run."""

import copy

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from .baselines import static_features
from .batches import RCL_FAMILIES, rcl_runs
from .inputs import model_channels
from .scaling import fit_family_scaler


# Fixed from the accepted 295-patient 35 Hz input range, before this comparison.
MAX_FRAMES = 2784
CHANNELS = 38
WIDTH = 4
WEIGHT_DECAY = 0.01
INPUT_FEATURES = MAX_FRAMES * CHANNELS * 2


def flatten_record(record, mean, scale):
    """Time-major signed values plus validity; no stretching or truncation."""
    values, valid = model_channels(record)
    if values.shape[0] > MAX_FRAMES:
        raise ValueError('sequence exceeds frozen maximum; do not truncate')
    usable = np.logical_and.reduce([valid[:, family].any(axis=1)
                                    for family in RCL_FAMILIES])
    valid = valid & usable[:, None]
    if not valid.any():
        raise ValueError('patient has no usable R/C/L time point')
    prepared = np.zeros((MAX_FRAMES, CHANNELS * 2), dtype=np.float32)
    length = len(values)
    prepared[:length, :CHANNELS] = np.where(valid,
        (values - mean[None, :]) / scale[None, :], 0)
    prepared[:length, CHANNELS:] = valid.astype(np.float32)
    return prepared.reshape(INPUT_FEATURES)


def prepare_fold(study, train_ids, other_ids):
    """Fit static and sequence transformations on training patients only."""
    static, _ = static_features(study, train_ids, [other_ids])
    mean, scale = fit_family_scaler([rcl_runs(study.record(i)) for i in train_ids])
    ids = list(train_ids) + list(other_ids)
    flat = np.stack([flatten_record(study.record(i), mean, scale) for i in ids])
    return (flat[:len(train_ids)], flat[len(train_ids):],
            static[0], static[1], mean.tolist(), scale.tolist())


class FlatMLP(nn.Module):
    def __init__(self, static_features):
        super().__init__()
        self.hidden = nn.Linear(INPUT_FEATURES, WIDTH)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(WIDTH + static_features, 1)

    def forward(self, flat, static):
        hidden = self.dropout(F.relu(self.hidden(flat)))
        return self.head(torch.cat((hidden, static), dim=1)).squeeze(1)


def fit_mlp(prepared, train_labels, eval_labels, seed, device, fixed_epochs=None):
    train_x, eval_x, train_static, eval_static, _, _ = prepared
    if fixed_epochs is not None and not 1 <= fixed_epochs <= 100:
        raise ValueError('invalid fixed epoch count')
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = FlatMLP(train_static.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=WEIGHT_DECAY)
    arrays = [torch.from_numpy(x) for x in
              (train_x, eval_x, train_static, eval_static)]
    train_x, eval_x, train_static, eval_static = arrays
    train_y = torch.as_tensor(train_labels, dtype=torch.float32)
    eval_y = (torch.as_tensor(eval_labels, dtype=torch.float32)
              if fixed_epochs is None else None)
    rng = np.random.default_rng(seed)
    lowest_loss, patience_reference, stale = np.inf, np.inf, 0
    best_state, best_epoch = None, None
    for epoch in range(1, (fixed_epochs or 100) + 1):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 16):
            indices = order[start:start + 16]
            batch = torch.as_tensor(indices, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(train_x[batch].to(device), train_static[batch].to(device)),
                train_y[batch].to(device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite MLP training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is None:
            model.eval()
            with torch.no_grad():
                losses = []
                for start in range(0, len(eval_x), 16):
                    end = start + 16
                    logits = model(eval_x[start:end].to(device),
                                   eval_static[start:end].to(device))
                    losses.append(float(F.binary_cross_entropy_with_logits(
                        logits, eval_y[start:end].to(device), reduction='sum')))
            validation_loss = sum(losses) / len(eval_x)
            if not np.isfinite(validation_loss):
                raise ValueError('nonfinite MLP validation loss')
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
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(eval_x), 16):
            logits = model(eval_x[start:start + 16].to(device),
                           eval_static[start:start + 16].to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    return np.asarray(probabilities, dtype=np.float64), {
        'best_epoch': best_epoch if fixed_epochs is None else fixed_epochs,
        'epochs_run': epoch,
        'validation_log_loss': float(lowest_loss) if fixed_epochs is None else None,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'seed': seed,
    }


def mlp_fold(study, split, allowed_ids, device):
    train_ids, test_ids, inner = study.fold_ids(split, allowed_ids)
    inner_rows = []
    for inner_index, (inner_train, inner_val) in enumerate(inner):
        prepared = prepare_fold(study, inner_train, inner_val)
        seed = 20261022 + 1000 * split['repeat'] + 100 * split['fold'] + inner_index
        probability, details = fit_mlp(prepared,
            study.frame.loc[inner_train, 'outcome'].to_numpy(),
            study.frame.loc[inner_val, 'outcome'].to_numpy(), seed, device)
        inner_rows.append({'inner_fold': inner_index,
                           'auc': float(roc_auc_score(study.frame.loc[inner_val, 'outcome'],
                                                      probability)), **details})
    refit_epochs = max(1, int(np.floor(np.median([row['best_epoch']
                                                 for row in inner_rows]) + 0.5)))
    prepared = prepare_fold(study, train_ids, test_ids)
    predictions, refits = [], []
    for base in (20261022, 20261023, 20261024):
        seed = base + 1000 * split['repeat'] + 100 * split['fold']
        probability, details = fit_mlp(prepared,
            study.frame.loc[train_ids, 'outcome'].to_numpy(),
            None, seed, device, fixed_epochs=refit_epochs)
        predictions.append(probability)
        refits.append(details)
    return np.mean(predictions, axis=0), {
        'inner_fits': inner_rows, 'outer_refit_epochs': refit_epochs,
        'outer_refits': refits, 'sequence_mean': prepared[4],
        'sequence_scale': prepared[5], 'train_n': len(train_ids),
        'test_n': len(test_ids),
    }
