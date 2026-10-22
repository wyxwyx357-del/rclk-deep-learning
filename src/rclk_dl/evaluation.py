"""Patient-paired OOF summaries and conditional stratified bootstrap."""

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def weighted_auc(y, prediction, weights):
    order = np.argsort(prediction, kind='mergesort')
    ranked = prediction[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    below = np.cumsum(negative, axis=1) - negative
    denominator = weights[:, y == 1].sum(axis=1) * weights[:, y == 0].sum(axis=1)
    return np.sum(positive * (below + 0.5 * negative), axis=1) / denominator


def evaluate_oof(rows, frame, allowed_ids, models, comparisons):
    """Each model must predict every allowed patient once per repeat."""
    ids = sorted(allowed_ids)
    labels = frame.loc[ids, 'outcome'].to_numpy(dtype=np.int8)
    by_key = {}
    for row in rows:
        key = (row['model'], row['repeat'], row['audit_id'])
        if key in by_key or row['audit_id'] not in allowed_ids:
            raise ValueError('duplicate or unexpected OOF patient prediction')
        if int(row['outcome']) != int(frame.loc[row['audit_id'], 'outcome']):
            raise ValueError('OOF outcome differs from frozen patient label')
        by_key[key] = float(row['probability'])
    probabilities = {}
    repeated = []
    calibration = []
    for model in models:
        for repeat in range(3):
            try:
                p = np.asarray([by_key[(model, repeat, i)] for i in ids])
            except KeyError as error:
                raise ValueError(f'incomplete OOF predictions: {error}') from error
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError('invalid OOF probability')
            probabilities[model, repeat] = p
            repeated.append({'model': model, 'repeat': repeat, 'n': len(ids),
                             'auc': float(roc_auc_score(labels, p)),
                             'brier': float(brier_score_loss(labels, p))})
            # Fixed probability bins are descriptive and never used to tune a model.
            for bin_index in range(10):
                selection = (p >= bin_index / 10) & (p < (bin_index + 1) / 10
                                                    if bin_index < 9 else p <= 1)
                if selection.any():
                    calibration.append({'model': model, 'repeat': repeat,
                                        'bin': bin_index, 'n': int(selection.sum()),
                                        'mean_predicted': float(p[selection].mean()),
                                        'observed_fraction': float(labels[selection].mean())})
    if len(by_key) != len(models) * 3 * len(ids):
        raise ValueError('unexpected extra OOF predictions')

    rng = np.random.default_rng(20261019)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    weights = np.zeros((10000, len(ids)), dtype=np.int16)
    weights[:, positive] = rng.multinomial(len(positive),
        np.full(len(positive), 1 / len(positive)), size=len(weights))
    weights[:, negative] = rng.multinomial(len(negative),
        np.full(len(negative), 1 / len(negative)), size=len(weights))
    bootstrap = {}
    for model in models:
        auc = np.mean([weighted_auc(labels, probabilities[model, repeat], weights)
                       for repeat in range(3)], axis=0)
        brier = np.mean([np.sum(weights * (probabilities[model, repeat] - labels) ** 2,
                                 axis=1) / len(ids) for repeat in range(3)], axis=0)
        bootstrap[model] = {'auc': auc, 'brier': brier}

    summary = []
    for model in models:
        for metric in ('auc', 'brier'):
            point = np.mean([row[metric] for row in repeated if row['model'] == model])
            draws = bootstrap[model][metric]
            summary.append({'comparison': model, 'metric': metric,
                            'estimate': float(point),
                            'ci_low': float(np.quantile(draws, 0.025)),
                            'ci_high': float(np.quantile(draws, 0.975))})
    paired_repeats = []
    for left, right in comparisons:
        for metric in ('auc', 'brier'):
            points = []
            for repeat in range(3):
                a = next(row[metric] for row in repeated
                         if row['model'] == left and row['repeat'] == repeat)
                b = next(row[metric] for row in repeated
                         if row['model'] == right and row['repeat'] == repeat)
                points.append(a - b)
                paired_repeats.append({'repeat': repeat, 'comparison': left + '-' + right,
                                       'metric': metric, 'difference': float(a - b)})
            draws = bootstrap[left][metric] - bootstrap[right][metric]
            summary.append({'comparison': left + '-' + right, 'metric': metric,
                            'estimate': float(np.mean(points)),
                            'ci_low': float(np.quantile(draws, 0.025)),
                            'ci_high': float(np.quantile(draws, 0.975))})
    return {'repeat_metrics': repeated, 'summary': summary,
            'paired_repeat_differences': paired_repeats, 'calibration_bins': calibration,
            'bootstrap_draws': len(weights), 'patient_n': len(ids)}


def mechanism_trigger(evaluation):
    by_name = {(row['comparison'], row['metric']): row['estimate']
               for row in evaluation['summary']}
    if any(by_name[(name, 'auc')] < 0.01 for name in ('M3-M1', 'M3-M2')):
        return False
    for name in ('M3-M1', 'M3-M2'):
        positive = sum(row['difference'] > 0 for row in evaluation['paired_repeat_differences']
                       if row['comparison'] == name and row['metric'] == 'auc')
        if positive < 2:
            return False
    return True


def plot_calibration(calibration_bins, out):
    """Plot descriptive OOF calibration; no refitting or threshold choice."""
    cache = Path(out).parent / 'matplotlib-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    models = sorted({row['model'] for row in calibration_bins})
    for model in models:
        bins = [row for row in calibration_bins if row['model'] == model]
        first = True
        for index in range(10):
            group = [row for row in bins if row['bin'] == index]
            if group:
                n = sum(row['n'] for row in group)
                ax.scatter(sum(row['mean_predicted'] * row['n'] for row in group) / n,
                           sum(row['observed_fraction'] * row['n'] for row in group) / n,
                           label=model if first else None, s=25)
                first = False
    ax.set(xlabel='Mean predicted probability', ylabel='Observed pregnancy fraction',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
