"""Fold-local clinical/static processing and frozen Elastic Net baselines."""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import M2_NAMES, MQ_NAMES
from .study import CATEGORICAL, K_SHAPE, NUMERIC, SHAPE


LOGISTIC_GRID = {'model__C': [0.01, 0.1, 1, 10],
                 'model__l1_ratio': [0.25, 0.5, 0.75, 1]}


def numeric_columns(model, include_k=False):
    shape = K_SHAPE if include_k else SHAPE
    extra = M2_NAMES if model == 'M2' else MQ_NAMES if model == 'MQ' else ()
    return list(NUMERIC + shape + extra)


def frame_for(study, ids, model='M1', include_k=False):
    ids = list(ids)
    frame = study.frame.loc[ids, list(NUMERIC + CATEGORICAL + K_SHAPE + ('outcome',))].copy()
    if model in ('M2', 'MQ'):
        get = study.m2 if model == 'M2' else study.mq
        extra = pd.DataFrame([get(i) for i in ids], index=ids)
        frame = frame.join(extra, validate='one_to_one')
    return frame


def preprocessor(numbers):
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median',
                                                     keep_empty_features=True)),
                              ('scale', StandardScaler())]), numbers),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore',
                                                        sparse_output=False))]), list(CATEGORICAL)),
    ])


def static_features(study, train_ids, other_id_groups, include_k=False,
                    include_quality=False):
    """Fit imputation, encoding and scale on this training fold only."""
    model = 'MQ' if include_quality else 'M1'
    numbers = numeric_columns(model, include_k)
    train = frame_for(study, train_ids, model, include_k)
    prep = preprocessor(numbers)
    prepared = [prep.fit_transform(train).astype(np.float32)]
    for ids in other_id_groups:
        prepared.append(prep.transform(frame_for(study, ids, model, include_k)).astype(np.float32))
    if any(not np.isfinite(a).all() for a in prepared):
        raise ValueError('nonfinite static inputs after fold-local preprocessing')
    return prepared, prep


def logistic_fold(study, model, train_ids, test_ids, inner,
                  include_k=False, seed=20260914):
    if model not in ('M1', 'M2', 'MQ'):
        raise ValueError('unknown logistic baseline')
    numbers = numeric_columns(model, include_k)
    train = frame_for(study, train_ids, model, include_k)
    test = frame_for(study, test_ids, model, include_k)
    indices = {patient: j for j, patient in enumerate(train_ids)}
    cv = [(np.array([indices[i] for i in a]), np.array([indices[i] for i in b]))
          for a, b in inner]
    pipe = Pipeline([('prepare', preprocessor(numbers)),
                     ('model', LogisticRegression(penalty='elasticnet', solver='saga',
                                                  max_iter=30000, tol=1e-4,
                                                  random_state=seed))])
    search = GridSearchCV(pipe, LOGISTIC_GRID, cv=cv, scoring='roc_auc', n_jobs=1,
                          error_score='raise', return_train_score=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        search.fit(train, train.outcome.to_numpy(dtype=int))
    probability = search.predict_proba(test)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError('nonfinite baseline probability')
    scores = [{**row, 'mean_inner_auc': float(search.cv_results_['mean_test_score'][j])}
              for j, row in enumerate(search.cv_results_['params'])]
    return probability, {
        'best_parameters': search.best_params_, 'best_inner_auc': float(search.best_score_),
        'grid_scores': scores,
        'convergence_warnings': sum(issubclass(w.category, ConvergenceWarning) for w in caught),
    }
