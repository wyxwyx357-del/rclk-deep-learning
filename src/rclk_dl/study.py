"""Frozen patient cohort, source review, folds, and read-only sequence cache."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import quality_features, simple_temporal_features
from .inputs import _one, read_case


NUMERIC = ('female_age', 'female_bmi', 'infertility_years',
           'transfer_embryo_count', 'endometrial_thickness_mm',
           'peristalsis_forward_count', 'peristalsis_reverse_count')
CATEGORICAL = ('infertility_type', 'embryo_type', 'endometrial_type', 'cycle_type')
SHAPE = ('F01', 'F07', 'F09')
K_SHAPE = SHAPE + ('F15',)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_review(review_path, flag_path, cohort):
    """Require a documented decision for every flagged clinical source record."""
    flagged = pd.read_csv(flag_path, usecols=['numeric_id']).numeric_id.drop_duplicates()
    flagged_ids = set(cohort.index[cohort.numeric_id.isin(flagged)])
    if len(flagged_ids) != 35:
        raise ValueError(f'expected 35 flagged cohort records, found {len(flagged_ids)}')
    if review_path is None:
        raise ValueError('clinical review CSV is required before formal training')
    review = pd.read_csv(review_path, dtype=str)
    if set(review.columns) != {'audit_id', 'status', 'evidence_note'} or review.audit_id.duplicated().any():
        raise ValueError('review needs unique audit_id,status,evidence_note rows')
    if set(review.audit_id) != flagged_ids:
        raise ValueError('review must cover exactly the 35 flagged audit IDs')
    allowed = {'confirmed', 'unresolved', 'confirmed_mismatch'}
    if not set(review.status).issubset(allowed):
        raise ValueError(f'review status must be one of {sorted(allowed)}')
    if review.evidence_note.isna().any() or review.evidence_note.str.strip().eq('').any():
        raise ValueError('each source decision needs an evidence note')
    if (review.status == 'confirmed_mismatch').any():
        raise ValueError('confirmed clinical mismatch: correct sources and regenerate cohort/splits first')
    return review.set_index('audit_id').status.to_dict()


def make_review_template(flag_path, cohort, out):
    """Write patient IDs only to a local ignored output, never to Git."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    flagged = set(pd.read_csv(flag_path, usecols=['numeric_id']).numeric_id)
    rows = cohort.loc[cohort.numeric_id.isin(flagged), ['numeric_id']].reset_index()
    if len(rows) != 35:
        raise ValueError('expected 35 flagged cohort records')
    rows['status'] = ''
    rows['evidence_note'] = ''
    out.parent.mkdir(parents=True, exist_ok=True)
    rows[['audit_id', 'status', 'evidence_note']].to_csv(out, index=False, encoding='utf-8-sig')
    return out


class Study:
    def __init__(self, cohort_path, formal_path, splits_path, root, flag_path,
                 review_path=None, require_review=True):
        self.paths = {'cohort': Path(cohort_path), 'formal': Path(formal_path),
                      'splits': Path(splits_path), 'flags': Path(flag_path)}
        if review_path is not None:
            self.paths['review'] = Path(review_path)
        self.root = Path(root)
        cohort = pd.read_csv(cohort_path)
        formal = pd.read_csv(formal_path, usecols=['case_id',
            'rsr_abs_median', 'cavity_width_strain_rate_abs_median',
            'longitudinal_wall_strain_rate_abs_median',
            'wall_curvature_change_rate_mm_inv_s_abs_median'])
        formal['numeric_id'] = formal.case_id.str.extract(r'_(\d+)_')[0].astype(int)
        formal = formal.rename(columns={
            'rsr_abs_median': 'formal_F01',
            'cavity_width_strain_rate_abs_median': 'formal_F07',
            'longitudinal_wall_strain_rate_abs_median': 'formal_F09',
            'wall_curvature_change_rate_mm_inv_s_abs_median': 'formal_F15'})
        cohort = cohort.merge(formal, on='numeric_id', validate='one_to_one', sort=False)
        if len(cohort) != 295 or cohort.audit_id.duplicated().any():
            raise ValueError('the frozen cohort must have 295 unique patients')
        for feature in K_SHAPE:
            if not np.allclose(cohort[feature], cohort['formal_' + feature],
                               rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f'{feature} differs from formal source')
        if sorted(cohort.outcome.unique()) != [0, 1] or int(cohort.outcome.sum()) != 191:
            raise ValueError('unexpected frozen outcome distribution')
        self.frame = cohort.set_index('audit_id')
        self.splits = json.loads(Path(splits_path).read_text(encoding='utf-8'))
        self._check_splits()
        self.review = (load_review(review_path, flag_path, self.frame)
                       if require_review else {})
        self._records = {}
        self._m2 = {}
        self._mq = {}

    def _check_splits(self):
        if len(self.splits) != 15 or {(s['repeat'], s['fold']) for s in self.splits} != {
                (r, f) for r in range(3) for f in range(5)}:
            raise ValueError('expected frozen 3 x 5 outer folds')
        all_ids = set(self.frame.index)
        for repeat in range(3):
            tested = []
            for s in (row for row in self.splits if row['repeat'] == repeat):
                train, test = set(s['outer_train_ids']), set(s['outer_test_ids'])
                if train & test or train | test != all_ids or len(s['inner']) != 5:
                    raise ValueError('invalid outer patient split')
                tested.extend(s['outer_test_ids'])
                for inner in s['inner']:
                    a, b = set(inner['train_ids']), set(inner['validation_ids'])
                    if a & b or a | b != train:
                        raise ValueError('invalid inner patient split')
                validation = [i for part in s['inner'] for i in part['validation_ids']]
                if len(validation) != len(train) or set(validation) != train:
                    raise ValueError('each inner training patient needs one validation appearance')
            if len(tested) != len(all_ids) or set(tested) != all_ids:
                raise ValueError('each patient needs one test appearance per repeat')

    def ids(self, analysis='primary'):
        if analysis == 'primary':
            return set(self.frame.index)
        if analysis == 'k':
            ids = set(self.frame.index[self.frame.F15.notna()])
            if len(ids) != 288:
                raise ValueError('expected 288 K patients')
            return ids
        if analysis == 'sensitivity':
            return set(self.frame.index) - {key for key, status in self.review.items()
                                             if status == 'unresolved'}
        raise ValueError(f'unknown analysis: {analysis}')

    def fold_ids(self, split, allowed):
        train = [i for i in split['outer_train_ids'] if i in allowed]
        test = [i for i in split['outer_test_ids'] if i in allowed]
        inner = [([i for i in part['train_ids'] if i in allowed],
                  [i for i in part['validation_ids'] if i in allowed])
                 for part in split['inner']]
        for one in (train, test, *(item for pair in inner for item in pair)):
            if len(one) < 2 or set(self.frame.loc[one, 'outcome']) != {0, 1}:
                raise ValueError('restricted split has only one outcome class')
        if set(train) & set(test) or len(train) + len(test) != len(allowed):
            raise ValueError('restricted outer split is not a partition')
        return train, test, inner

    def record(self, audit_id, include_k=False):
        case_id = self.frame.loc[audit_id, 'case_id']
        if audit_id not in self._records or (include_k and 'K' not in self._records[audit_id]['values']):
            self._records[audit_id] = read_case(case_id, self.root, include_k=include_k)
        return self._records[audit_id]

    def m2(self, audit_id):
        if audit_id not in self._m2:
            self._m2[audit_id] = simple_temporal_features(self.record(audit_id))
        return self._m2[audit_id]

    def mq(self, audit_id):
        if audit_id not in self._mq:
            self._mq[audit_id] = quality_features(self.record(audit_id), self.root)
        return self._mq[audit_id]

    def sequence_file_hashes(self, ids=None):
        """Hash the exact formal arrays, QC and topology evidence used by readers."""
        result = {}
        ids = self.frame.index if ids is None else ids
        for audit_id in ids:
            row = self.frame.loc[audit_id]
            patient = self.root / row.case_id
            core = patient / '02_追踪质控与形变'
            folders = (
                (core / '10_rsr_fourway', '*_rsr_fourway_v1.npz'),
                (core / '11_anatomical_deformation', '*_anatomical_deformation_v1.npz'),
                (core / '05_radial_tracking_corrected', '*_p3_image_boundary_evidence_v2.npz'),
                (core / '07_artifact_qc', '*_artifact_qc_v2_1.npz'),
            )
            if pd.notna(row.F15):
                folders += ((patient / '03_DICOM物理标定', '*_dicom_curvature_v1.npz'),)
            for folder, pattern in folders:
                path = _one(folder, pattern)
                result[str(path.relative_to(self.root)).replace('\\', '/')] = sha256(path)
        return result

    def source_hashes(self, files=None):
        result = {name: sha256(path) for name, path in self.paths.items()}
        files = self.sequence_file_hashes() if files is None else files
        digest = hashlib.sha256()
        for relative_path, file_hash in sorted(files.items()):
            digest.update(relative_path.encode('utf-8'))
            digest.update(file_hash.encode('ascii'))
        result['formal_sequence_and_qc_tree'] = digest.hexdigest()
        return result
