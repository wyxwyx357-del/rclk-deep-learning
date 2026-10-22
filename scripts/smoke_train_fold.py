"""One actual-data training update on the first frozen outer training fold.

This is a program smoke test. It never reads outer test patient arrays or
reports a validation score, and it is not the formal nested-CV experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
sys.path.insert(0, str(REPO / 'src'))
from rclk_dl.batches import pad_patient_runs, rcl_runs
from rclk_dl.inputs import read_case
from rclk_dl.scaling import fit_family_scaler, scale_patient_runs
from rclk_dl.tcn import PatientTCN


NUMERIC = ['female_age', 'female_bmi', 'infertility_years',
           'transfer_embryo_count', 'endometrial_thickness_mm',
           'peristalsis_forward_count', 'peristalsis_reverse_count',
           'F01', 'F07', 'F09']
CATEGORICAL = ['infertility_type', 'embryo_type', 'endometrial_type', 'cycle_type']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('D:/uterus_2.0/最终追踪特征输出'))
    parser.add_argument('--cohort', type=Path,
                        default=WORK / 'explore_outputs/ER_fusion_auc_295_20261022/analysis_input_295.csv')
    parser.add_argument('--formal', type=Path,
                        default=Path('D:/uterus_2.0/最终追踪特征输出/全患者汇总/'
                                     '拓扑风险帧屏蔽与正式特征重算_20260914/最终正式主分析_F01-F20.csv'))
    parser.add_argument('--splits', type=Path,
                        default=WORK / 'explore_outputs/ER_fusion_auc_295_20261022/cv_splits.json')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/tcn_fold_smoke.json')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'output already exists: {args.out}; choose a new --out path')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA PyTorch is unavailable; run scripts/check_device.py')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    started = time.monotonic()

    cohort = pd.read_csv(args.cohort)
    formal = pd.read_csv(args.formal, usecols=['case_id'])
    formal['numeric_id'] = formal.case_id.str.extract(r'_(\d+)_')[0].astype(int)
    joined = cohort.merge(formal, on='numeric_id', validate='one_to_one', sort=False)
    if len(joined) != 295 or joined.audit_id.duplicated().any():
        raise ValueError('frozen 295-patient cohort does not match formal case IDs')
    by_id = joined.set_index('audit_id')
    splits = json.loads(args.splits.read_text(encoding='utf-8'))
    fold = next(s for s in splits if s['repeat'] == 0 and s['fold'] == 0)
    train_ids, test_ids = fold['outer_train_ids'], fold['outer_test_ids']
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(by_id.index):
        raise ValueError('frozen outer split is not a patient partition')
    if len(train_ids) < 16:
        raise ValueError('training fold is too small for the frozen batch size')
    selected = by_id.loc[train_ids[:16]]

    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    category = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    static_prep = ColumnTransformer([('numeric', numeric, NUMERIC),
                                     ('category', category, CATEGORICAL)])
    static_prep.fit(by_id.loc[train_ids])
    static = static_prep.transform(selected).astype(np.float32)
    patient_runs = [rcl_runs(read_case(case_id, args.root)) for case_id in selected.case_id]
    means, scales = fit_family_scaler(patient_runs)
    scaled = scale_patient_runs(patient_runs, means, scales)
    batch = tuple(tensor.to(device) for tensor in pad_patient_runs(scaled, static))
    labels = torch.as_tensor(selected.outcome.to_numpy(dtype=np.float32), device=device)

    torch.manual_seed(20261022)
    if device == 'cuda':
        torch.cuda.manual_seed_all(20261022)
        torch.cuda.reset_peak_memory_stats()
    model = PatientTCN(static_features=static.shape[1], width=8, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    model.train()
    loss = F.binary_cross_entropy_with_logits(model(*batch), labels)
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite loss')
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('nonfinite gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    result = {
        'status': 'program_smoke_passed', 'formal_training': False,
        'outer_repeat': 0, 'outer_fold': 0, 'outer_test_read': False,
        'batch_patients': len(selected), 'batch_runs': sum(map(len, patient_runs)),
        'sequence_scaler_fit_patients': len(selected),
        'static_preprocessor_fit_patients': len(train_ids),
        'static_features': static.shape[1], 'model_parameters': sum(p.numel() for p in model.parameters()),
        'loss_finite': True, 'gradients_finite': True, 'one_optimizer_step': True,
        'device': device, 'torch': torch.__version__, 'sklearn': sklearn.__version__,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'elapsed_s': time.monotonic() - started,
        'source_sha256': {name: sha256(path) for name, path in
                          [('cohort', args.cohort), ('formal', args.formal), ('splits', args.splits)]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
