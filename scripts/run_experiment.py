"""Run the prespecified patient-level nested-CV experiments.

Formal runs require a completed local 35-case source review. All patient-level
predictions and model details are written under ignored outputs/, not Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
sys.path.insert(0, str(REPO / 'src'))
from rclk_dl.baselines import logistic_fold
from rclk_dl.evaluation import evaluate_oof, mechanism_trigger, plot_calibration
from rclk_dl.study import Study, make_review_template, sha256
from rclk_dl.training import tcn_fold


def code_hash():
    digest = hashlib.sha256()
    files = sorted((REPO / 'src').rglob('*.py')) + sorted((REPO / 'scripts').glob('*.py'))
    files += [REPO / 'protocol/model_plan.json']
    for path in files:
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def paths(parser):
    parser.add_argument('--root', type=Path,
                        default=Path('D:/uterus_2.0/最终追踪特征输出'))
    parser.add_argument('--cohort', type=Path,
                        default=WORK / 'explore_outputs/ER_fusion_auc_295_20261022/analysis_input_295.csv')
    parser.add_argument('--formal', type=Path,
                        default=Path('D:/uterus_2.0/最终追踪特征输出/全患者汇总/'
                                     '拓扑风险帧屏蔽与正式特征重算_20260914/最终正式主分析_F01-F20.csv'))
    parser.add_argument('--splits', type=Path,
                        default=WORK / 'explore_outputs/ER_fusion_auc_295_20261022/cv_splits.json')
    parser.add_argument('--flags', type=Path,
                        default=WORK / 'clinician_incremental_305_20261019/source_row_audit/'
                                     '06_原35例来源问题_含姓名仅限内部.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument('--analysis', required=True,
                        choices=('prepare-review', 'primary', 'sensitivity', 'k', 'mechanism'))
    parser.add_argument('--review', type=Path, help='Local 35-case source-review CSV')
    parser.add_argument('--out', type=Path, help='Output directory; default outputs/<analysis>')
    parser.add_argument('--primary-dir', type=Path,
                        default=REPO / 'outputs/primary',
                        help='Completed primary run for mechanism trigger and paired predictions')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    out = args.out or REPO / 'outputs' / args.analysis
    if args.analysis == 'prepare-review':
        study = Study(args.cohort, args.formal, args.splits, args.root,
                      args.flags, require_review=False)
        target = out / 'clinical_review_35.csv'
        make_review_template(args.flags, study.frame, target)
        print(f'Created local review template: {target} (35 cases; no statuses assumed)')
        return
    if args.review is None:
        parser.error('--review is required for any formal experiment')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this Python environment')
    device = torch.device(args.device)
    study = Study(args.cohort, args.formal, args.splits, args.root,
                  args.flags, review_path=args.review)
    sequence_hashes = study.sequence_file_hashes()
    hashes = study.source_hashes(sequence_hashes)
    allowed = study.ids('sensitivity' if args.analysis == 'sensitivity'
                        else 'k' if args.analysis == 'k' else 'primary')
    if args.analysis == 'primary':
        models = ('M1', 'M2', 'MQ', 'M3')
        comparisons = (('M3', 'M1'), ('M3', 'M2'), ('M3', 'MQ'))
    elif args.analysis == 'sensitivity':
        if len(allowed) == 295:
            parser.error('no unresolved source records: sensitivity refit is unnecessary')
        models = ('M1', 'M2', 'MQ', 'M3')
        comparisons = (('M3', 'M1'), ('M3', 'M2'), ('M3', 'MQ'))
    elif args.analysis == 'k':
        models = ('M3_RCL', 'M3_RCLK')
        comparisons = (('M3_RCLK', 'M3_RCL'),)
    else:
        primary_summary = json.loads((args.primary_dir / 'run_summary.json').read_text(encoding='utf-8'))
        if not primary_summary.get('mechanism_trigger'):
            parser.error('prespecified primary trigger was not met')
        if primary_summary['source_sha256'] != hashes:
            parser.error('primary run uses a different data or review version')
        models = ('M3_shuffle', 'M3_spatial_mean', 'M3_mask_quality')
        comparisons = tuple(('M3', model) for model in models)

    frozen = {'analysis': args.analysis, 'source_sha256': hashes,
              'code_sha256': code_hash(), 'device': args.device,
              'models': models, 'patients': len(allowed),
              'positive': int(study.frame.loc[sorted(allowed), 'outcome'].sum())}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != frozen:
            raise ValueError('existing output belongs to a different code or source version')
    else:
        if any(out.iterdir()):
            raise ValueError('new output directory is not empty')
        write_json(manifest_path, frozen)
        write_json(out / 'formal_sequence_file_hashes.json', sequence_hashes)
    started = time.monotonic()
    predictions = []
    for split in study.splits:
        train_ids, test_ids, inner = study.fold_ids(split, allowed)
        for model in models:
            path = out / f"repeat{split['repeat']}_fold{split['fold']}_{model}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding='utf-8'))
            else:
                if model in ('M1', 'M2', 'MQ'):
                    probability, details = logistic_fold(study, model, train_ids,
                                                          test_ids, inner)
                else:
                    include_k = model == 'M3_RCLK'
                    control = model.removeprefix('M3_') if model.startswith('M3_') and args.analysis == 'mechanism' else 'none'
                    probability, details = tcn_fold(study, split, allowed, device,
                        include_k=include_k, static_k=args.analysis == 'k', control=control)
                payload = {'repeat': split['repeat'], 'fold': split['fold'],
                           'model': model, 'predictions': [
                               {'audit_id': i, 'outcome': int(study.frame.loc[i, 'outcome']),
                                'probability': float(p)} for i, p in zip(test_ids, probability)],
                           'fit_details': details}
                write_json(path, payload)
                print(f"complete repeat={split['repeat']} fold={split['fold']} model={model}", flush=True)
            if (payload['repeat'], payload['fold'], payload['model']) != (
                    split['repeat'], split['fold'], model):
                raise ValueError('saved fold result identity mismatch')
            if {row['audit_id'] for row in payload['predictions']} != set(test_ids) or (
                    len(payload['predictions']) != len(test_ids)):
                raise ValueError('saved fold predictions do not match frozen test IDs')
            predictions.extend({'repeat': split['repeat'], 'fold': split['fold'],
                                'model': model, **row} for row in payload['predictions'])

    compared_models = models
    if args.analysis == 'mechanism':
        primary_rows = pd.read_csv(args.primary_dir / 'oof_predictions.csv')
        predictions.extend(primary_rows.loc[primary_rows.model.eq('M3')].to_dict('records'))
        compared_models = ('M3',) + models
    evaluated = evaluate_oof(predictions, study.frame, allowed, compared_models, comparisons)
    pd.DataFrame(predictions).to_csv(out / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
    for key in ('repeat_metrics', 'summary', 'paired_repeat_differences', 'calibration_bins'):
        pd.DataFrame(evaluated[key]).to_csv(out / f'{key}.csv', index=False, encoding='utf-8-sig')
    plot_calibration(evaluated['calibration_bins'], out / 'calibration.png')
    if study.source_hashes() != hashes or code_hash() != frozen['code_sha256']:
        raise ValueError('code or input changed during run')
    summary = {**frozen, 'status': 'complete', 'elapsed_s': time.monotonic() - started,
               'patient_n': evaluated['patient_n'],
               'bootstrap_draws': evaluated['bootstrap_draws'],
               'mechanism_trigger': mechanism_trigger(evaluated) if args.analysis == 'primary' else None,
               'summary': evaluated['summary'],
               'conditional_on_fixed_oof_predictions': True,
               'validation_status': 'exploratory_internal_development_cohort'}
    write_json(out / 'run_summary.json', summary)
    print(json.dumps({'status': 'complete', 'analysis': args.analysis,
                      'patient_n': len(allowed), 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
