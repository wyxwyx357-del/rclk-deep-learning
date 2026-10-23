"""Run the prespecified framewise R/C/L MLP against saved 260-patient OOF results.

This entrypoint is for use after review. Importing it never starts training.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from rclk_dl.evaluation import evaluate_oof
from rclk_dl.frame_mlp import CHANNELS, WIDTH, WEIGHT_DECAY, frame_mlp_fold
from rclk_dl.study import Study, sha256
from run_experiment import code_hash, paths, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path,
                        default=REPO / 'outputs/exploratory_260')
    parser.add_argument('--out', type=Path, default=REPO / 'outputs/frame_mlp_260')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this Python environment')
    baseline = json.loads((args.baseline_dir / 'run_summary.json').read_text(encoding='utf-8'))
    if (baseline['status'] != 'complete' or baseline['analysis'] != 'sensitivity'
            or baseline['patients'] != 260 or not {'M1', 'M3'} <= set(baseline['models'])):
        raise ValueError('baseline must be the completed 260-patient sensitivity run')
    study = Study(args.cohort, args.formal, args.splits, args.root,
                  args.flags, review_path=args.review)
    allowed = study.ids('sensitivity')
    if len(allowed) != 260 or study.source_hashes() != baseline['source_sha256']:
        raise ValueError('cohort, review, or formal input differs from saved TCN run')
    saved = pd.read_csv(args.baseline_dir / 'oof_predictions.csv')
    saved = saved[saved.model.isin(('M1', 'M3'))]
    expected_folds = {(split['repeat'], audit_id): split['fold']
                      for split in study.splits
                      for audit_id in split['outer_test_ids'] if audit_id in allowed}
    for row in saved.itertuples(index=False):
        if expected_folds.get((row.repeat, row.audit_id)) != row.fold:
            raise ValueError('saved baseline prediction uses a different patient fold')
    # This also rejects duplicate, missing, or relabeled baseline predictions.
    evaluate_oof(saved.to_dict('records'), study.frame, allowed,
                 ('M1', 'M3'), (('M3', 'M1'),))
    frozen = {'source_sha256': baseline['source_sha256'],
              'baseline_code_sha256': baseline['code_sha256'],
              'baseline_oof_sha256': sha256(args.baseline_dir / 'oof_predictions.csv'),
              'comparison_code_sha256': code_hash(),
              'protocol_sha256': sha256(REPO / 'protocol/FRAME_MLP_REVIEW_中文.md'),
              'baseline_dir': str(args.baseline_dir.resolve()),
              'device': args.device, 'patients': len(allowed),
              'frame_features': 2 * CHANNELS,
              'width': WIDTH, 'weight_decay': WEIGHT_DECAY}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != frozen:
            raise ValueError('existing output belongs to a different code or source version')
    else:
        if any(args.out.iterdir()):
            raise ValueError('new output directory is not empty')
        write_json(manifest, frozen)
    rows = saved.to_dict('records')
    for split in study.splits:
        train_ids, test_ids, _ = study.fold_ids(split, allowed)
        output = args.out / f"repeat{split['repeat']}_fold{split['fold']}_FRAME_MLP.json"
        if output.exists():
            payload = json.loads(output.read_text(encoding='utf-8'))
        else:
            probability, details = frame_mlp_fold(study, split, allowed,
                                                  torch.device(args.device))
            payload = {'repeat': split['repeat'], 'fold': split['fold'],
                       'model': 'FRAME_MLP',
                       'predictions': [
                           {'audit_id': audit_id,
                            'outcome': int(study.frame.loc[audit_id, 'outcome']),
                            'probability': float(value)}
                           for audit_id, value in zip(test_ids, probability)],
                       'fit_details': details}
            write_json(output, payload)
            print(f"complete repeat={split['repeat']} fold={split['fold']} FRAME_MLP",
                  flush=True)
        if (payload['repeat'], payload['fold'], payload['model']) != (
                split['repeat'], split['fold'], 'FRAME_MLP'):
            raise ValueError('saved fold identity changed')
        if [item['audit_id'] for item in payload['predictions']] != test_ids:
            raise ValueError('saved fold patient order changed')
        rows.extend({'repeat': split['repeat'], 'fold': split['fold'],
                     'model': 'FRAME_MLP', **item}
                    for item in payload['predictions'])
    if (study.source_hashes() != frozen['source_sha256']
            or code_hash() != frozen['comparison_code_sha256']
            or sha256(REPO / 'protocol/FRAME_MLP_REVIEW_中文.md') != frozen['protocol_sha256']
            or sha256(args.baseline_dir / 'oof_predictions.csv') != frozen['baseline_oof_sha256']):
        raise ValueError('code or input changed during comparison')
    result = evaluate_oof(rows, study.frame, allowed, ('M1', 'M3', 'FRAME_MLP'),
                          (('FRAME_MLP', 'M3'), ('FRAME_MLP', 'M1')))
    pd.DataFrame(rows).to_csv(args.out / 'oof_predictions.csv', index=False)
    write_json(args.out / 'run_summary.json', {'status': 'complete', **frozen,
                                               **result})
    print('complete: 260-patient framewise MLP comparison', flush=True)


if __name__ == '__main__':
    main()

