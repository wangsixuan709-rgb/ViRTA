"""Standard ten-fold orchestration for the full model."""
import csv
from datetime import datetime
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

from prepare_tenfold import FOLDS, PROTOCOL, SPLIT_SEED, digest, prepare, protocol as check_protocol, save, validate_manifest

METRICS = ('accuracy', 'macro_f1', 'fake_precision', 'fake_recall', 'fake_f1',
           'real_precision', 'real_recall', 'real_f1')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def summarize(output, checkpoint, folds=FOLDS, protocol=PROTOCOL):
    if protocol != check_protocol(folds):
        raise ValueError('Only standard ten-fold summaries are supported')
    output, checkpoint = Path(output), Path(checkpoint)
    rows = []
    for fold in range(folds):
        path = output / f'fold_{fold:02d}' / 'result.json'
        if path.is_file():
            record = json.loads(path.read_text())
            assert record['fold'] == fold
            rows.append(record)
    with (output / 'fold_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fold', 'seed', 'test_rows', 'best_epoch', 'trained_epochs', *METRICS])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    if len(rows) != folds:
        return len(rows)
    mean = {key: statistics.mean(row[key] for row in rows) for key in METRICS}
    std = {key: statistics.stdev(row[key] for row in rows) for key in METRICS}
    summary = {'state': 'completed', 'protocol': protocol, 'completed_folds': folds,
               'aggregation': f'Unweighted arithmetic mean of all {folds} held-out test evaluations; no best-fold selection.',
               'mean': mean, 'sample_std': std, 'folds': rows,
               'total_test_predictions': sum(row['test_rows'] for row in rows),
               'mean_test_rows': statistics.mean(row['test_rows'] for row in rows)}
    save(output / 'summary.json', summary)
    save(str(checkpoint) + '.test_metrics.json', mean)
    save(str(checkpoint) + '.epoch.json', {'selection': 'ten_fold_mean',
                                         'folds': [row['selection'] for row in rows]})
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['statistic', *METRICS])
        writer.writeheader()
        writer.writerow({'statistic': 'mean', **mean})
        writer.writerow({'statistic': 'sample_std', **std})
    lines = [f'# {folds} 折交叉验证结果', '', f'协议：{protocol}；每条样本恰好测试一次。',
             f'以下是全部 {folds} 次测试指标的算术平均值（百分比）；每折从头训练，不按测试表现选折。', '',
             f'| 指标 | {folds} 折平均值（%） |', '|---|---:|']
    lines += [f'| {k} | {100 * mean[k]:.4f} |' for k in METRICS]
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return folds


def worker_command(python, script, args):
    command = [str(python), '-B', '-u', str(script)]
    for key, value in vars(args).items():
        if value is None or value is False:
            continue
        flag = '--_cv-worker' if key == '_cv_worker' else '--' + key.replace('_', '-')
        if value is True:
            command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def verify_worker(args):
    if args.cv_splits is None or args.cv_fold is None or args.no_validation:
        raise ValueError('A fold worker requires a verified CV manifest and an independent validation set')
    manifest = validate_manifest(args.cv_splits, args.dataset, SPLIT_SEED, verify_images=True,
                                 folds=getattr(args, 'cv_count', FOLDS))
    if not 0 <= args.cv_fold < len(manifest['folds']):
        raise ValueError('Fold index is outside the declared CV manifest')
    fold = manifest['folds'][args.cv_fold]
    if Path(args.data_root).resolve() != Path(fold['data_root']):
        raise ValueError('The worker data root does not match its declared fold')
    return fold


def run(args):
    from copy import deepcopy
    from virta_dataset import build_dataset, configure_data_root
    folds = getattr(args, 'cv_count', FOLDS)
    check_protocol(folds)
    if args.no_validation:
        raise ValueError('All experiments require CV with independent validation; --no-validation is incompatible')
    if args.cv_fold is not None:
        raise ValueError('--cv-fold is internal; start the public entry without it to run every fold')
    if args.output is None:
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        args.output = str(Path(__file__).resolve().parent / 'virta_runs' / f'{args.dataset}_{folds}fold_{stamp}' / 'best.pth')
    checkpoint = Path(args.output).resolve()
    args.output = str(checkpoint)
    if args.vprm_checkpoint:
        args.vprm_checkpoint = str(Path(args.vprm_checkpoint).resolve())
    if not args.vprm_checkpoint or not Path(args.vprm_checkpoint).is_file():
        raise FileNotFoundError('Provide --vprm-checkpoint for full-model training')
    configure_data_root(args.dataset, args.data_root)
    source = Path(build_dataset(args.dataset, 'train', False).csv_path).parent.resolve()
    root = Path(__file__).resolve().parent
    shared = root / 'data_splits' / 'tenfold_80_10_10' / f'{args.dataset}_seed{SPLIT_SEED}'
    shared.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = (Path(args.cv_splits).resolve() if args.cv_splits
                     else prepare(source, shared, args.dataset, SPLIT_SEED, folds=folds))
    manifest = validate_manifest(manifest_path, args.dataset, SPLIT_SEED, verify_images=True, folds=folds)
    args.cv_splits = str(manifest_path)
    args.data_root = str(source)
    output = Path(str(checkpoint) + '.cv')
    config = vars(deepcopy(args))
    config.update(cv_resume=False, cv_prepare_only=False)
    if (output / 'plan.json').is_file():
        if not args.cv_resume:
            raise FileExistsError('Existing CV run: use a fresh --output or explicit --cv-resume')
        plan = json.loads((output / 'plan.json').read_text())
        if plan['config'] != config or plan['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Resume configuration or split manifest differs from the saved plan')
    else:
        if checkpoint.exists():
            raise FileExistsError('Preserve the old single-run checkpoint; select a fresh output')
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / 'source_snapshot'
        snapshot.mkdir()
        for path in root.glob('*.py'):
            shutil.copy2(path, snapshot / path.name)
        for folder in ('vendor', 'pretrained', 'vprm_corruption'):
            (snapshot / folder).symlink_to(root / folder, target_is_directory=True)
        packages = {}
        for name in ('torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        pretrained = Path(os.environ.get('VIRTA_PRETRAINED_DIR', os.environ.get('MMBE_PRETRAINED_DIR', root / 'pretrained')))
        backbone = pretrained / ('clip_cn_vit-b-16.pt' if args.dataset == 'weibo' else 'ViT-B-16.pt')
        plan = {'created_at': now(), 'protocol': manifest['protocol'], 'dataset': args.dataset, 'seed': args.seed,
                'split_seed': SPLIT_SEED,
                'folds': folds, 'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                'checkpoint_prefix': str(checkpoint), 'output': str(output), 'config': config,
                'source_sha256': {path.name: digest(path) for path in snapshot.glob('*.py')},
                'vprm_sha256': digest(args.vprm_checkpoint) if args.vprm_checkpoint else None,
                'backbone_sha256': {str(backbone.resolve()): digest(backbone)} if backbone.is_file() else {},
                'environment': {'python': sys.version, 'packages': packages},
                'test_evaluations_per_fold': 1, 'fresh_model_optimizer_and_scheduler_per_fold': True,
                'selection': 'Each fold uses ONLY its own validation Macro F1 and loss; test after selection.',
                'aggregation': f'Unweighted mean of all {folds} test results; no incomplete final mean.'}
        save(output / 'plan.json', plan)
        save(str(checkpoint) + '.config.json', config)
        save(str(checkpoint) + '.cv.json', {'output': str(output), 'manifest': str(manifest_path), 'folds': folds})
    snapshot = output / 'source_snapshot'
    for name, expected in plan['source_sha256'].items():
        if digest(snapshot / name) != expected:
            raise ValueError(f'The frozen source snapshot changed: {name}')
    if args.vprm_checkpoint and digest(args.vprm_checkpoint) != plan['vprm_sha256']:
        raise ValueError('VPRM checkpoint changed since the run was prepared')
    for path, expected in plan['backbone_sha256'].items():
        if digest(path) != expected:
            raise ValueError('The pretrained backbone changed since the run was prepared')
    status = {'state': 'prepared', 'runner_pid': os.getpid(), 'updated_at': now(),
              'completed_folds': summarize(output, checkpoint, folds, manifest['protocol']), 'total_folds': folds}
    save(output / 'status.json', status)
    if args.cv_prepare_only:
        print('CV_PREPARED', output, flush=True)
        return
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(args.seed),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        for fold in manifest['folds']:
            number = fold['fold']
            folder = output / f'fold_{number:02d}'
            if (folder / 'result.json').is_file():
                result = json.loads((folder / 'result.json').read_text())
                if result.get('checkpoint_retained', True) and digest(folder / 'best.pth') != result['checkpoint_sha256']:
                    raise ValueError('A completed fold checkpoint changed')
                continue
            if folder.exists():
                folder.rename(folder.with_name(folder.name + '_incomplete_' + str(os.getpid())))
            folder.mkdir()
            child_args = deepcopy(args)
            child_args._cv_worker, child_args.cv_fold = True, number
            child_args.cv_prepare_only = child_args.cv_resume = False
            child_args.data_root, child_args.output = fold['data_root'], str(folder / 'best.pth')
            command = worker_command(sys.executable, snapshot / 'train_virta.py', child_args)
            save(folder / 'command.json', command)
            with (folder / 'train.log').open('w') as stream:
                process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
                status.update(state='running', fold=number, child_pid=process.pid,
                              log=str(folder / 'train.log'), updated_at=now())
                save(output / 'status.json', status)
                print('CV_START', number, process.pid, flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f'Fold {number} failed with exit={code}; see {folder / "train.log"}')
            metrics = json.loads((folder / 'best.pth.test_metrics.json').read_text())
            selection = json.loads((folder / 'best.pth.epoch.json').read_text())
            predictions = json.loads((folder / 'best.pth.predictions.json').read_text())
            if len(predictions['labels']) != fold['counts']['test']['total']:
                raise ValueError('The fold did not test its entire held-out test set')
            result = {'fold': number, 'seed': args.seed, 'test_rows': len(predictions['labels']),
                      'best_epoch': selection['epoch'], 'trained_epochs': selection['trained_epochs'],
                      **{key: metrics[key] for key in METRICS}, 'selection': selection,
                      'checkpoint': child_args.output, 'checkpoint_sha256': digest(child_args.output)}
            if getattr(args, 'discard_fold_weights', False):
                if number == 0 and getattr(args, 'tsne_first_fold', False):
                    for name in ('best.pth.features.npz', 'tsne.png', 'tsne.pdf', 'tsne.svg', 'tsne.metadata.json'):
                        if not (folder / name).is_file():
                            raise FileNotFoundError(f'Cannot remove first-fold weights before saving {name}')
                Path(child_args.output).unlink()
                result['checkpoint_retained'] = False
            save(folder / 'result.json', result)
            status.update(completed_folds=summarize(output, checkpoint, folds, manifest['protocol']), child_pid=None, updated_at=now())
            save(output / 'status.json', status)
            print('CV_FOLD_COMPLETE', number, json.dumps(metrics), flush=True)
        assert summarize(output, checkpoint, folds, manifest['protocol']) == folds
        export_saved_predictions(checkpoint, checkpoint.parent / 'errors.csv')
        status.update(state='completed', completed_folds=folds, finished_at=now(), updated_at=now(), child_pid=None)
        save(output / 'status.json', status)
        print('CV_COMPLETE', output / 'summary.json', flush=True)
        print(f'CV_{folds}FOLD_MEAN', json.dumps(json.loads((output / 'summary.json').read_text())['mean']), flush=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error), updated_at=now())
        save(output / 'status.json', status)
        raise


def export_saved_predictions(checkpoint, output):
    """Compatibility evaluation: export the ONE saved test pass; do not test again."""
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = json.loads(Path(str(checkpoint) + '.cv.json').read_text())
    check_protocol(metadata['folds'])
    summary = json.loads((Path(metadata['output']) / 'summary.json').read_text())
    if (summary['protocol'] != PROTOCOL or summary['completed_folds'] != FOLDS
            or [fold['fold'] for fold in summary['folds']] != list(range(FOLDS))):
        raise ValueError('Evaluation requires all ten completed folds of the standard 80/10/10 protocol')
    errors = []
    for fold in summary['folds']:
        predictions = json.loads((Path(fold['checkpoint'] + '.predictions.json')).read_text())
        for index, (label, prediction, news_id) in enumerate(zip(predictions['labels'], predictions['predictions'], predictions['news_ids'])):
            if label != prediction:
                errors.append({'fold': fold['fold'], 'row_index': index, 'news_id': news_id,
                               'label': label, 'prediction': prediction})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fold', 'row_index', 'news_id', 'label', 'prediction'])
        writer.writeheader()
        writer.writerows(errors)
    config = json.loads(Path(str(checkpoint) + '.config.json').read_text())
    save(output.with_suffix('.summary.json'), {**summary['mean'], 'dataset': config['dataset'],
         'completed_folds': summary['completed_folds'], 'aggregation': summary['aggregation'],
         'test_rows': summary['total_test_predictions'], 'mean_test_rows': summary['mean_test_rows'],
         'visual_restoration': 'VPRM',
         'atm_keep_ratio': config['atm_keep_ratio'], 'errors': len(errors),
         'source': 'saved per-fold test predictions; no repeated test inference'})
