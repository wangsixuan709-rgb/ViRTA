"""Runnable checks: intact pairs, rotation coverage, ten workers and honest means."""
from collections import Counter
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import cross_validation as cv
from prepare_tenfold import FOLDS, prepare, rotation, stratified_bins, validate_manifest


def check_rotation():
    for fake, real, train_size in [(4627, 4778, 7524), (2524, 10131, 10124)]:
        rows = [{'label': label, 'pair_hash': f'{label}-{i}'}
                for label, total in [(0, fake), (1, real)] for i in range(total)]
        rows[1]['pair_hash'] = rows[0]['pair_hash']
        rows[fake + 1]['pair_hash'] = rows[fake]['pair_hash']
        bins = stratified_bins(rows, 666)
        assert bins == stratified_bins(rows, 666)
        assert bins != stratified_bins(rows, 667)
        seen = {s: Counter() for s in ('train', 'val', 'test')}
        total = fake + real
        test_folds = []
        for fold in range(FOLDS):
            split = rotation(bins, fold)
            assert len(split['test']) in (total // 10, total // 10 + 1)
            assert len(split['val']) in (total // 10, total // 10 + 1)
            assert len(split['train']) == train_size
            assert sorted(i for indices in split.values() for i in indices) == list(range(total))
            keys = {s: {rows[i]['pair_hash'] for i in ids} for s, ids in split.items()}
            assert not (keys['train'] & keys['val'] or keys['train'] & keys['test'] or keys['val'] & keys['test'])
            test_folds.append(set(split['test']))
            for name, indices in split.items():
                seen[name].update(indices)
                for label, count in [(0, fake), (1, real)]:
                    expected = count * {'train': .8, 'val': .1, 'test': .1}[name]
                    assert abs(sum(rows[i]['label'] == label for i in indices) - expected) < 2
        assert all(not (left & right) for i, left in enumerate(test_folds) for right in test_folds[i + 1:])
        for name, expected in [('train', 8), ('val', 1), ('test', 1)]:
            assert len(seen[name]) == total and set(seen[name].values()) == {expected}
    print('PASS: both datasets, fixed seed, stratification, intact pairs, 8/1/1 coverage and disjoint test folds')


def check_orchestration():
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'source'
        source.mkdir()
        for split, begin, end in [('train', 0, 28), ('val', 28, 32), ('test', 32, 40)]:
            with (source / f'twitter_{split}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['text', 'image', 'label'])
                writer.writeheader()
                for i in range(begin, end):
                    image = source / f'{i}.png'
                    Image.new('RGB', (3, 3), (i, i, i)).save(image)
                    writer.writerow({'text': f'row {i}', 'image': str(image), 'label': i % 2})
        manifest_path = prepare(source, root / 'splits', 'twitter', 666, workers=1)
        manifest = validate_manifest(manifest_path, 'twitter', 666, verify_images=True)
        assert manifest['pool_rows'] == 40
        assert manifest['sample_coverage'] == {'train': 8, 'val': 1, 'test': 1}
        worker = SimpleNamespace(cv_splits=str(manifest_path), cv_fold=0, no_validation=False,
                                 dataset='twitter', seed=42, data_root=manifest['folds'][0]['data_root'])
        assert cv.verify_worker(worker) == manifest['folds'][0], 'Training seed changed the shared split seed'
        incompatible = root / 'old_manifest.json'
        incompatible.write_text(json.dumps({**manifest, 'protocol': 'ten_bin_rotation_70_10_20'}))
        try:
            validate_manifest(incompatible)
            raise AssertionError('The old 7/1/2 protocol was accepted')
        except ValueError:
            pass
        assert prepare(source, root / 'splits', 'twitter', 666) == manifest_path
        try:
            validate_manifest(manifest_path, seed=42)
            raise AssertionError('Mismatched seed was accepted')
        except ValueError:
            pass
        calls = []

        class Process:
            def __init__(self, command, **kwargs):
                number = int(command[command.index('--cv-fold') + 1])
                output = Path(command[command.index('--output') + 1])
                assert '--_cv-worker' in command and command[command.index('--seed') + 1] == '666'
                assert Path(command[command.index('--data-root') + 1]) == Path(manifest['folds'][number]['data_root'])
                assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
                assert number not in calls and not output.exists()
                calls.append(number)
                self.pid = 10000 + number
                output.write_bytes(f'fresh checkpoint {number}'.encode())
                cv.save(str(output) + '.test_metrics.json', dict.fromkeys(cv.METRICS, .5 + number / 100))
                cv.save(str(output) + '.epoch.json', {'epoch': 2, 'trained_epochs': 5})
                cv.save(str(output) + '.predictions.json', {'labels': [0, 1] * 2, 'predictions': [0, 1] * 2,
                                                          'news_ids': [str(i) for i in range(4)]})
                # No final mean is published from fewer than ten results.
                if number < 9:
                    assert not (output.parent.parent / 'summary.json').exists()

            def wait(self):
                return 0

        dataset = ModuleType('virta_dataset')
        dataset.configure_data_root = lambda *_: None
        dataset.build_dataset = lambda *_: SimpleNamespace(csv_path=source / 'twitter_train.csv')
        args = SimpleNamespace(dataset='twitter', data_root=str(source), seed=666, no_validation=False,
                               cv_fold=None, cv_splits=str(manifest_path), cv_resume=False, cv_prepare_only=False,
                               output=str(root / 'run' / 'best.pth'), vprm_checkpoint=str(manifest_path), atm_keep_ratio=.7, _cv_worker=False)
        with patch.dict('sys.modules', {'virta_dataset': dataset}), patch.object(cv.subprocess, 'Popen', Process):
            cv.run(args)
            assert calls == list(range(10))
            result = json.loads((Path(args.output + '.cv') / 'summary.json').read_text())
            assert abs(result['mean']['accuracy'] - .545) < 1e-12
            assert result['completed_folds'] == 10 and result['total_test_predictions'] == 40
            assert result['sample_std']['accuracy'] > 0
            args.cv_resume = True
            cv.run(args)
            assert calls == list(range(10)), 'Resume reran completed folds'
        original = (source / 'twitter_train.csv').read_bytes()
        (source / 'twitter_train.csv').write_bytes(original + b'\n')
        try:
            validate_manifest(manifest_path)
            raise AssertionError('Source mutation was accepted')
        except ValueError:
            pass
    print('PASS: ten fresh workers, rejection of old 7/1/2 splits, arithmetic mean, resume and mutation detection')


def check_tenfold_only():
    import subprocess
    import sys
    import numpy as np
    import torch
    import train_virta as training
    import evaluate_virta_errors as independent
    from prepare_tenfold import protocol
    from sklearn.model_selection import KFold

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'protocol': 'stratified_5fold_70_10_20', 'folds': [{}] * 5}))
        prefix = root / 'best.pth'
        cv.save(str(prefix) + '.cv.json', {'folds': 5, 'output': str(root)})
        actions = [lambda: protocol(5), lambda: rotation([[]] * FOLDS, 0, folds=5),
                   lambda: prepare(root, root / 'invalid', 'twitter', folds=5),
                   lambda: validate_manifest(manifest),
                   lambda: cv.run(SimpleNamespace(cv_count=5)),
                   lambda: cv.summarize(root, prefix, folds=5),
                   lambda: cv.export_saved_predictions(prefix, root / 'errors.csv')]
        for action in actions:
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError('Unsupported fold protocol was accepted')
        assert not (root / 'invalid').exists() and not (root / 'errors.csv').exists()
        Path(str(prefix) + '.cv.json').unlink()
        cv.save(str(prefix) + '.config.json', {'cv_count': 5})
        try:
            independent.main(SimpleNamespace(checkpoint=prefix, output=root / 'errors.csv'))
        except ValueError:
            pass
        else:
            raise AssertionError('Independent evaluation accepted an unsupported fold checkpoint')
        recorded = root / 'fold_00'
        cv.save(str(prefix) + '.config.json', {'cv_count': FOLDS, 'data_root': str(recorded),
                                              'cv_splits': str(root / 'manifest.json')})
        try:
            independent.main(SimpleNamespace(checkpoint=prefix, output=root / 'errors.csv',
                                              data_root=str(root / 'source')))
        except ValueError:
            pass
        else:
            raise AssertionError('Independent evaluation accepted another fold data root')
        with patch.object(independent, 'configure_data_root') as configure, \
             patch.object(independent, 'build_dataset', side_effect=RuntimeError('stop before inference')), \
             patch.object(independent.torch, 'load', return_value={}):
            try:
                independent.main(SimpleNamespace(checkpoint=prefix, output=root / 'errors.csv',
                                                  data_root=None, dataset='weibo', cpu=True))
            except RuntimeError as error:
                assert str(error) == 'stop before inference'
            configure.assert_called_once_with('weibo', str(recorded))
        rejected = subprocess.run([sys.executable, '-B', str(Path(training.__file__)), '--cv-count', '5'],
                                  capture_output=True, text=True)
        assert rejected.returncode == 2 and 'invalid choice' in rejected.stderr
    print('PASS: CLI, split generation, workers, summaries and evaluation reject unsupported folds; fold data auto-selected')

    model = SimpleNamespace(risk_coefficients=torch.zeros(5), risk_intercept=torch.tensor(0.),
                            risk_threshold=torch.tensor(.5))
    features = np.random.default_rng(666).normal(size=(20, 5))
    targets = np.asarray([0, 1] * 10)
    metrics = {'risk_features': features, 'error_targets': targets}
    with patch.object(training, 'evaluate', side_effect=lambda *a, **kw: dict(metrics)), \
         patch('sklearn.model_selection.KFold', wraps=KFold) as splitter:
        result = training.calibrate_risk(model, None, torch.device('cpu'))
        assert result['folds'] == FOLDS and result['error_f1_cross_fitted'] is not None
        assert splitter.call_args.kwargs['n_splits'] == FOLDS
        metrics = {'risk_features': features[:5], 'error_targets': targets[:5]}
        splitter.reset_mock()
        result = training.calibrate_risk(model, None, torch.device('cpu'))
        assert result['folds'] == 0 and result['error_f1_cross_fitted'] is None
        splitter.assert_not_called()
    print('PASS: risk cross-fitting uses ten folds or explicitly skips insufficient samples')



def check_pool_gradient():
    import torch
    from alignment import _pool_weights, _pool_one
    torch.manual_seed(666)
    for height, width in [(3, 7), (141, 180), (197, 256)]:
        original = torch.randn(height, width, requires_grad=True)
        other = original.detach().clone().requires_grad_(True)
        gradient = torch.randn(40, 40)
        expected = torch.nn.functional.adaptive_avg_pool2d(original[None], (40, 40))[0]
        actual = _pool_weights(height, other.device, other.dtype) @ other @ _pool_weights(width, other.device, other.dtype).T
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        (expected * gradient).sum().backward()
        (actual * gradient).sum().backward()
        torch.testing.assert_close(original.grad, other.grad, rtol=1e-5, atol=1e-5)
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
            value = original.detach().cuda().requires_grad_(True)
            pooled = _pool_one(value)
            native = torch.nn.functional.adaptive_avg_pool2d(value.detach()[None], (40, 40))[0]
            torch.testing.assert_close(pooled, native, rtol=0, atol=0)
            (pooled * gradient.cuda()).sum().backward()
            torch.testing.assert_close(value.grad.cpu(), original.grad, rtol=1e-5, atol=1e-5)
    print('PASS: deterministic pooling preserves native forward and analytical gradients, including upsampling')


if __name__ == '__main__':
    check_rotation()
    check_orchestration()
    check_tenfold_only()
    check_pool_gradient()
