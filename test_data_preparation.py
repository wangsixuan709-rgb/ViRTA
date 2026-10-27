"""Check raw user data -> CSV -> ten folds and a freshly generated VPRM split."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image
import torch

from prepare_tenfold import prepare, validate_manifest
from virta_dataset import build_dataset, collate_fn, configure_data_root


def main():
    project = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')

    def run(script, *arguments, cwd=None):
        result = subprocess.run([sys.executable, '-B', str(project / script), *map(str, arguments)],
                                cwd=cwd, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        for dataset in ('weibo', 'gossipcop'):
            raw, output = root / dataset, root / f'{dataset}_source'
            if dataset == 'weibo':
                (raw / 'tweets').mkdir(parents=True)
                for category in ('rumor', 'nonrumor'):
                    images = raw / f'{category}_images'
                    images.mkdir()
                    for split in ('train', 'test'):
                        lines = []
                        for i in range(5):
                            name = f'{split}_{i}.png'
                            Image.new('RGB', (32, 40), (i * 20, 50, 90)).save(images / name)
                            lines.extend([f'{category}_{split}_{i}', f'https://example.com/{name}',
                                          f'图文数据准备检查 {category} {split} {i}'])
                        (raw / 'tweets' / f'{split}_{category}.txt').write_text('\n'.join(lines) + '\n')
                run('preprocess/weibo_preprocess.py', '--data-root', raw, '--output-dir', output, '--write')
            else:
                for split in ('train', 'test'):
                    images = raw / 'Images' / f'gossip_{split}'
                    images.mkdir(parents=True)
                    with (raw / f'gossip_{split}.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=['news_id', 'content', 'image', 'label'])
                        writer.writeheader()
                        for i in range(10):
                            name = f'{i}.png'
                            Image.new('RGB', (32, 40), (i * 20, 70, 110)).save(images / name)
                            writer.writerow({'news_id': f'{split}_{i}',
                                             'content': f'{split} {i} ' + 'An example news article. ' * 100,
                                             'image': name, 'label': 'fake' if i % 2 == 0 else 'real'})
                run('preprocess/gossip_preprocess.py', '--data-root', raw, '--output-dir', output, '--write')
            manifest_path = prepare(output, root / f'{dataset}_tenfold', dataset, workers=1)
            manifest = validate_manifest(manifest_path, dataset, 666, verify_images=True)
            assert manifest['pool_rows'] == 20 and manifest['class_counts'] == {'fake': 10, 'real': 10}
            assert manifest['sample_coverage'] == {'train': 8, 'val': 1, 'test': 1}
            assert len(manifest['folds']) == 10
            configure_data_root(dataset, manifest['folds'][0]['data_root'])
            data = build_dataset(dataset, 'test', False)
            batch = collate_fn([data[0], data[1]])
            assert batch['image'].shape == (2, 3, 224, 224) and torch.isfinite(batch['image']).all()
            assert batch['input_ids'].shape[-1] == (256 if dataset == 'weibo' else 248)
            if dataset == 'gossipcop':
                assert batch['input_ids'].shape[1] > 1
            print(f'PASS: {dataset} raw data, direct CSVs, user-sized ten folds and image/text loading')

        clean = root / 'clean-image'
        clean.mkdir()
        for i in range(10):
            Image.new('RGB', (32, 40), (i * 20, 120, 140)).save(clean / f'{i}.png')
        run('train_vprm.py', 'prepare', '--language', 'zh', '--root', clean, cwd=root)
        spec = json.loads((root / 'vprm_runs/split.json').read_text())
        assert spec['root'] == str(clean)
        assert {name: len(rows) for name, rows in spec['splits'].items()} == {'train': 8, 'val': 1, 'test': 1}
        print('PASS: VPRM prepare generates its default split from supplied images')


if __name__ == '__main__':
    main()
