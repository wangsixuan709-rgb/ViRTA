"""Rebuild fixed train/validation/test CSVs from content fingerprints, without relabeling."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    manifest = json.loads(args.manifest.read_text())
    if args.output_dir.exists():
        raise FileExistsError(f'Use a new output directory: {args.output_dir}')
    wanted = {(row['text_sha256'], row['label']) for rows in manifest['splits'].values() for row in rows}
    pool, image_hashes = defaultdict(deque), {}
    for source in args.inputs:
        frame = pd.read_csv(source)
        for index, row in frame.iterrows():
            text = row.get('text', row.get('content', ''))
            text = '' if pd.isna(text) else str(text)
            label = int(row['label'])
            if label not in (0, 1):
                raise ValueError(f'Invalid binary label in {source}, row {index}')
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if (text_hash, label) not in wanted:
                continue
            image = Path(str(row.get('image', row.get('images', ''))).split('|')[0])
            if not image.is_absolute():
                if manifest['dataset'] == 'gossipcop' and len(image.parts) == 1:
                    image = source.parent / 'Images' / source.stem / image.name
                else:
                    image = source.parent / image
            image = image.resolve()
            if not image.is_file():
                continue
            if image not in image_hashes:
                image_hashes[image] = file_digest(image)
            pool[(text_hash, image_hashes[image], label)].append((text, image))
    outputs = {}
    for split, records in manifest['splits'].items():
        rows = []
        for record in records:
            key = record['text_sha256'], record['image_sha256'], record['label']
            if not pool[key]:
                raise ValueError(f'Missing original text/image/label for {split}: {record["sample_id"]}')
            text, image = pool[key].popleft()
            rows.append({'news_id': record['sample_id'], 'text': text,
                         'image': os.path.relpath(image, args.output_dir.resolve()), 'label': record['label']})
        outputs[split] = rows
    args.output_dir.mkdir(parents=True)
    prefix = '' if manifest['dataset'] == 'weibo' else ('gossip_' if manifest['dataset'] == 'gossipcop' else 'twitter_')
    for split, rows in outputs.items():
        pd.DataFrame(rows, columns=['news_id', 'text', 'image', 'label']).to_csv(
            args.output_dir / f'{prefix}{split}.csv', index=False)
    print(json.dumps({'dataset': manifest['dataset'], 'rows': {k: len(v) for k, v in outputs.items()},
                      'labels_unchanged': True, 'image_paths': 'relative to output CSVs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args())
