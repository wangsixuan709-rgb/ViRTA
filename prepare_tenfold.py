"""Standard stratified ten-fold CV with duplicate-aware ten-bin membership."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import struct

FOLDS = 10
SPLITS = ('train', 'val', 'test')
PROTOCOL = 'stratified_10fold_80_10_10'
SPLIT_SEED = 666


def protocol(folds):
    if folds != FOLDS:
        raise ValueError('Only standard ten-fold 80/10/10 cross-validation is supported')
    return PROTOCOL


def sample_coverage(folds):
    protocol(folds)
    return {'train': 8, 'val': 1, 'test': 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint_image(path):
    from PIL import Image
    before = path.stat()
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as source:
        if source.mode == 'P' and 'transparency' in source.info:
            source = source.convert('RGBA')
        rgb = source.convert('RGB')
        info = {'file_sha256': hashlib.sha256(data).hexdigest(),
                'rgb_sha256': hashlib.sha256(struct.pack('>II', *rgb.size) + rgb.tobytes()).hexdigest()}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Image changed while hashing: {path}')
    return str(path), info


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def csv_name(dataset, split):
    return f'{split}.csv' if dataset == 'weibo' else f'{"gossip" if dataset == "gossipcop" else dataset}_{split}.csv'


def stratified_bins(rows, seed):
    """Keep exact duplicate news pairs intact, with exact per-bin class quotas."""
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row['pair_hash']].append(i)
    totals = Counter(row['label'] for row in rows)
    if set(totals) != {0, 1} or min(totals.values()) < FOLDS:
        raise ValueError('Ten-fold splitting needs at least ten samples of each binary class')
    # Stagger the two classes' remainders so total bin sizes differ by at most one.
    quotas = [[0, 0] for _ in range(FOLDS)]
    cursor = 0
    for label in (0, 1):
        for offset in range(FOLDS):
            quotas[(cursor + offset) % FOLDS][label] = totals[label] // FOLDS + (offset < totals[label] % FOLDS)
        cursor = (cursor + totals[label]) % FOLDS
    # Alternate large/small bins. For the two supplied pools this keeps the
    # eight-bin training size constant while validation/test differ by one row.
    quotas = [quotas[i] for pair in zip(range(5), range(5, 10)) for i in pair]
    remaining = [q.copy() for q in quotas]
    bins = [[] for _ in range(FOLDS)]
    rng = random.Random(seed)
    duplicates = [group for group in grouped.values() if len(group) > 1]
    rng.shuffle(duplicates)
    duplicates.sort(key=len, reverse=True)
    for group in duplicates:
        counts = [sum(rows[i]['label'] == k for i in group) for k in (0, 1)]
        choices = list(range(FOLDS))
        rng.shuffle(choices)
        choices = [f for f in choices if all(counts[k] <= remaining[f][k] for k in (0, 1))]
        if not choices:
            raise ValueError('An intact duplicate group cannot fit ten stratified bins; review the source pool')
        dest = max(choices, key=lambda f: sum(counts[k] * remaining[f][k] / quotas[f][k] for k in (0, 1)))
        bins[dest].extend(group)
        for k in (0, 1):
            remaining[dest][k] -= counts[k]
    for label in (0, 1):
        singles = [g[0] for g in grouped.values() if len(g) == 1 and rows[g[0]]['label'] == label]
        rng.shuffle(singles)
        offset = 0
        for fold in range(FOLDS):
            end = offset + remaining[fold][label]
            bins[fold].extend(singles[offset:end])
            offset = end
        assert offset == len(singles)
    for fold, indices in enumerate(bins):
        rng.shuffle(indices)
        assert [sum(rows[i]['label'] == k for i in indices) for k in (0, 1)] == quotas[fold]
    assert sorted(i for b in bins for i in b) == list(range(len(rows)))
    assert len({rows[i]['pair_hash'] for b in bins for i in b}) == len(grouped)
    owner = {}
    for fold, indices in enumerate(bins):
        for i in indices:
            assert owner.setdefault(rows[i]['pair_hash'], fold) == fold
    return bins


def rotation(bins, fold, folds=FOLDS):
    protocol(folds)
    if not 0 <= fold < folds or len(bins) != FOLDS:
        raise ValueError('Invalid fold index or ten-bin membership')
    test = [fold]
    val = [(fold + 1) % FOLDS]
    train = [b for b in range(FOLDS) if b not in test + val]
    return {name: [i for b in selected for i in bins[b]]
            for name, selected in [('train', train), ('val', val), ('test', test)]}


def validate_manifest(path, dataset=None, seed=None, verify_images=False, folds=FOLDS):
    protocol(folds)
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest['protocol'] != protocol(folds) or len(manifest['folds']) != folds:
        raise ValueError(f'Expected {protocol(folds)}; incompatible fold manifests cannot be used')
    if manifest.get('sample_coverage') != sample_coverage(folds):
        raise ValueError('Unexpected train/validation/test sample coverage')
    if manifest.get('split_algorithm') != 'stratified_balanced_bins_v2':
        raise ValueError('Regenerate the split with balanced integer remainders before starting experiments')
    if dataset is not None and manifest['dataset'] != dataset:
        raise ValueError('The fold manifest belongs to another dataset')
    if seed is not None and manifest['seed'] != seed:
        raise ValueError('The training seed and split seed must match')
    for name, expected in manifest['source_csv_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'Source data changed: {name}')
    for fold, record in enumerate(manifest['folds']):
        assert record['fold'] == fold
        for name, expected in record['csv_sha256'].items():
            if digest(Path(record['data_root']) / name) != expected:
                raise ValueError(f'Fold CSV changed: {fold}, {name}')
    if verify_images:
        fingerprints = json.loads((path.parent / 'image_fingerprints.json').read_text())
        for name, expected in fingerprints.items():
            stat = Path(name).stat()
            if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise ValueError(f'Source image changed since split preparation: {name}')
    return manifest


def prepare(source, output, dataset, seed=666, workers=8, folds=FOLDS):
    """Pool old train/val/test without changing labels; never overwrite old splits."""
    protocol(folds)
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = output / 'manifest.json'
    if manifest_path.is_file():
        current = validate_manifest(manifest_path, dataset, seed, verify_images=True, folds=folds)
        if current['source_root'] != str(source):
            raise ValueError('This cached split has a different source root')
        return manifest_path
    import pandas as pd
    output.mkdir(parents=True, exist_ok=False)
    rows, fields, sources = [], None, {}
    for split in SPLITS:
        path = source / csv_name(dataset, split)
        if not path.is_file():
            if split == 'val':
                continue
            raise FileNotFoundError(path)
        sources[str(path)] = digest(path)
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            raw_rows = list(reader)
            if fields is not None and reader.fieldnames != fields:
                raise ValueError('All source CSVs must have identical columns')
            fields = reader.fieldnames
        frame = pd.read_csv(path)
        assert len(raw_rows) == len(frame)
        for index, raw in enumerate(raw_rows):
            label = int(raw['label'])
            if label not in (0, 1):
                raise ValueError('Labels must be unchanged binary labels: 0=fake, 1=real')
            text_key = 'text' if 'text' in raw else 'content'
            text = frame.iloc[index][text_key]
            text = '' if pd.isna(text) else str(text)
            image_key = 'image' if 'image' in raw else 'images'
            images = []
            for name in raw[image_key].split('|'):
                image = Path(name)
                if not image.is_absolute():
                    image = (source / 'Images' / f'gossip_{split}' / image.name
                             if dataset == 'gossipcop' and len(image.parts) == 1 else source / image)
                images.append(str(image.resolve(strict=True)))
            raw[image_key] = '|'.join(images)
            rows.append({'raw': raw, 'label': label, 'images': images,
                         'source_split': split, 'source_row': index,
                         'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    total = Counter(row['label'] for row in rows)
    paths = sorted({name for row in rows for name in row['images']})
    fingerprints = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (name, info) in enumerate(pool.map(fingerprint_image, map(Path, paths)), 1):
            stat = Path(name).stat()
            fingerprints[name] = {**info, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if count % 2000 == 0 or count == len(paths):
                print('CV_HASHED_IMAGES', count, len(paths), flush=True)
    for row in rows:
        # The dataset loader consumes the FIRST image; identity must match that behavior.
        row['pair_hash'] = row['text_sha256'] + ':' + fingerprints[row['images'][0]]['rgb_sha256']
    bins = stratified_bins(rows, seed)
    membership = []
    for number, indices in enumerate(bins):
        for i in indices:
            membership.append({k: rows[i][k] for k in ('label', 'pair_hash', 'source_split', 'source_row')}
                              | {'pool_index': i, 'bin': number})
    save(output / 'membership.json', membership)
    save(output / 'image_fingerprints.json', fingerprints)
    records, coverage = [], {s: Counter() for s in SPLITS}
    for fold in range(folds):
        folder = output / f'fold_{fold:02d}'
        folder.mkdir()
        assignment = rotation(bins, fold, folds)
        counts, hashes, pairs = {}, {}, {}
        for split, indices in assignment.items():
            coverage[split].update(indices)
            pairs[split] = {rows[i]['pair_hash'] for i in indices}
            counts[split] = {'fake': sum(rows[i]['label'] == 0 for i in indices),
                             'real': sum(rows[i]['label'] == 1 for i in indices), 'total': len(indices)}
            path = folder / csv_name(dataset, split)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[i]['raw'] for i in indices)
            hashes[path.name] = digest(path)
        assert sorted(i for indices in assignment.values() for i in indices) == list(range(len(rows)))
        assert not (pairs['train'] & pairs['val'] or pairs['train'] & pairs['test'] or pairs['val'] & pairs['test'])
        records.append({'fold': fold, 'data_root': str(folder), 'counts': counts, 'csv_sha256': hashes,
                      'test_bin': fold, 'validation_bin': (fold + 1) % FOLDS,
                      'exact_pair_cross_split_overlaps': 0})
    expected_coverage = sample_coverage(folds)
    for split, expected in expected_coverage.items():
        assert set(coverage[split][i] for i in range(len(rows))) == {expected}
    assert all(coverage['train'][i] + coverage['val'][i] == folds - 1 for i in range(len(rows)))
    for name, expected in sources.items():
        assert digest(name) == expected, 'Source CSV changed during split preparation'
    manifest = {'protocol': protocol(folds), 'split_algorithm': 'stratified_balanced_bins_v2',
                'seed': seed, 'dataset': dataset, 'source_root': str(source),
                'source_csv_sha256': sources, 'pool_rows': len(rows),
                'class_counts': {'fake': total[0], 'real': total[1]},
                'labels_changed': 0, 'rows_removed': 0, 'folds': records,
                'test_evaluations_per_trained_model': 1,
                'sample_coverage': expected_coverage,
                'duplicate_pair_groups_kept_intact': sum(n > 1 for n in Counter(r['pair_hash'] for r in rows).values()),
                'rounding': 'Stratified ten-bin integer quotas; per-fold counts can differ by a few rows from the table.',
                'group_identity': 'Exact loader text AND decoded RGB pixels of the first image; not event-level grouping.'}
    save(manifest_path, manifest)
    write_distribution(output, manifest)
    print('CV_SPLITS_READY', manifest_path, flush=True)
    return manifest_path


def write_distribution(output, manifest):
    """Use the exact Split / Real News / Fake News / Total layout requested."""
    output = Path(output)
    lines = [f'# {manifest["dataset"]}: {manifest["protocol"]}', '',
             'All samples keep their original labels. Each sample is tested exactly once.',
             'Integer rounding makes validation/test sizes differ by one row across folds.', '']
    rows = []
    names = {'train': 'Training Set', 'val': 'Validation Set', 'test': 'Test Set'}
    for fold in manifest['folds']:
        lines += [f'## Fold {fold["fold"] + 1}', '', '| Split | Real News | Fake News | Total |',
                  '|---|---:|---:|---:|']
        for split in SPLITS:
            counts = fold['counts'][split]
            lines.append(f'| {names[split]} | {counts["real"]} | {counts["fake"]} | {counts["total"]} |')
            rows.append({'fold': fold['fold'] + 1, 'Split': names[split],
                         'Real News': counts['real'], 'Fake News': counts['fake'], 'Total': counts['total']})
        lines += [f'| **Total** | **{manifest["class_counts"]["real"]}** | '
                  f'**{manifest["class_counts"]["fake"]}** | **{manifest["pool_rows"]}** |', '']
    (output / 'distribution.md').write_text('\n'.join(lines) + '\n')
    with (output / 'distribution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fold', 'Split', 'Real News', 'Fake News', 'Total'])
        writer.writeheader()
        writer.writerows(rows)
