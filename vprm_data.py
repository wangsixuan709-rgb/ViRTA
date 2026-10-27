"""Reproducible clean pairs with global and smoothly varying local corruptions."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode
from vprm_corruption import corrupt

CORRUPTIONS = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
               'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'pixelate', 'jpeg_compression')
CLASSES = CORRUPTIONS + ('clean',)
PROBABILITIES = (.05, .25, .4, .25, .05)
LOCAL_DEGRADATION_PROBABILITY = 0.5
from clip_encoders import MEAN, STD


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def prepare(root, output, seed=42):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite split: {output}')
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'} and p.is_file())
    records, invalid, duplicates, seen = [], [], [], {}
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert('RGB'); im.load()
                # Pixel hashes also catch identical images in different file encodings.
                pixel_hash = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                size = im.size
            file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            if pixel_hash in seen:
                duplicates.append({'path': str(p.relative_to(root)), 'same_as': seen[pixel_hash]})
                continue
            seen[pixel_hash] = str(p.relative_to(root))
            records.append({'path': str(p.relative_to(root)), 'sha256': file_hash, 'size': size})
        except Exception as exc:
            invalid.append({'path': str(p), 'error': str(exc)})
    if len(records) < 10:
        raise ValueError('Need at least 10 valid unique images')
    random.Random(seed).shuffle(records)
    ntrain, nval = int(len(records)*.8), int(len(records)*.1)
    result = {'version': 1, 'root': str(root), 'seed': seed,
              'splits': {'train': records[:ntrain], 'val': records[ntrain:ntrain+nval], 'test': records[ntrain+nval:]},
              'invalid': invalid, 'duplicates_excluded': duplicates}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return {k: len(v) for k,v in result['splits'].items()}


def evaluation_plan(n, seed, rotation=0):
    """At 265: clean=17, ten corruptions=18 and four=17; quotas rotate."""
    rng = np.random.default_rng(seed)
    counts = np.full(15, n//15, dtype=int)
    order = [(i + rotation) % 14 for i in range(14)] + [14]
    for i in order[:n % 15]: counts[i] += 1
    plan = []
    for k, count in enumerate(counts):
        if k == 14:
            plan.extend([(k, 0)] * count); continue
        raw = np.array(PROBABILITIES) * count
        quota = np.floor(raw).astype(int)
        # Largest remainder with rotated tie order, then randomize sample assignment.
        priorities = sorted(range(5), key=lambda j: (-(raw[j]-quota[j]), (j-rotation-k)%5))
        for j in priorities[:count-int(quota.sum())]: quota[j] += 1
        for j, c in enumerate(quota): plan.extend([(k, j+1)] * c)
    rng.shuffle(plan)
    return plan


def apply_corruption(hq, kind, severity, resize_size, corruption_seed):
    small = TF.resize(hq, [resize_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    state = np.random.get_state()
    try:
        np.random.seed(corruption_seed)
        array = corrupt(small.permute(1,2,0).numpy(), severity=severity, corruption_name=CLASSES[kind])
    finally:
        np.random.set_state(state)
    return TF.resize(torch.from_numpy(array.copy()).permute(2,0,1), [hq.shape[-1]],
                     interpolation=InterpolationMode.BILINEAR, antialias=True)


def smooth_region_weights(resolution, rng):
    """Continuous curved regions with clean/light/heavy plateaus and smooth joins."""
    axis = torch.linspace(-1, 1, resolution)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    angle, phase = rng.uniform(0, 2*np.pi, size=2)
    along = float(np.cos(angle))*x + float(np.sin(angle))*y
    across = -float(np.sin(angle))*x + float(np.cos(angle))*y
    field = along + .15*torch.sin(2*across + float(phase))
    q0, q1, q2, q3 = torch.quantile(field, torch.tensor([.22, .33, .67, .78]))

    def smoothstep(low, high):
        t = ((field-low)/(high-low)).clamp(0, 1)
        return t.square()*(3-2*t)

    first, second = smoothstep(q0, q1), smoothstep(q2, q3)
    return torch.stack([1-first, first-second, second])


class VPRMPairs(Dataset):
    def __init__(self, manifest, split, round_id=0, seed=42, resolution=224,
                 local_degradation_probability=LOCAL_DEGRADATION_PROBABILITY):
        if not 0 <= local_degradation_probability <= 1:
            raise ValueError('local_degradation_probability must be in [0, 1]')
        manifest = Path(manifest).resolve()
        spec = json.loads(manifest.read_text())
        self.root, self.records = Path(spec['root']).expanduser(), spec['splits'][split]
        if not self.root.is_absolute():
            self.root = (manifest.parent / self.root).resolve()
        self.split, self.round_id, self.seed, self.resolution = split, round_id, seed, resolution
        self.local_degradation_probability = local_degradation_probability
        self.plan = None if split == 'train' else evaluation_plan(len(self.records), seed_for(seed, split, round_id, 'plan'), round_id)
        self.local_plan = np.zeros(len(self.records), dtype=bool)
        if self.plan is not None:
            mode_rng = np.random.default_rng(seed_for(seed, split, round_id, 'local-plan'))
            for kind in range(len(CORRUPTIONS)):
                indices = [i for i, pair in enumerate(self.plan) if pair[0] == kind]
                mode_rng.shuffle(indices)
                self.local_plan[indices[:int(len(indices)*local_degradation_probability)]] = True

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample_seed = seed_for(self.seed, self.split, self.round_id, record['path'])
        rng = np.random.default_rng(sample_seed)
        with Image.open(self.root / record['path']) as source: image = source.convert('RGB')
        r = self.resolution
        if self.split == 'train':
            if min(image.size) < r:
                image = TF.resize(image, r, interpolation=InterpolationMode.BICUBIC)
            left = int(rng.integers(image.width-r+1)); top = int(rng.integers(image.height-r+1))
            image = TF.crop(image, top, left, r, r)
            if rng.random() < .5: image = TF.hflip(image)
            kind = int(rng.integers(15))
            severity = int(rng.choice(np.arange(1,6), p=PROBABILITIES)) if kind != 14 else 0
        else:
            image = TF.center_crop(TF.resize(image, 256, interpolation=InterpolationMode.BILINEAR), [r,r])
            kind, severity = self.plan[index]
        hq = TF.pil_to_tensor(image)
        resize_size = r if kind == 14 else int(rng.integers(r//4, r))
        corruption_seed = int(rng.integers(2**32-1))
        local_rng = np.random.default_rng(seed_for(sample_seed, 'local'))
        use_local = (local_rng.random() < self.local_degradation_probability if self.split == 'train'
                     else self.local_plan[index])
        if kind == 14:
            lq = hq.clone()
            mode = 'clean'
        elif use_local:
            light_severity = int(local_rng.integers(1, 3))
            severity = int(local_rng.integers(4, 6))
            light_size = int(local_rng.integers(3*r//4, r))
            resize_size = int(local_rng.integers(r//4, r//2))
            light = apply_corruption(hq, kind, light_severity, light_size, seed_for(corruption_seed, 'light'))
            heavy = apply_corruption(hq, kind, severity, resize_size, seed_for(corruption_seed, 'heavy'))
            weights = smooth_region_weights(r, local_rng)
            lq = (hq.float()*weights[0] + light.float()*weights[1] + heavy.float()*weights[2])
            lq = lq.round().clamp(0, 255).to(torch.uint8)
            mode = 'local'
        else:
            lq = apply_corruption(hq, kind, severity, resize_size, corruption_seed)
            mode = 'global'
        return {'hq': TF.normalize(hq.float()/255, MEAN, STD),
                'lq': TF.normalize(lq.float()/255, MEAN, STD), 'kind': kind, 'severity': severity,
                'path': record['path'], 'sample_seed': sample_seed,
                'corruption_seed': corruption_seed, 'resize_size': resize_size,
                'degradation_mode': mode}
