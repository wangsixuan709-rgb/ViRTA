"""Check smooth local augmentation and original global-pair reproducibility on CPU."""
import argparse
import importlib.util
import inspect
import sys
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import vprm_data as data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for seed in range(5):
        weights = data.smooth_region_weights(224, np.random.default_rng(seed))
        assert weights.shape == (3, 224, 224) and (weights >= 0).all() and (weights <= 1).all()
        torch.testing.assert_close(weights.sum(0), torch.ones(224, 224), rtol=0, atol=2e-7)
        assert all(float((w == 1).float().mean()) > .2 for w in weights)
        assert ((weights > 0) & (weights < 1)).any()
        torch.testing.assert_close(weights, data.smooth_region_weights(224, np.random.default_rng(seed)), rtol=0, atol=0)
    print('PASS: continuous weights have clean/light/heavy interiors, smooth joins and repeatable seeds', flush=True)

    with tempfile.TemporaryDirectory(prefix='vprm-local-check-') as directory:
        root = Path(directory)
        image = np.random.default_rng(42).integers(0, 256, (288, 288, 3), dtype=np.uint8)
        for i in range(3):
            Image.fromarray(image).save(root / f'scene{i}.png')
        records = [{'path': f'scene{i}.png'} for i in range(3)]
        spec = {'root': str(root), 'splits': {k: records for k in ['train', 'val', 'test']}}
        manifest = root / 'split.json'
        manifest.write_text(json.dumps(spec))
        ds = data.VPRMPairs(manifest, 'val', local_degradation_probability=1)
        captured = []
        original_weights = data.smooth_region_weights

        def weights(resolution, rng):
            result = original_weights(resolution, rng)
            captured.append(result)
            return result

        # Exercise the existing real corruption functions with both light/heavy settings.
        with patch.object(data, 'smooth_region_weights', weights):
            for kind in range(len(data.CORRUPTIONS)):
                ds.plan[0] = (kind, 3)
                ds.local_plan[0] = True
                before = np.random.get_state()
                pair = ds[0]
                after = np.random.get_state()
                assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
                assert pair['degradation_mode'] == 'local' and pair['severity'] in [4, 5]
                assert pair['hq'].shape == pair['lq'].shape == (3, 224, 224)
                assert torch.isfinite(pair['lq']).all()
                clean, light, heavy = [w == 1 for w in captured[-1]]
                assert torch.equal(pair['hq'][:, clean], pair['lq'][:, clean])
                assert not torch.equal(pair['hq'][:, light], pair['lq'][:, light])
                assert not torch.equal(pair['hq'][:, heavy], pair['lq'][:, heavy])
                repeated = ds[0]
                assert torch.equal(pair['hq'], repeated['hq']) and torch.equal(pair['lq'], repeated['lq'])
        print('PASS: all 14 real corruptions preserve clean pixels, alter light/heavy regions and preserve caller RNG', flush=True)
        ds.plan[0] = (14, 0)
        clean = ds[0]
        assert clean['degradation_mode'] == 'clean' and torch.equal(clean['hq'], clean['lq'])
        ds.plan[0] = (0, 3)
        local = ds[0]
        global_ds = data.VPRMPairs(manifest, 'val', local_degradation_probability=0)
        global_ds.plan[0] = (0, 3)
        global_pair = global_ds[0]
        assert global_pair['degradation_mode'] == 'global'
        assert torch.equal(local['hq'], global_pair['hq'])
        spec['root'] = '.'
        manifest.write_text(json.dumps(spec))
        relative = data.VPRMPairs(manifest, 'val', local_degradation_probability=0)
        assert relative.root == root
        torch.testing.assert_close(relative[0]['hq'], global_pair['hq'], rtol=0, atol=0)
        print('PASS: original clean target, whole-image corruption and relative manifest paths remain valid', flush=True)
        if args.reference:
            spec['root'] = str(root)
            manifest.write_text(json.dumps(spec))
            module_spec = importlib.util.spec_from_file_location('reference_vprm_data', args.reference)
            reference = importlib.util.module_from_spec(module_spec)
            with patch.dict(sys.modules, {'cfrm_corruption': importlib.import_module('vprm_corruption')}):
                module_spec.loader.exec_module(reference)
            reference_pairs = getattr(reference, 'VPRMPairs', None) or reference.CFRMPairs
            options = ({'local_degradation_probability': 0}
                       if 'local_degradation_probability' in inspect.signature(reference_pairs).parameters else {})
            for split in ['train', 'val', 'test']:
                old = reference_pairs(manifest, split, round_id=2, **options)
                new = data.VPRMPairs(manifest, split, round_id=2, local_degradation_probability=0)
                for index in range(2):
                    before, after = old[index], new[index]
                    for key in before:
                        if torch.is_tensor(before[key]):
                            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
                        else:
                            assert before[key] == after[key], (split, index, key)
            print('PASS: original global/clean sampling matches the previous implementation exactly', flush=True)
    print('ALL_LOCAL_DEGRADATION_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
