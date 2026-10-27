"""CPU workflow regression with synthetic encoders/data; downloads no model weights."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

import ViRTA as model_module
import evaluate_virta_errors as independent
import train_virta as training
from virta_dataset import collate_fn


class Encoders(nn.Module):
    def __init__(self, language, **kwargs):
        super().__init__()
        self.language = language
        self.text_width = 768 if language == 'zh' else 512
        self.context_length = 256 if language == 'zh' else 248
        self.clip = nn.Module()
        self.clip.visual = nn.Module()
        self.clip.visual.proj = nn.Parameter(torch.randn(768, 512) / 32)
        self.clip.text_projection = nn.Parameter(torch.randn(self.text_width, 512) / 32)
        self.clip.positional_embedding_res = nn.Parameter(torch.zeros(248, 512))
        self.vprm = nn.Linear(1, 1)

    def load_vprm(self, path):
        assert Path(path).is_file()

    def encode_text(self, ids, mask):
        basis = torch.arange(self.text_width, device=ids.device).float()[None, None] / 100
        tokens = torch.sin(ids.float()[:, :, None] / 50000 + basis)
        end = mask.sum(-1).long() - 1 if self.language == 'en' else torch.zeros(len(ids), dtype=torch.long)
        return tokens, tokens[torch.arange(len(ids)), end]

    def encode_image(self, image):
        signal = self.vprm(image.mean((1, 2, 3))[:, None])[:, None]
        tokens = torch.sin(signal + torch.arange(197 * 768).reshape(1, 197, 768) / 10000)
        return tokens, tokens[:, 0], tokens[:, 1:].transpose(1, 2).reshape(len(tokens), 768, 14, 14)


class News(torch.utils.data.Dataset):
    def __init__(self, dataset, split):
        self.dataset_name, self.split = dataset, split
        self.max_length = 256 if dataset == 'weibo' else 248
        self.csv_path = Path('synthetic') / f'{split}.csv'
        self.frame = pd.DataFrame({'news_id': range(4), 'text': ['sample'] * 4,
                                   'image': ['synthetic.jpg'] * 4, 'label': [0, 1, 0, 1]})

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        length = self.max_length
        ids = torch.zeros(1 + index % 2, length, dtype=torch.long)
        ids[:, :8] = torch.tensor([101, 5, 6, 7, 8, 9, 10, 102])
        if self.dataset_name != 'weibo':
            ids[:, 0], ids[:, 7] = 49406, 49407
        return {'input_ids': ids, 'attention_mask': ids.ne(0), 'token_type_ids': torch.zeros_like(ids),
                'image': torch.full((3, 224, 224), .1 * index), 'label': torch.tensor(index % 2),
                'news_id': str(index), 'token_count': torch.tensor(8),
                'chunk_spans': torch.tensor([[0, 6]]).expand(len(ids), -1)}


def arguments(output, checkpoint, dataset):
    # Read the real CLI parser without invoking main, so this checks current defaults.
    import ast
    tree = ast.parse(Path(training.__file__).read_text())
    guard = tree.body[-1]
    scope = {'argparse': argparse, 'FOLDS': training.FOLDS}
    exec(compile(ast.Module(body=guard.body[:-1], type_ignores=[]), 'train_virta.py', 'exec'), scope)
    return scope['parser'].parse_args(['--dataset', dataset, '--vprm-checkpoint', str(checkpoint),
                                      '--output', str(output), '--batch-size', '2', '--workers', '0',
                                      '--accumulation-steps', '2', '--epochs', '6',
                                      '--lr-warmup-epochs', '0', '--atm-keep-ratio', '0.1'])


def check_reference(path):
    alignment_spec = importlib.util.spec_from_file_location('reference_alignment', path.with_name('alignment.py'))
    previous_alignment = importlib.util.module_from_spec(alignment_spec)
    alignment_spec.loader.exec_module(previous_alignment)
    spec = importlib.util.spec_from_file_location('reference_model', path)
    reference = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'alignment': previous_alignment}):
        spec.loader.exec_module(reference)

    class LegacyEncoders(Encoders):
        def __init__(self, language):
            super().__init__(language)
            self.cfrm = self.vprm
            del self.vprm

        load_cfrm = Encoders.load_vprm

        def encode_image(self, image):
            signal = self.cfrm(image.mean((1, 2, 3))[:, None])[:, None]
            tokens = torch.sin(signal + torch.arange(197 * 768).reshape(1, 197, 768) / 10000)
            return tokens, tokens[:, 0], tokens[:, 1:].transpose(1, 2).reshape(len(tokens), 768, 14, 14)

    reference.CLIPEncoders = LegacyEncoders
    for dataset in ['weibo', 'gossipcop']:
        torch.manual_seed(42)
        old = reference.MMBE(dataset=dataset, cfrm_checkpoint=__file__, atrm_keep_ratio=.1).eval()
        new = model_module.ViRTA(dataset=dataset, atm_keep_ratio=.1).eval()
        new.load_state_dict(old.state_dict())
        batch = collate_fn([News(dataset, 'test')[0], News(dataset, 'test')[1]])
        before, a = training.model_forward(old, batch)
        after, b = training.model_forward(new, batch)
        for left, right in [(before, after)] + [(a[k], b[k]) for k in ['weights', 'S', 'fused_features', 'branch_logits']]:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        previous_loss, _ = training.fusion_classification_loss(before, a, batch['label'])
        current_loss, _ = training.fusion_classification_loss(after, b, batch['label'])
        torch.testing.assert_close(previous_loss, current_loss, rtol=0, atol=0)
    print('PASS: legacy full-model checkpoints load strictly; logits, features, weights and loss match exactly')

    restore_spec = importlib.util.spec_from_file_location('reference_restoration', path.with_name('cfrm_encoder.py'))
    previous_restore = importlib.util.module_from_spec(restore_spec)
    restore_spec.loader.exec_module(previous_restore)
    from vprm_encoder import VPRM, check_checkpoint, FORMAT, STAGES
    old = previous_restore.RestoredGatedBlock(32, 4)
    new = VPRM(32, 4)
    new.load_state_dict(old.state_dict())
    x = torch.randn(1, 32, 5, 5, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    before, after = old(x), new(y)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    before.square().mean().backward(); after.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for (_, left), (_, right) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    legacy = {'format': previous_restore.FORMAT, 'language': 'zh', 'stages': STAGES,
              'cfrm': old.state_dict()}
    check_checkpoint(legacy, 'zh')
    assert 'vprm' in legacy and 'cfrm' not in legacy
    check_checkpoint({'format': FORMAT, 'language': 'zh', 'stages': STAGES,
                      'vprm': new.state_dict()}, 'zh')
    print('PASS: VPRM outputs and gradients match exactly; legacy/current restoration checkpoints are accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert list(inspect.signature(model_module.ViRTA.forward).parameters) == [
        'self', 'input_ids', 'attention_mask', 'token_type_ids', 'image']
    real_evaluate = training.evaluate

    def evaluate(model, loader, device, *args, **kwargs):
        metrics = real_evaluate(model, loader, device, *args, **kwargs)
        if loader.dataset.split == 'val':
            metrics.update(macro_f1=.75, loss=.5)
        return metrics

    def create_model(**kwargs):
        model = model_module.ViRTA(**kwargs)
        with torch.no_grad():
            model.classifier.bias.add_(2)
        return model

    with patch.object(model_module, 'CLIPEncoders', Encoders):
        if args.reference:
            check_reference(args.reference)
        with tempfile.TemporaryDirectory(prefix='virta-full-workflow-') as directory:
            root = Path(directory)
            checkpoint = root / 'synthetic_restore.pt'
            checkpoint.touch()
            with patch.object(training, 'ViRTA', create_model), \
                 patch.object(independent, 'ViRTA', create_model), \
                 patch.object(training, 'build_dataset', lambda name, split, train: News(name, split)), \
                 patch.object(independent, 'build_dataset', lambda name, split, train: News(name, split)), \
                 patch.object(training, 'has_explicit_validation', lambda _: True), \
                 patch.object(training, 'evaluate', evaluate):
                for dataset, stopped in [('weibo', 4), ('gossipcop', 3)]:
                    output = root / dataset / 'best.pth'
                    training.train_one_fold(arguments(output, checkpoint, dataset))
                    epoch = json.loads(Path(str(output) + '.epoch.json').read_text())
                    assert epoch['epoch'] == 1 and epoch['trained_epochs'] == stopped
                    assert epoch['early_stopping'] and epoch['patience'] == stopped - 1
                    state = torch.load(output, weights_only=True)
                    assert any(k.startswith('encoders.vprm.') for k in state)
                    assert all('fgafm.adaptive_fusion.' + k in state for k in ['0.weight', '3.weight'])
                    errors = output.parent / 'errors.csv'
                    independent.main(SimpleNamespace(dataset=dataset, data_root=None, cpu=True,
                                                      checkpoint=output, output=errors, batch_size=3, workers=0))
                    before = json.loads(Path(str(output) + '.test_metrics.json').read_text())
                    after = json.loads(errors.with_suffix('.summary.json').read_text())
                    for key in ['accuracy', 'macro_f1', 'fake_precision', 'fake_recall', 'fake_f1',
                                'real_precision', 'real_recall', 'real_f1']:
                        assert abs(before[key] - after[key]) < 1e-12, (dataset, key)
                    print(f'PASS: {dataset} training, patience={stopped - 1}, checkpoint reload and independent test agree')
    # VPRM pretraining reuses the initialized restoration modules and frozen visual backbone.
    with patch('clip_encoders.CLIPEncoders', Encoders):
        from vprm_encoder import VPRMViTEncoder
        restored = VPRMViTEncoder('zh')
        assert isinstance(restored.vprm, nn.Linear)
        assert not any(p.requires_grad for p in restored.features.parameters())
    print('PASS: restoration pretraining still constructs a frozen visual backbone and trainable restoration')


if __name__ == '__main__':
    main()
