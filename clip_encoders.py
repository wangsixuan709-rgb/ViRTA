"""Official English-CLIP-248 / Chinese-CLIP-256 encoders, with unprojected token sequences."""
from pathlib import Path
import os
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PRETRAINED = Path(os.environ.get('VIRTA_PRETRAINED_DIR',
                               os.environ.get('MMBE_PRETRAINED_DIR', ROOT / 'pretrained'))).expanduser()
if not (ROOT / 'vendor').is_dir() and (ROOT.parent / 'vendor').is_dir():
    ROOT = ROOT.parent
for source in (ROOT / 'vendor/English-CLIP', ROOT / 'vendor/Chinese-CLIP'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def language_for(dataset):
    if dataset not in {'weibo', 'gossipcop', 'twitter'}:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return 'zh' if dataset == 'weibo' else 'en'


def tokenize_article(text, language, overlap=48):
    """248 includes SOT/EOT: 246 content tokens, stride 198, no article cap."""
    if language == 'en':
        from model.english_clip import _tokenizer
        content = _tokenizer.encode(text)
        capacity, length, start_id, end_id = 246, 248, 49406, 49407
        if not 0 <= overlap < capacity:
            raise ValueError('overlap must be between 0 and 245')
        spans = []
        start = 0
        while True:
            end = min(start+capacity, len(content))
            spans.append((start,end))
            if end == len(content): break
            start += capacity-overlap
    elif language == 'zh':
        from cn_clip.clip import _tokenizer
        content = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))
        length, start_id, end_id = 256, 101, 102
        spans = [(0,min(254,len(content)))]
    else:
        raise ValueError('language must be en or zh')
    ids = torch.zeros(len(spans), length, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for k,(start,end) in enumerate(spans):
        values = [start_id] + content[start:end] + [end_id]
        ids[k,:len(values)] = torch.tensor(values)
        mask[k,:len(values)] = True
    return {'input_ids':ids, 'attention_mask':mask, 'token_type_ids':torch.zeros_like(ids),
            'chunk_spans':torch.tensor(spans), 'token_count':torch.tensor(len(content)+2)}


class CLIPEncoders(nn.Module):
    def __init__(self, language):
        super().__init__()
        self.language = language
        if language == 'en':
            from model.english_clip import load_from_clip
            self.clip, _ = load_from_clip('ViT-B/16', device='cpu',
                                          download_root=str(PRETRAINED))
            self.text_width, self.context_length = 512, 248
        elif language == 'zh':
            from cn_clip.clip import load_from_name
            self.clip, _ = load_from_name('ViT-B-16', device='cpu',
                                          download_root=str(PRETRAINED))
            self.text_width, self.context_length = 768, 256
        else:
            raise ValueError('language must be en or zh')
        self.clip.float()
        from vprm_encoder import make_vprm
        self.vprm = make_vprm()
        # Global CLIP projections are unused by this token-based classifier.
        self.clip.text_projection.requires_grad_(False)
        self.clip.visual.proj.requires_grad_(False)
        self.clip.logit_scale.requires_grad_(False)

    def load_vprm(self,path):
        from vprm_encoder import check_checkpoint
        state = torch.load(path,map_location='cpu',weights_only=False)
        check_checkpoint(state,self.language)
        self.clip.visual.load_state_dict(state['backbone'])
        self.vprm.load_state_dict(state['vprm'])

    def encode_text(self, ids, mask):
        if ids.ndim != 2 or ids.shape[1] != self.context_length:
            raise ValueError(f'Expected [B, {self.context_length}] text tokens')
        if self.language == 'en':
            tokens = self.clip.encode_text_full(ids)
            pooled = tokens[torch.arange(len(ids), device=ids.device), ids.argmax(-1)]
        else:
            tokens = self.clip.bert(ids, attention_mask=mask)[0]
            pooled = tokens[:, 0]
        return tokens, pooled

    def encode_image(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 224, 224):
            raise ValueError('Expected [B, 3, 224, 224] CLIP-normalized images')
        from vprm_encoder import forward_vit
        tokens,_ = forward_vit(self.clip.visual,image,self.vprm,self.training)
        return tokens,tokens[:,0],tokens[:,1:].transpose(1,2).reshape(len(tokens),768,14,14)
