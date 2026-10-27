"""ViRTA: Visual Semantic Restoration and Adaptive Token Aggregation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
from collections import OrderedDict

import torch
from torch import Tensor, nn
from clip_encoders import CLIPEncoders, language_for
from alignment import ATM, FineGrainedAdaptiveFusionModule
from torch.utils.checkpoint import checkpoint


def paper_state_dict(state_dict):
    """Read pre-rename checkpoints strictly, without changing their tensors."""
    prefixes = (
        ('encoders.cfrm', 'encoders.vprm'),
        ('atrm_keep_ratio', 'atm_keep_ratio'), ('atrm', 'atm'),
        ('text_fusion_head', 'fgafm.text_feature_head'),
        ('image_fusion_head', 'fgafm.image_feature_head'),
        ('fusion_gate', 'fgafm.adaptive_fusion'), ('fusion_classifier', 'classifier'),
    )

    def rename(key):
        for old, new in prefixes:
            if key == old or key.startswith(old + '.'):
                return new + key[len(old):]
        return key

    result = OrderedDict()
    for key, value in state_dict.items():
        target = rename(key)
        if target in result:
            raise ValueError(f'Conflicting legacy/current checkpoint key: {target}')
        result[target] = value
    if hasattr(state_dict, '_metadata'):
        result._metadata = OrderedDict((rename(k), v) for k, v in state_dict._metadata.items())
    return result



def pairwise_js(probabilities: Tensor) -> Tensor:
    eps, values = 1e-8, []
    from itertools import combinations
    for left, right in combinations(range(probabilities.shape[1]), 2):
        mean = (probabilities[:, left] + probabilities[:, right]) * 0.5
        value = 0.5 * (
            (probabilities[:, left] * (probabilities[:, left].clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
            + (probabilities[:, right] * (probabilities[:, right].clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
        )
        values.append(value)
    return torch.stack(values, dim=-1)


class ViRTA(nn.Module):
    def __init__(self, dataset: str = "weibo", model_dim: int = 512,
                 branch_dim: int = 48, dropout: float = 0.2,
                 vprm_checkpoint: Optional[str] = None,
                 atm_keep_ratio: float = 0.5, text_chunk_batch_size: int = 4):
        super().__init__()
        if model_dim != 512:
            raise ValueError("CLIP joint embedding dimension must be 512")
        if not 0 < atm_keep_ratio <= 1:
            raise ValueError('atm_keep_ratio must be in (0,1]')
        if not isinstance(text_chunk_batch_size, int) or text_chunk_batch_size < 1:
            raise ValueError('text_chunk_batch_size must be a positive integer')
        self.text_chunk_batch_size = text_chunk_batch_size
        self.register_buffer('atm_keep_ratio', torch.tensor(atm_keep_ratio,dtype=torch.float64))
        self.encoders = CLIPEncoders(language_for(dataset))
        if vprm_checkpoint is not None:
            self.encoders.load_vprm(vprm_checkpoint)
        hidden_size = self.encoders.text_width
        self.text_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.image_projection = nn.Linear(768, model_dim, bias=False)
        with torch.no_grad():
            self.text_projection.weight.copy_(self.encoders.clip.text_projection.T)
            self.image_projection.weight.copy_(self.encoders.clip.visual.proj.T)
        self.text_slots = max(1,int((self.encoders.context_length-1)*atm_keep_ratio))
        self.atm = ATM(self.text_slots, max(1,int(196*atm_keep_ratio)))
        self.fgafm = FineGrainedAdaptiveFusionModule(hidden_size, model_dim, branch_dim, dropout)
        self.classifier = nn.Linear(branch_dim * 2 + 1600, 2)
        # Training-only supervision keeps weak branches alive; final inference uses classifier.
        self.auxiliary_branch_classifiers = nn.ModuleList([nn.Linear(d, 2) for d in (branch_dim, branch_dim, 1600)])
        self.register_buffer("risk_coefficients", torch.zeros(5))
        self.register_buffer("risk_intercept", torch.tensor(0.0))
        self.register_buffer("risk_threshold", torch.tensor(0.5))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(paper_state_dict(state_dict), strict=strict, assign=assign)

    def _encode_chunk_batch(self, ids, mask):
        tokens, pooled = self.encoders.encode_text(ids, mask)
        projected = self.text_projection(tokens)
        refined = self.atm.refine_text(projected, mask, self.encoders.language)
        return refined, pooled

    def encode_article(self, ids, mask):
        if ids.ndim == 2:
            ids, mask = ids[:,None], mask[:,None]
        if ids.ndim != 3 or ids.shape != mask.shape or ids.shape[-1] != self.encoders.context_length:
            raise ValueError('Expected matching [B,K,context_length] token ids and mask')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Text masks must be boolean or 0/1')
        mask = mask.bool()
        valid = mask.any(-1)
        if not valid.any(-1).all():
            raise ValueError('Every article needs at least one valid chunk')
        flat_ids, flat_mask = ids[valid], mask[valid]
        positions = torch.arange(ids.shape[-1], device=ids.device)[None]
        if not torch.equal(flat_mask, positions < flat_mask.sum(-1,keepdim=True)):
            raise ValueError('Chunk masks must be contiguous prefixes')
        if not (flat_mask.sum(-1) >= 2).all():
            raise ValueError('Every chunk needs start and end tokens')
        pieces, pooled = [], []
        # Bound encoder activation memory without truncating long articles.
        for start in range(0,len(flat_ids),self.text_chunk_batch_size):
            args = (flat_ids[start:start+self.text_chunk_batch_size],
                    flat_mask[start:start+self.text_chunk_batch_size])
            if self.training and torch.is_grad_enabled():
                refined, global_text = checkpoint(self._encode_chunk_batch,*args,use_reentrant=False)
            else:
                refined, global_text = self._encode_chunk_batch(*args)
            pieces.append(refined); pooled.append(global_text)
        refined, pooled = torch.cat(pieces), torch.cat(pooled)
        batch, chunks = valid.shape
        flat_indices = valid.flatten().nonzero().flatten()
        output = refined.new_zeros(batch*chunks,self.text_slots+1,512).index_copy(0,flat_indices,refined)
        globals_ = pooled.new_zeros(batch*chunks,pooled.shape[-1]).index_copy(0,flat_indices,pooled)
        globals_ = globals_.view(batch,chunks,-1).sum(1) / valid.sum(-1,keepdim=True)
        refined_mask = valid.unsqueeze(-1).expand(-1,-1,self.text_slots+1).reshape(batch,-1)
        return output.reshape(batch,-1,512), refined_mask, globals_

    def forward(self, input_ids: Optional[Tensor], attention_mask: Optional[Tensor],
                token_type_ids: Optional[Tensor], image: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required")
        refined_text, text_mask, pooled_text = self.encode_article(input_ids, attention_mask)
        image_regions, global_image, feature_map = self.encoders.encode_image(image)
        image_projected = self.image_projection(image_regions)
        refined_image = self.atm.refine_image(image_projected)
        fused_features, features, fusion = self.fgafm(
            refined_image, refined_text, text_mask, pooled_text, image_projected[:,0])
        weights = fusion['weights']
        logits = self.classifier(fused_features).relu()

        # Auxiliary heads are supervised during training but never used for the final prediction.
        branch_logits = torch.stack([head(features[index])
                                     for index, head in enumerate(self.auxiliary_branch_classifiers)], 1)
        branch_probs = torch.softmax(branch_logits, -1)
        fused_probs = torch.softmax(logits, -1)
        conflict_score = pairwise_js(branch_probs).mean(-1)
        fused_entropy = -(fused_probs * fused_probs.clamp_min(1e-8).log()).sum(-1) / torch.log(logits.new_tensor(2.0))
        gate_probabilities = weights / weights.sum(-1, keepdim=True)
        gate_entropy = -(gate_probabilities * gate_probabilities.clamp_min(1e-8).log()).sum(-1) / torch.log(logits.new_tensor(3.0))
        risk_features = torch.stack([1.0 - fused_probs.max(-1).values, fused_entropy, conflict_score,
                                     gate_entropy, 1.0 - weights.max(-1).values], -1)
        risk_score = torch.sigmoid(risk_features @ self.risk_coefficients + self.risk_intercept)
        aux = {**fusion,
               "branch_logits": branch_logits, "branch_probs": branch_probs,
               "conflict_score": conflict_score, "fused_entropy": fused_entropy, "gate_entropy": gate_entropy,
               "risk_features": risk_features, "risk_score": risk_score,
               "risk_flag": risk_score >= self.risk_threshold,
               "refined_text": refined_text, "refined_image": refined_image,
               "image_regions": image_regions, "image_feature_map": feature_map,
               "fused": logits}
        return logits, aux
