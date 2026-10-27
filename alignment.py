"""ViRTA Adaptive Token Module and Fine-Grained Adaptive Fusion Module."""
import torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
from clip_encoders import ROOT  # Registers the vendored official modules.
from model.cross_net import TokenAggregation


class ATM(nn.Module):
    """Adaptive Token Module: retain the global token and aggregate local tokens."""
    def __init__(self, text_slots, image_slots=39):
        super().__init__()
        self.image = TokenAggregation(dim=512, keeped_patches=image_slots)
        self.text = TokenAggregation(dim=512, keeped_patches=text_slots)

    def refine_image(self, tokens):
        return torch.cat([tokens[:, :1], self.image(tokens[:, 1:])], dim=1)

    def refine_text(self, tokens, mask, language):
        # FineLIP keeps EOT outside aggregation. Chinese-CLIP keeps its global CLS.
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        end = mask.sum(-1).long() - 1
        global_index = end if language == 'en' else torch.zeros_like(end)
        aggregate_mask = mask & positions.ne(global_index[:, None])
        # Every valid input contains two special tokens, including empty text.
        if not aggregate_mask.any(-1).all():
            raise ValueError('ATM text chunks must contain start and end tokens')
        refined = self.text(tokens, keep_policy=aggregate_mask.to(tokens.dtype))
        global_token = tokens[torch.arange(len(tokens), device=tokens.device), global_index]
        return torch.cat([refined, global_token[:, None]], dim=1)


def fine_grained_matching(image, text, text_mask):
    """Only matched news pairs, not the retrieval batch-by-batch score matrix."""
    if (image.ndim != 3 or text.ndim != 3 or image.shape[0] != text.shape[0]
            or image.shape[-1] != text.shape[-1] or text_mask.shape != text.shape[:2]):
        raise ValueError('Invalid Fine-Grained Cross-Modal Matching token/mask shapes')
    text_mask = text_mask.bool()
    if not text_mask.any(-1).all():
        raise ValueError('Fine-Grained Cross-Modal Matching requires at least one valid text token per article')
    similarity = torch.bmm(F.normalize(image, dim=-1), F.normalize(text, dim=-1).transpose(1, 2))
    a = similarity.masked_fill(~text_mask[:, None], float('-inf')).amax(-1)
    b = similarity.amax(1).masked_fill(~text_mask, 0)
    rv = a.mean(-1)
    rt = b.sum(-1) / text_mask.sum(-1)
    return {'S': similarity.masked_fill(~text_mask[:, None], 0), 'a': a, 'b': b,
            'r_v': rv, 'r_t': rt, 'R': rv + rt, 'text_mask': text_mask}


@lru_cache(maxsize=256)
def _pool_weights(size, device, dtype):
    """Exact adaptive-average windows; constant matrices have no autograd state."""
    bins = torch.arange(40, device=device)
    start = bins * size // 40
    end = ((bins + 1) * size + 39) // 40
    position = torch.arange(size, device=device)
    return ((position[None] >= start[:, None]) & (position[None] < end[:, None])).to(dtype) / (end - start)[:, None]


def _pool_one(similarity):
    if (similarity.is_cuda and similarity.requires_grad and torch.is_grad_enabled()
            and torch.are_deterministic_algorithms_enabled()):
        # Preserve the native forward exactly. Its analytical gradient is a pair
        # of deterministic matrix products, avoiding CUDA's atomic pooling backward.
        native = F.adaptive_avg_pool2d(similarity.detach().unsqueeze(0), (40, 40)).squeeze(0)
        left = _pool_weights(similarity.shape[0], similarity.device, similarity.dtype)
        right = _pool_weights(similarity.shape[1], similarity.device, similarity.dtype)
        differentiable = left @ similarity @ right.T
        return native + (differentiable - differentiable.detach())
    return F.adaptive_avg_pool2d(similarity.unsqueeze(0), (40, 40)).squeeze(0)


def pool_similarity(similarity, text_mask):
    """Pool valid columns only; padding and other articles cannot change the result."""
    if similarity.ndim != 3 or similarity.shape[1] < 1 or text_mask.shape != (similarity.shape[0],similarity.shape[2]):
        raise ValueError('Expected S [B,P,M] and mask [B,M]')
    if not text_mask.bool().any(-1).all():
        raise ValueError('Each article needs valid correspondence columns')
    return torch.stack([_pool_one(s[:,mask.bool()])
                        for s,mask in zip(similarity,text_mask)])


def branch_head(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim),
                         nn.ReLU(), nn.Dropout(dropout))


class FineGrainedAdaptiveFusionModule(nn.Module):
    """Paper Sec. 3.4: token matching, bidirectional scores and adaptive fusion."""

    def __init__(self, text_width, model_dim=512, branch_dim=48, dropout=.2):
        super().__init__()
        self.text_feature_head = branch_head(text_width, branch_dim, dropout)
        self.image_feature_head = branch_head(model_dim, branch_dim, dropout)
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(2, branch_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(branch_dim, 3))

    def forward(self, image, text, text_mask, pooled_text, global_image):
        evidence = fine_grained_matching(image, text, text_mask)
        z_t = self.text_feature_head(pooled_text)
        z_v = self.image_feature_head(global_image)
        pooled_s = pool_similarity(evidence['S'], text_mask)
        z_m = pooled_s.flatten(1)
        features = (z_t, z_v, z_m)
        gate_input = torch.stack([evidence['r_v'], evidence['r_t']], -1)
        weights = torch.softmax(self.adaptive_fusion(gate_input), -1)
        weighted = [value * weights[:, i:i+1] for i, value in enumerate(features)]
        fused = torch.cat(weighted, -1)
        return fused, features, {
            **evidence, 'weights': weights, 'raw_weights': weights,
            'pooled_S': pooled_s, 'gate_input': gate_input,
            'weighted_branch_features': weighted, 'fused_features': fused,
            'text_branch': z_t, 'image_branch': z_v, 'multimodal_branch': z_m}
