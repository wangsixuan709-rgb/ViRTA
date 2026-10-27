"""Visual Patch Restoration Module (VPRM), paper Sec. 3.2."""
import torch
from torch import nn
from nafnet_original import NAFBlock


class VisualPatchRestorationBranch(nn.Module):
    """Candidate R = A + F(A), with A = the sole front NAFBlock(X)."""

    def __init__(self, channels=768, groups=16):
        super().__init__()
        if groups <= 0 or channels % groups:
            raise ValueError('channels must be divisible by positive groups')
        hidden_channels = channels * 4
        self.pre_naf = NAFBlock(channels)
        self.conv_in = nn.Conv2d(channels, hidden_channels, 1)
        self.group_norm = nn.GroupNorm(groups, hidden_channels)
        self.group_conv = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1, groups=groups)
        self.gelu = nn.GELU()
        self.intra_group_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, 1, groups=groups),
        )
        self.pwconv = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x):
        a = self.pre_naf(x)
        h = self.conv_in(a)
        h = self.group_norm(h)
        h = self.group_conv(h)
        h = self.gelu(h)
        h = h * self.intra_group_attn(h)
        # Internal residual uses A before channel expansion; no tail NAFBlock.
        return a + self.pwconv(h)


class RMXBlock(nn.Module):
    """Group-wise channel mean/max descriptors from the paper's RMX Block."""

    def __init__(self, groups=16):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape
        grouped = x.reshape(b, self.groups, c // self.groups, h, w)
        return torch.cat([grouped.mean(dim=2), grouped.amax(dim=2)], dim=1)


class GroupSpatialGate(nn.Module):
    """One gate per input channel group and patch, preserving H and W."""

    def __init__(self, channels=768, groups=16):
        super().__init__()
        if groups <= 0 or channels % groups:
            raise ValueError('channels must be divisible by positive groups')
        self.groups = groups
        self.rmx = RMXBlock(groups)
        self.cross_group = nn.Conv2d(2 * groups, groups, 1)
        self.spatial = nn.Conv2d(groups, groups, 3, padding=1, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        descriptors = self.rmx(x)
        logits = self.cross_group(descriptors)
        return self.sigmoid(self.spatial(logits))


class VPRM(nn.Module):
    """Y = X + broadcast(M(X)) * (R(X) - X)."""

    def __init__(self, channels=768, groups=16):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.restoration = VisualPatchRestorationBranch(channels, groups)
        self.gate = GroupSpatialGate(channels, groups)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected [B, {self.channels}, H, W] features')
        restored = self.restoration(x)
        # Read original X, not the front NAFBlock output A.
        gate = self.gate(x)
        b, c, h, w = x.shape
        correction = (restored - x).reshape(
            b, self.groups, c // self.groups, h, w)
        gated_correction = correction * gate.to(correction.dtype).unsqueeze(2)
        return x + gated_correction.reshape_as(x)


STAGES = (4, 8, 12)
FORMAT = 'virta-vprm-clip-vit-b16-stages-v1'
LEGACY_FORMAT = 'restored-group-spatial-gate-clip-vit-b16-stages-v1'


def make_vprm():
    # One Visual Patch Restoration Module at each selected ViT block.
    return nn.ModuleDict({str(i): VPRM(768, 16) for i in STAGES})


def forward_vit(visual, image, vprm=None, checkpoint_segments=False):
    if image.ndim != 4 or tuple(image.shape[1:]) != (3,224,224):
        raise ValueError('VPRM ViT requires [B,3,224,224] CLIP-normalized images')
    if len(visual.transformer.resblocks) != 12:
        raise ValueError('Expected 12 ViT-B/16 transformer blocks')
    x = visual.conv1(image.to(visual.conv1.weight.dtype)).flatten(2).transpose(1,2)
    cls = visual.class_embedding.to(x.dtype).expand(len(x),1,-1)
    x = visual.ln_pre(torch.cat([cls,x],1) + visual.positional_embedding.to(x.dtype)).transpose(0,1)
    stages = []
    def run(module, value):
        if checkpoint_segments and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(module,value,use_reentrant=False)
        return module(value)
    for index, block in enumerate(visual.transformer.resblocks,1):
        x = run(block,x)
        if index in STAGES:
            tokens = x.transpose(0,1)
            patches = tokens[:,1:].transpose(1,2).reshape(len(tokens),768,14,14)
            if vprm is not None:
                patches = run(vprm[str(index)],patches)
                # CLS bypasses VPRM; restored patches feed the next transformer stage.
                tokens = torch.cat([tokens[:,:1],patches.flatten(2).transpose(1,2)],1)
                x = tokens.transpose(0,1)
            stages.append(patches)
    return visual.ln_post(x.transpose(0,1)), stages


def check_checkpoint(state, language):
    """Validate architecture/language and normalize the old restoration field."""
    if state.get('format') not in {FORMAT, LEGACY_FORMAT} or state.get('language') != language or tuple(state.get('stages',())) != STAGES:
        raise ValueError(f'Expected {language} VPRM ViT-B/16 checkpoint ({FORMAT}) at stages {STAGES}')
    if 'vprm' in state and 'cfrm' in state:
        raise ValueError('Conflicting legacy/current restoration fields')
    if 'vprm' not in state:
        if 'cfrm' not in state:
            raise ValueError('Missing VPRM weights')
        state['vprm'] = state.pop('cfrm')


class VPRMViTEncoder(nn.Module):
    def __init__(self, language, checkpoint_segments=False):
        super().__init__()
        from clip_encoders import CLIPEncoders
        encoders = CLIPEncoders(language)
        self.features = encoders.clip.visual
        self.language = language
        self.vprm = encoders.vprm
        self.features.requires_grad_(False).eval()
        self.checkpoint_segments = checkpoint_segments

    def train(self,mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward_stages(self,image,enable_vprm=True):
        return forward_vit(self.features,image,self.vprm if enable_vprm else None,
                           self.checkpoint_segments and self.training and enable_vprm)[1]

    def forward(self,image):
        tokens,_ = forward_vit(self.features,image,self.vprm,self.checkpoint_segments and self.training)
        return tokens,tokens[:,0],tokens[:,1:].transpose(1,2).reshape(len(tokens),768,14,14)

    def load_training_checkpoint(self,path):
        state = torch.load(path,map_location='cpu',weights_only=False)
        check_checkpoint(state,self.language)
        self.features.load_state_dict(state['backbone'])
        self.vprm.load_state_dict(state['vprm'])
        return self
