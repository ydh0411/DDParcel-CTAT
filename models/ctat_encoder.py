# models/ctat_encoder.py
"""CTAT hierarchical Transformer encoder with windowed attention at top resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .cta_block import CTABlock, entmax
except ImportError:
    from cta_block import CTABlock, entmax


class PatchEmbed(nn.Module):
    """4x4 conv stride=4: [B, C, H, W] -> [B, (H/4)*(W/4), embed_dim]"""
    def __init__(self, in_channels=7, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class ModalityEmbed(nn.Module):
    """Learnable embedding per modality type (FA/Trace/MinEig/MidEig)."""
    def __init__(self, num_modalities=4, embed_dim=96):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(1, num_modalities, 1, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.embed


class ModalityCompetitiveFusion(nn.Module):
    """Same-location sparse competition across modalities.

    Input and output use [B, M, N, C], where each of the M modalities owns a
    token at the same spatial patch index N. The gate is normalized across M.
    """
    def __init__(self, dim, num_modalities=4, alpha=2.0, preserve_scale=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.alpha = alpha
        self.preserve_scale = preserve_scale
        self.score = nn.Linear(dim, 1)
        self.last_gate = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B, M, N, C] tokens, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {x.shape[1]}")

        scores = self.score(x).squeeze(-1)  # [B, M, N]
        gate = entmax(scores.transpose(1, 2), alpha=self.alpha, dim=-1)
        gate = gate.transpose(1, 2).unsqueeze(-1)  # [B, M, N, 1]
        self.last_gate = gate.detach()

        scale = self.num_modalities if self.preserve_scale else 1.0
        return x * gate * scale, gate


class PatchMerging(nn.Module):
    """2x2->1 spatial merge: concat 4 neighbors -> Linear(4*C_in -> C_out)."""
    def __init__(self, in_channels, out_channels, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        self.proj = nn.Linear(4 * in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, H, W):
        B, MN, C = x.shape
        M = self.num_modalities
        x = x.view(B, M, H, W, C)
        x = x.view(B, M, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, M, (H//2)*(W//2), 4*C)
        x = x.view(B, M*(H//2)*(W//2), 4*C)
        return self.norm(self.proj(x)), H // 2, W // 2


class WindowCTABlock(nn.Module):
    """CTABlock with window-partitioned attention for stage 0 (16K tokens -> 8x8 windows)."""
    def __init__(self, dim, num_heads=8, window_size=8, mlp_ratio=4, alpha=2.0):
        super().__init__()
        self.window_size = window_size
        self.cta = CTABlock(dim, num_heads, mlp_ratio, alpha=alpha)

    def set_alpha(self, alpha):
        self.cta.set_alpha(alpha)

    def forward(self, x, H, W):
        B, MN, C = x.shape
        M = MN // (H * W)
        wh, ww = self.window_size, self.window_size
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        x = x.view(B, M, H, W, C)
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, M, Hp//wh, wh, Wp//ww, ww, C)
        x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        n_win = (Hp//wh) * (Wp//ww)
        x = x.view(B * n_win, M * wh * ww, C)
        x = self.cta(x)
        x = x.view(B, n_win, M * wh * ww, C)
        x = x.view(B, Hp//wh, Wp//ww, M, wh, ww, C)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        x = x.view(B, M, Hp, Wp, C)
        if pad_h or pad_w:
            x = x[:, :, :H, :W, :]
        return x.reshape(B, M * H * W, C)


class CTATEncoder(nn.Module):
    """Hierarchical encoder: 4 stages + bottleneck, 4-modality input."""
    def __init__(self, in_channels=7, num_modalities=4, embed_dim=96,
                 num_heads=8, window_size=8, depths=[2,2,2,6], mlp_ratio=4,
                 alpha=2.0, patch_size=4):
        super().__init__()
        self.num_modalities = num_modalities
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.modality_embed = ModalityEmbed(num_modalities, embed_dim)
        num_patches = (256 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim))
        self.modality_competition = ModalityCompetitiveFusion(
            embed_dim, num_modalities=num_modalities, alpha=alpha)
        self.last_modality_gate = None

        self.stage0 = nn.ModuleList([
            WindowCTABlock(embed_dim, num_heads, window_size, mlp_ratio, alpha)
            for _ in range(depths[0])])
        self.merge0 = PatchMerging(embed_dim, embed_dim*2, num_modalities)

        dim1 = embed_dim * 2
        self.stage1 = nn.ModuleList([
            CTABlock(dim1, num_heads, mlp_ratio, alpha=alpha) for _ in range(depths[1])])
        self.merge1 = PatchMerging(dim1, dim1*2, num_modalities)

        dim2 = dim1 * 2
        self.stage2 = nn.ModuleList([
            CTABlock(dim2, num_heads, mlp_ratio, alpha=alpha) for _ in range(depths[2])])
        self.merge2 = PatchMerging(dim2, dim2*2, num_modalities)

        dim3 = dim2 * 2
        self.stage3 = nn.ModuleList([
            CTABlock(dim3, num_heads, mlp_ratio, alpha=alpha) for _ in range(depths[3])])
        self.merge3 = PatchMerging(dim3, dim3*2, num_modalities)

        dim4 = dim3 * 2
        self.bottleneck = nn.ModuleList([
            CTABlock(dim4, num_heads, mlp_ratio, alpha=alpha) for _ in range(2)])

        self.stage_dims = [embed_dim, dim1, dim2, dim3, dim4]

    def set_alpha(self, alpha):
        self.modality_competition.set_alpha(alpha)
        for stage in [self.stage0, self.stage1, self.stage2, self.stage3, self.bottleneck]:
            for blk in stage:
                blk.set_alpha(alpha)

    def forward(self, x):
        B = x.shape[0]
        M = self.num_modalities
        expected_channels = M * self.in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {x.shape[1]}")
        # Split 28-channel input into 4 modalities x 7 slices
        x_mods = x.chunk(M, dim=1)
        tokens = torch.stack([self.patch_embed(xm) for xm in x_mods], dim=1)
        tokens = self.modality_embed(tokens)
        tokens = tokens + self.pos_embed
        tokens, gate = self.modality_competition(tokens)
        self.last_modality_gate = gate.detach()
        H = W = 256 // self.patch_size  # 64
        tokens = tokens.view(B, M * tokens.shape[2], self.embed_dim)

        # Stage 0: windowed attention
        for blk in self.stage0:
            tokens = blk(tokens, H, W)
        skip0, H0, W0 = tokens, H, W
        tokens, H, W = self.merge0(tokens, H, W)

        # Stage 1: global attention
        for blk in self.stage1:
            tokens = blk(tokens)
        skip1, H1, W1 = tokens, H, W
        tokens, H, W = self.merge1(tokens, H, W)

        # Stage 2
        for blk in self.stage2:
            tokens = blk(tokens)
        skip2, H2, W2 = tokens, H, W
        tokens, H, W = self.merge2(tokens, H, W)

        # Stage 3
        for blk in self.stage3:
            tokens = blk(tokens)
        skip3, H3, W3 = tokens, H, W
        tokens, H, W = self.merge3(tokens, H, W)

        # Bottleneck
        for blk in self.bottleneck:
            tokens = blk(tokens)

        return tokens, [skip0, skip1, skip2, skip3], [H0, H1, H2, H3], [W0, W1, W2, W3]
