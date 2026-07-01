# CTAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Competitive Token Attention Transformer (CTAT) for brain parcellation from multi-modal DTI — replacing DDParcel's CNN encoder with a sparsemax-attention Transformer encoder while keeping the 2.5D three-view fusion + CNN decoder.

**Architecture:** Hierarchical Transformer encoder (4 stages + bottleneck, windowed attention at stage 0) + CNN decoder (4 stages with bilinear upsample + token-to-spatial skip fusion) + deep supervision auxiliary heads. Core innovation: sparsemax replaces softmax in MHA for cross-modal token competition.

**Tech Stack:** Python 3.11, PyTorch 2.11, reuse DDParcel data pipeline (`data_loader/`, `normalize.py`, `models/losses.py`)

---

## File Map

| File | Purpose |
|------|---------|
| `models/cta_block.py` | sparsemax, entmax, SparsemaxAttention, CompetitiveFFN, CTABlock |
| `models/ctat_encoder.py` | PatchEmbed, ModalityEmbed, PatchMerging, WindowCTABlock, CTATEncoder |
| `models/ctat_decoder.py` | ConvBlock, DecoderStage, CTATDecoder |
| `models/ctat_network.py` | CTAT full network: encoder + decoder + deep supervision heads + classifier |
| `models/ctat_solver.py` | Training loop with α-entmax annealing schedule |
| `scripts/train_ctat.py` | Training entry point per view (axial/coronal/sagittal) |
| `scripts/infer_ctat.py` | Three-view inference + DDParcel-compatible post-processing |

---

### Task 1: sparsemax + entmax + SparsemaxAttention + CompetitiveFFN + CTABlock

**Files:**
- Create: `models/cta_block.py`

- [ ] **Step 1: Write complete cta_block.py**

```python
# models/cta_block.py
"""CTAT core: sparsemax attention + competitive FFN building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(logits, dim=-1):
    """Euclidean projection onto probability simplex. Produces sparse distributions.
    Reference: Martins & Astudillo, ICML 2016."""
    ndim = logits.ndim
    if dim != -1 and dim != ndim - 1:
        logits = logits.transpose(dim, ndim - 1)
    z_sorted, _ = torch.sort(logits, dim=-1, descending=True)
    cssv = z_sorted.cumsum(dim=-1) - 1.0
    n = logits.size(-1)
    k = torch.arange(1, n + 1, dtype=logits.dtype, device=logits.device)
    cond = z_sorted > cssv / k
    k_z = cond.sum(dim=-1, keepdim=True).to(logits.dtype)
    tau = cssv.gather(-1, (k_z - 1).long().clamp(min=0)) / k_z.clamp(min=1.0)
    tau = tau.squeeze(-1)
    tau = torch.where(k_z.squeeze(-1) > 0, tau, torch.full_like(tau, float('inf')))
    output = torch.clamp(logits - tau.unsqueeze(-1), min=0)
    if dim != -1 and dim != ndim - 1:
        output = output.transpose(dim, ndim - 1)
    return output


def entmax(logits, alpha=1.5, dim=-1, n_iter=50):
    """alpha-entmax: softmax(alpha=1) to sparsemax(alpha=2).
    Reference: Peters et al., ACL 2019."""
    if alpha == 1.0:
        return F.softmax(logits, dim=dim)
    if alpha == 2.0:
        return sparsemax(logits, dim=dim)
    ndim = logits.ndim
    if dim != -1 and dim != ndim - 1:
        logits = logits.transpose(dim, ndim - 1)
    tau_min = logits.max(dim=-1, keepdim=True).values - 1.0
    tau_max = logits.max(dim=-1, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        p = torch.clamp((alpha - 1) * (logits - tau), min=0) ** (1.0 / (alpha - 1))
        sum_p = p.sum(dim=-1, keepdim=True)
        tau_min = torch.where(sum_p > 1.0, tau, tau_min)
        tau_max = torch.where(sum_p <= 1.0, tau, tau_max)
    p = torch.clamp((alpha - 1) * (logits - tau), min=0) ** (1.0 / (alpha - 1))
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    if dim != -1 and dim != ndim - 1:
        p = p.transpose(dim, ndim - 1)
    return p


class SparsemaxAttention(nn.Module):
    """Multi-head self-attention with entmax replacing softmax.
    alpha=1.0 -> softmax (dense), alpha=2.0 -> sparsemax (competitive)."""
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, alpha=2.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.alpha = alpha
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = entmax(attn, alpha=self.alpha, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class CompetitiveFFN(nn.Module):
    """FFN with channel-level sparsemax gate — sparse channel activation."""
    def __init__(self, dim, mlp_ratio=4, drop=0.0, alpha=2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.alpha = alpha
        self.gate = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden_dim, dim), nn.Dropout(drop),
        )

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        gate = entmax(self.gate(x), alpha=self.alpha, dim=-1)
        return x + gate * self.mlp(x)


class CTABlock(nn.Module):
    """Competitive Token Attention Block: LN->SparsemaxAttn->LN->C-FFN, with residuals."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, attn_drop=0., proj_drop=0.,
                 ffn_drop=0., alpha=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparsemaxAttention(dim, num_heads, attn_drop, proj_drop, alpha)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = CompetitiveFFN(dim, mlp_ratio, ffn_drop, alpha)

    def set_alpha(self, alpha):
        self.attn.set_alpha(alpha)
        self.ffn.set_alpha(alpha)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 2: Run unit test**

```bash
python3 -c "
import torch; import sys; sys.path.insert(0,'models')
from cta_block import sparsemax, entmax, CTABlock

# Test sparsemax
t = torch.randn(3, 10)
s = sparsemax(t)
assert torch.allclose(s.sum(-1), torch.ones(3), atol=1e-5)
assert (s == 0).any(), 'should have zeros'
print(f'sparsemax OK, sparsity={(s==0).float().mean():.2%}')

# Test entmax
for a in [1.0, 1.5, 2.0]:
    e = entmax(t, alpha=a)
    assert torch.allclose(e.sum(-1), torch.ones(3), atol=1e-5)
print('entmax OK')

# Test CTABlock
b = CTABlock(dim=96, num_heads=8, alpha=2.0)
x = torch.randn(2, 1024, 96)
y = b(x); assert y.shape == x.shape
b.set_alpha(1.0); y2 = b(x)
assert not torch.equal(y, y2), 'alpha should change output'
print(f'CTABlock OK: in={x.shape} out={y.shape}')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/cta_block.py && git commit -m "feat: add sparsemax, entmax, SparsemaxAttention, CompetitiveFFN, CTABlock"
```

---

### Task 2: Encoder (PatchEmbed, ModalityEmbed, PatchMerging, WindowCTABlock, CTATEncoder)

**Files:**
- Create: `models/ctat_encoder.py`

- [ ] **Step 1: Write encoder**

```python
# models/ctat_encoder.py
"""CTAT hierarchical Transformer encoder with windowed attention at top resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cta_block import CTABlock


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
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.modality_embed = ModalityEmbed(num_modalities, embed_dim)

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
        for stage in [self.stage0, self.stage1, self.stage2, self.stage3, self.bottleneck]:
            for blk in stage:
                blk.set_alpha(alpha)

    def forward(self, x):
        B = x.shape[0]
        M = self.num_modalities
        # Split 28-channel input into 4 modalities x 7 slices
        x_mods = x.chunk(M, dim=1)
        tokens = torch.stack([self.patch_embed(xm) for xm in x_mods], dim=1)
        tokens = self.modality_embed(tokens)
        H = W = 256 // self.patch_size  # 64
        tokens = tokens.view(B, M * tokens.shape[2], self.embed_dim)

        # Stage 0: windowed attention
        for blk in self.stage0:
            tokens = blk(tokens, H, W)
        skip0 = tokens
        tokens, H, W = self.merge0(tokens, H, W)

        # Stage 1: global attention
        for blk in self.stage1:
            tokens = blk(tokens)
        skip1 = tokens
        tokens, H, W = self.merge1(tokens, H, W)

        # Stage 2
        for blk in self.stage2:
            tokens = blk(tokens)
        skip2 = tokens
        tokens, H, W = self.merge2(tokens, H, W)

        # Stage 3
        for blk in self.stage3:
            tokens = blk(tokens)
        skip3 = tokens
        tokens, H, W = self.merge3(tokens, H, W)

        # Bottleneck
        for blk in self.bottleneck:
            tokens = blk(tokens)

        return tokens, [skip0, skip1, skip2, skip3], [64,32,16,8], [64,32,16,8]
```

- [ ] **Step 2: Test**

```bash
python3 -c "
import torch; import sys; sys.path.insert(0,'models')
from ctat_encoder import CTATEncoder
enc = CTATEncoder(in_channels=7, num_modalities=4, embed_dim=96, alpha=2.0)
x = torch.randn(2, 28, 256, 256)
bn, skips, Hs, Ws = enc(x)
print(f'bottleneck: {bn.shape}')  # [2, 64, 1536]
for i,(s,h,w) in enumerate(zip(skips,Hs,Ws)):
    print(f'skip{i}: {s.shape}, grid={h}x{w}')
# skip0:[2,16384,96],64x64 / skip1:[2,4096,192],32x32 / skip2:[2,1024,384],16x16 / skip3:[2,256,768],8x8
enc.set_alpha(1.0); bn2,_,_,_ = enc(x)
assert not torch.equal(bn, bn2)
print('Encoder test passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/ctat_encoder.py && git commit -m "feat: add CTAT encoder with windowed attention"
```

---

### Task 3: Decoder

**Files:**
- Create: `models/ctat_decoder.py`

- [ ] **Step 1: Write decoder**

```python
# models/ctat_decoder.py
"""CTAT CNN decoder — token-to-spatial skip fusion + progressive upsampling to 256x256."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3->BN->ReLU->Conv3x3->BN, with residual."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class DecoderStage(nn.Module):
    """Upsample x2 -> concat skip -> 1x1 fusion -> ConvBlocks."""
    def __init__(self, in_ch, skip_ch, out_ch, num_blocks=2):
        super().__init__()
        self.fusion = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.blocks = nn.ModuleList([ConvBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class CTATDecoder(nn.Module):
    """
    CNN decoder upsampling tokens from 4x4 bottleneck to 256x256 feature maps.
    Grid path: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256.
    Encoder stage dims: [96, 192, 384, 768, 1536]
    """
    def __init__(self, encoder_stage_dims, decoder_dim=96, num_modalities=4):
        super().__init__()
        self.M = num_modalities
        dims = encoder_stage_dims  # [dim0, dim1, dim2, dim3, dim4]

        # Skip projection: M*dim_i -> decoder_dim via 1x1 conv
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(num_modalities * dims[0], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[1], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[2], decoder_dim, 1),
            nn.Conv2d(num_modalities * dims[3], decoder_dim, 1),
        ])

        # 4 decoder stages: 4->8, 8->16, 16->32, 32->64
        self.stage3 = DecoderStage(num_modalities * dims[4], decoder_dim, decoder_dim)
        self.stage2 = DecoderStage(decoder_dim, decoder_dim, decoder_dim)
        self.stage1 = DecoderStage(decoder_dim, decoder_dim, decoder_dim)
        self.stage0 = DecoderStage(decoder_dim, decoder_dim, 64)

    def _skip2spatial(self, tokens, proj, H, W):
        """[B, M*N, C] -> [B, decoder_dim, H, W]"""
        B = tokens.shape[0]
        return proj(tokens.view(B, self.M * tokens.shape[-1], H, W))

    def forward(self, bottleneck, skips, H_list, W_list):
        """
        bottleneck: [B, M*16, dim4=1536]
        skips: [skip0, skip1, skip2, skip3] at grids 64,32,16,8
        Returns: [B, 64, 256, 256]
        """
        B = bottleneck.shape[0]
        M, dim4 = self.M, bottleneck.shape[-1]

        # Bottleneck tokens -> spatial: [B, M*dim4, 4, 4]
        x = bottleneck.view(B, M, 16, dim4).view(B, M, 4, 4, dim4)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, M * dim4, 4, 4)

        # Convert skips to spatial (reverse order: deep->shallow)
        s3 = self._skip2spatial(skips[3], self.skip_projs[3], H_list[3], W_list[3])
        s2 = self._skip2spatial(skips[2], self.skip_projs[2], H_list[2], W_list[2])
        s1 = self._skip2spatial(skips[1], self.skip_projs[1], H_list[1], W_list[1])
        s0 = self._skip2spatial(skips[0], self.skip_projs[0], H_list[0], W_list[0])

        x = self.stage3(x, s3)   # 4x4 -> 8x8
        x = self.stage2(x, s2)   # 8x8 -> 16x16
        x = self.stage1(x, s1)   # 16x16 -> 32x32
        x = self.stage0(x, s0)   # 32x32 -> 64x64

        # Final upsample: 64x64 -> 128x128 -> 256x256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x  # [B, 64, 256, 256]
```

- [ ] **Step 2: Test with encoder**

```bash
python3 -c "
import torch; import sys; sys.path.insert(0,'models')
from ctat_encoder import CTATEncoder
from ctat_decoder import CTATDecoder

enc = CTATEncoder(in_channels=7, num_modalities=4, embed_dim=96)
x = torch.randn(2, 28, 256, 256)
bn, skips, Hs, Ws = enc(x)

dec = CTATDecoder(enc.stage_dims, decoder_dim=96, num_modalities=4)
feat = dec(bn, skips, Hs, Ws)
print(f'Decoder output: {feat.shape}')  # [2, 64, 256, 256]
assert feat.shape == (2, 64, 256, 256)
print('Decoder test passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/ctat_decoder.py && git commit -m "feat: add CTAT CNN decoder with skip fusion"
```

---

### Task 4: Full CTAT Network with Deep Supervision

**Files:**
- Create: `models/ctat_network.py`

- [ ] **Step 1: Write network**

```python
# models/ctat_network.py
"""CTAT full network: encoder + decoder + deep supervision heads + classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ctat_encoder import CTATEncoder
from ctat_decoder import CTATDecoder


class CTAT(nn.Module):
    """
    Competitive Token Attention Transformer for brain parcellation.
    
    Input:  [B, 28, 256, 256]  (4 modalities x 7 thick slices)
    Output: [B, 82, 256, 256]  logits for 82 FreeSurfer classes
    
    Deep supervision: auxiliary classifier heads at decoder stages 1, 2, 3.
    Returns main logits + list of auxiliary logits for training.
    """
    def __init__(self, num_classes=82, in_channels=7, num_modalities=4, embed_dim=96,
                 num_heads=8, window_size=8, depths=[2,2,2,6], mlp_ratio=4,
                 alpha=2.0):
        super().__init__()
        self.encoder = CTATEncoder(in_channels, num_modalities, embed_dim,
                                   num_heads, window_size, depths, mlp_ratio, alpha)
        self.decoder = CTATDecoder(self.encoder.stage_dims, decoder_dim=embed_dim,
                                   num_modalities=num_modalities)
        
        # Main classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
        # Deep supervision heads (attached to decoder stage outputs)
        # Decoder stage internal dims: stage3,stage2,stage1,stage0 = [96,96,96,64]
        self.ds_heads = nn.ModuleList([
            self._make_ds_head(96, num_classes),   # after stage3 (8x8)
            self._make_ds_head(96, num_classes),   # after stage2 (16x16)
            self._make_ds_head(96, num_classes),   # after stage1 (32x32)
        ])
    
    def _make_ds_head(self, in_ch, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, num_classes, 1),
        )
    
    def set_alpha(self, alpha):
        self.encoder.set_alpha(alpha)
    
    def forward(self, x, return_aux=True):
        """
        Args:
            x: [B, 28, 256, 256]
            return_aux: if True, return (main_logits, aux_logits_list)
        Returns:
            main_logits: [B, num_classes, 256, 256]
            aux_logits: list of [B, num_classes, 8x8], [B, num_classes, 16x16], [B, num_classes, 32x32]
        """
        bottleneck, skips, H_list, W_list = self.encoder(x)
        
        # Decoder internally tracks stage outputs for deep supervision
        B = bottleneck.shape[0]
        M = self.encoder.num_modalities
        dim4 = bottleneck.shape[-1]
        
        # Bottleneck -> spatial
        feat = bottleneck.view(B, M, 16, dim4).view(B, M, 4, 4, dim4)
        feat = feat.permute(0, 1, 4, 2, 3).reshape(B, M * dim4, 4, 4)
        
        # Skip projections
        s3 = self.decoder._skip2spatial(skips[3], self.decoder.skip_projs[3], H_list[3], W_list[3])
        s2 = self.decoder._skip2spatial(skips[2], self.decoder.skip_projs[2], H_list[2], W_list[2])
        s1 = self.decoder._skip2spatial(skips[1], self.decoder.skip_projs[1], H_list[1], W_list[1])
        s0 = self.decoder._skip2spatial(skips[0], self.decoder.skip_projs[0], H_list[0], W_list[0])
        
        aux_logits = []
        
        feat = self.decoder.stage3(feat, s3)   # 4->8, 96ch
        aux_logits.append(self.ds_heads[0](feat))
        
        feat = self.decoder.stage2(feat, s2)   # 8->16, 96ch
        aux_logits.append(self.ds_heads[1](feat))
        
        feat = self.decoder.stage1(feat, s1)   # 16->32, 96ch
        aux_logits.append(self.ds_heads[2](feat))
        
        feat = self.decoder.stage0(feat, s0)   # 32->64, 64ch
        
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # 64->128
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)  # 128->256
        
        main_logits = self.classifier(feat)  # [B, 82, 256, 256]
        
        # Upsample aux logits to match main logits size
        aux_logits_upsampled = []
        target_size = (256, 256)
        scales = [32, 16, 8]  # each aux is at 1/scale of target
        for i, aux in enumerate(aux_logits):
            aux_up = F.interpolate(aux, size=target_size, mode='bilinear', align_corners=False)
            aux_logits_upsampled.append(aux_up)
        
        if return_aux:
            return main_logits, aux_logits_upsampled
        return main_logits
```

- [ ] **Step 2: Test full network**

```bash
python3 -c "
import torch; import sys; sys.path.insert(0,'models')
from ctat_network import CTAT

net = CTAT(num_classes=82, in_channels=7, num_modalities=4, embed_dim=96, alpha=2.0)
x = torch.randn(2, 28, 256, 256)
main, aux = net(x)
print(f'Main logits: {main.shape}')  # [2, 82, 256, 256]
for i, a in enumerate(aux):
    print(f'Aux {i}: {a.shape}')     # [2, 82, 256, 256] each
net.set_alpha(1.0)
main2, _ = net(x)
assert not torch.equal(main, main2)
print('CTAT full network test passed')
# Parameter count
params = sum(p.numel() for p in net.parameters())
print(f'Total parameters: {params:,}')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/ctat_network.py && git commit -m "feat: add CTAT full network with deep supervision"
```

---

### Task 5: Training Solver with Alpha Annealing

**Files:**
- Create: `models/ctat_solver.py`
- Modify: `models/losses.py` (reuse DiceLoss, CrossEntropy2D, CombinedLoss)

- [ ] **Step 1: Write solver**

```python
# models/ctat_solver.py
"""CTAT training loop with alpha-entmax annealing schedule."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.losses import DiceLoss, CrossEntropy2D
from models.ctat_network import CTAT


class CombinedLoss(nn.Module):
    """Dice + weighted CE, copied from DDParcel losses.py pattern."""
    def __init__(self, weight_dice=1, weight_ce=1):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = CrossEntropy2D()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, pred, target, weight_map):
        target = target.long()
        if pred.is_cuda:
            target = target.cuda()
        soft_pred = F.softmax(pred, dim=1)
        dice_val = self.dice_loss(soft_pred, target).mean()
        ce_val = (self.ce_loss(pred, target) * weight_map).mean()
        return self.weight_dice * dice_val + self.weight_ce * ce_val, dice_val, ce_val


class AlphaScheduler:
    """
    Linear alpha annealing: alpha_start -> alpha_end over n_steps.
    alpha=1.0 = softmax (dense), alpha=2.0 = sparsemax (competitive).
    """
    def __init__(self, alpha_start=1.0, alpha_end=2.0, total_epochs=100, steps_per_epoch=1):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = total_epochs * steps_per_epoch
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def get_alpha(self):
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress


class CTATSolver:
    """Training loop for CTAT with deep supervision and alpha annealing."""
    def __init__(self, model, train_loader, val_loader=None, lr=1e-4, weight_decay=0.05,
                 alpha_start=1.0, alpha_end=2.0, total_epochs=100,
                 ds_weights=[0.25, 0.5, 0.75],  # aux loss weights (deep->shallow)
                 device='cuda', exp_dir='./experiments/ctat'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.exp_dir = exp_dir
        self.total_epochs = total_epochs
        self.ds_weights = ds_weights
        
        os.makedirs(exp_dir, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6)
        self.alpha_scheduler = AlphaScheduler(alpha_start, alpha_end, total_epochs)
        self.loss_fn = CombinedLoss(weight_dice=1, weight_ce=1)
        self.best_val_dice = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            weights = batch['weight'].to(self.device) if 'weight' in batch else None
            
            # Alpha annealing step
            alpha = self.alpha_scheduler.step()
            self.model.set_alpha(alpha)
            
            self.optimizer.zero_grad()
            main_logits, aux_logits = self.model(images, return_aux=True)
            
            # Main loss
            main_loss, dice_val, ce_val = self.loss_fn(main_logits, labels, weights)
            loss = main_loss
            
            # Deep supervision auxiliary losses
            for i, aux_logit in enumerate(aux_logits):
                aux_loss, _, _ = self.loss_fn(aux_logit, labels, weights)
                loss = loss + self.ds_weights[i] * aux_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler_lr.step()
        avg_loss = total_loss / len(self.train_loader)
        current_alpha = self.alpha_scheduler.get_alpha()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{self.total_epochs} | Loss: {avg_loss:.4f} | "
              f"Alpha: {current_alpha:.2f} | LR: {current_lr:.2e}")
        return avg_loss

    def validate(self):
        self.model.eval()
        # Set alpha=2.0 for validation (full competition)
        self.model.set_alpha(2.0)
        total_dice = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                main_logits = self.model(images, return_aux=False)
                soft_pred = F.softmax(main_logits, dim=1)
                # Per-class dice
                pred = soft_pred.argmax(dim=1)
                dice_scores = []
                for c in range(1, soft_pred.size(1)):  # skip background
                    pred_c = (pred == c).float()
                    target_c = (labels == c).float()
                    inter = (pred_c * target_c).sum()
                    union = pred_c.sum() + target_c.sum()
                    if union > 0:
                        dice_scores.append((2 * inter / union).item())
                if dice_scores:
                    total_dice += sum(dice_scores) / len(dice_scores)
                n_batches += 1
        avg_dice = total_dice / max(n_batches, 1)
        print(f"Validation Dice: {avg_dice:.4f}")
        return avg_dice

    def train(self):
        for epoch in range(self.total_epochs):
            self.train_epoch(epoch)
            if self.val_loader and (epoch + 1) % 5 == 0:
                val_dice = self.validate()
                if val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_dice': val_dice,
                    }, os.path.join(self.exp_dir, 'best_model.pkl'))
                    print(f"Saved best model (Dice: {val_dice:.4f})")
        # Save final checkpoint
        torch.save({
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.exp_dir, 'final_model.pkl'))
```

- [ ] **Step 2: Commit**

```bash
git add models/ctat_solver.py && git commit -m "feat: add CTAT training solver with alpha annealing"
```

---

### Task 6: Training Entry Script

**Files:**
- Create: `scripts/train_ctat.py`

- [ ] **Step 1: Write training script**

```python
# scripts/train_ctat.py
"""Train CTAT for one view (axial, coronal, or sagittal)."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from models.ctat_network import CTAT
from models.ctat_solver import CTATSolver
from data_loader.load_neuroimaging_data import AsegDatasetWithAugmentation_Fused_Input
from data_loader.augmentation import ToTensor, AugmentationPadImage, AugmentationRandomCrop


def get_train_loader(hdf5_dir, view='coronal', batch_size=8, num_workers=4):
    """Build training DataLoader from HDF5 files using DDParcel's existing dataset."""
    import glob
    hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, '*.hdf5')))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_dir}")
    
    dataset = AsegDatasetWithAugmentation_Fused_Input(
        hdf5_files, view=view,
        transform=ToTensor(),
        pad_transform=AugmentationPadImage(),
        crop_transform=AugmentationRandomCrop(output_size=(256, 256))
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_dir', required=True, help='Directory with training HDF5 files')
    parser.add_argument('--view', default='coronal', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--num_classes', type=int, default=82)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--exp_dir', default='./experiments/ctat')
    parser.add_argument('--alpha_start', type=float, default=1.0)
    parser.add_argument('--alpha_end', type=float, default=2.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Training CTAT on {args.view} view, device={device}")

    # Dataset
    train_loader = get_train_loader(args.hdf5_dir, args.view, args.batch_size)

    # Model
    model = CTAT(
        num_classes=args.num_classes,
        in_channels=7,
        num_modalities=4,
        embed_dim=args.embed_dim,
        alpha=args.alpha_start,
    )

    # Solver
    solver = CTATSolver(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        total_epochs=args.epochs,
        device=device,
        exp_dir=f"{args.exp_dir}-{args.view}",
    )

    solver.train()
    print(f"Training complete. Model saved to {args.exp_dir}-{args.view}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_ctat.py && git commit -m "feat: add CTAT training entry script"
```

---

### Task 7: Inference Script

**Files:**
- Create: `scripts/infer_ctat.py`

- [ ] **Step 1: Write inference script**

```python
# scripts/infer_ctat.py
"""Three-view CTAT inference + DDParcel-compatible post-processing."""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import nibabel as nib
from models.ctat_network import CTAT
from data_loader.load_neuroimaging_data import (
    OrigDataThickSlices_Fused_Input,
    map_label2aparc_aseg,
    map_prediction_sagittal2full,
)


def load_model(checkpoint_path, num_classes, device):
    model = CTAT(num_classes=num_classes, in_channels=7, num_modalities=4, embed_dim=96)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    model.set_alpha(2.0)  # Full competition at inference
    return model


def run_view_inference(model, data_list, device, batch_size=16):
    """Run inference on one view, accumulate probabilities into pred_prob tensor."""
    dataset = OrigDataThickSlices_Fused_Input(data_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = 82
    pred_prob = np.zeros((256, 256, 256, num_classes), dtype=np.float32)
    
    slice_idx = 0
    with torch.no_grad():
        for batch in loader:
            images = batch.to(device)
            logits = model(images, return_aux=False)
            probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
            probs = probs.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
            for p in probs:
                if slice_idx < 256:
                    pred_prob[:, :, slice_idx, :] = p
                slice_idx += 1
    
    return pred_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axial_ckpt', help='Axial view checkpoint (.pkl)')
    parser.add_argument('--coronal_ckpt', help='Coronal view checkpoint (.pkl)')
    parser.add_argument('--sagittal_ckpt', help='Sagittal view checkpoint (.pkl)')
    parser.add_argument('--input_dir', required=True, help='Dir with 4 normalized .nii.gz DTI maps')
    parser.add_argument('--output', required=True, help='Output .mgz segmentation')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load 4 DTI scalar maps (FA, Trace, MinEig, MidEig)
    modalities = ['FractionalAnisotropy', 'Trace', 'MinEigenvalue', 'MidEigenvalue']
    data = []
    for mod in modalities:
        img = nib.load(os.path.join(args.input_dir, f'dti-{mod}-reg-NormMasked.nii.gz'))
        data.append(img.get_fdata().astype(np.float32))
        assert data[-1].shape == (256, 256, 256), f"Expected 256^3, got {data[-1].shape}"
    
    pred_prob = np.zeros((256, 256, 256, 82), dtype=np.float32)
    view_weights = {'axial': 0.4, 'coronal': 0.4, 'sagittal': 0.2}
    
    # Axial view
    if args.axial_ckpt:
        model = load_model(args.axial_ckpt, 82, device)
        # Prepare axial data (swap axes)
        axial_data = [np.moveaxis(d, [0,1,2], [1,2,0]) for d in data]
        pred = run_view_inference(model, axial_data, device)
        pred = np.moveaxis(pred, [1,2,0], [0,1,2])
        pred_prob += view_weights['axial'] * pred
    
    # Coronal view
    if args.coronal_ckpt:
        model = load_model(args.coronal_ckpt, 82, device)
        pred = run_view_inference(model, data, device)
        pred_prob += view_weights['coronal'] * pred
    
    # Sagittal view
    if args.sagittal_ckpt:
        sag_classes = 54  # Sagittal uses 54-class mapping
        model = load_model(args.sagittal_ckpt, sag_classes, device)
        sag_data = [np.moveaxis(d, [0,1,2], [2,1,0]) for d in data]
        pred = run_view_inference(model, sag_data, device)
        # Expand 54->82 classes
        pred_expanded = map_prediction_sagittal2full(pred)
        pred_expanded = np.moveaxis(pred_expanded, [1,2,0], [0,1,2])
        pred_prob += view_weights['sagittal'] * pred_expanded
    
    # Argmax and label remapping
    hard_labels = np.argmax(pred_prob, axis=-1)
    aseg = map_label2aparc_aseg(hard_labels)
    
    # Save as MGH
    img = nib.MGHImage(aseg.astype(np.int16), np.eye(4))
    nib.save(img, args.output)
    print(f"Saved segmentation to {args.output}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/infer_ctat.py && git commit -m "feat: add CTAT three-view inference script"
```

---

## Plan Completion Checklist

After all tasks are done, verify:
1. `models/cta_block.py` — all building blocks working
2. `models/ctat_encoder.py` — encoder produces correct shapes at each stage
3. `models/ctat_decoder.py` — decoder reconstructs 256x256 from 4x4 bottleneck
4. `models/ctat_network.py` — end-to-end forward pass [B,28,256,256] -> [B,82,256,256]
5. `models/ctat_solver.py` — training loop with alpha annealing runs
6. `scripts/train_ctat.py` — launches training from CLI
7. `scripts/infer_ctat.py` — three-view inference produces valid .mgz output
