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
    Reference: Peters et al., ACL 2019.
    Fast paths: softmax for alpha<=1.05, sparsemax for alpha>=1.95."""
    if alpha <= 1.05:
        return F.softmax(logits, dim=dim)
    if alpha >= 1.95:
        return sparsemax(logits, dim=dim)
    ndim = logits.ndim
    if dim != -1 and dim != ndim - 1:
        logits = logits.transpose(dim, ndim - 1)
    tau_max = logits.max(dim=-1, keepdim=True).values
    tau_min = tau_max - 1.0 / max(alpha - 1.0, 1e-10)
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


class SparseGatedFFN(nn.Module):
    """FFN with channel-level sparsemax gate — sparse channel activation.

    NOTE: This is NOT the modality competition mechanism. The modality competition
    happens in ModalityCompetitiveFusion (ctat_encoder.py). This class provides
    per-token channel sparsity inside the CTABlock's feed-forward sublayer."""
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
        return gate * self.mlp(x)


class CTABlock(nn.Module):
    """Competitive Token Attention Block: LN->SparsemaxAttn->LN->C-FFN, with residuals."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, attn_drop=0., proj_drop=0.,
                 ffn_drop=0., alpha=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparsemaxAttention(dim, num_heads, attn_drop, proj_drop, alpha)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SparseGatedFFN(dim, mlp_ratio, ffn_drop, alpha)

    def set_alpha(self, alpha):
        self.attn.set_alpha(alpha)
        self.ffn.set_alpha(alpha)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
