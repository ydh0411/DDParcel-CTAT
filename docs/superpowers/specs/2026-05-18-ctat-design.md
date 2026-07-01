# CTAT: Competitive Token Attention Transformer for Brain Parcellation from Diffusion MRI

> **UPDATED FRAMING (2026-06-05):** See `2026-06-05-ctat-modality-competitive-token-fusion.md`
> for the current research plan with ModalityCompetitiveFusion gate, safer claims,
> and comprehensive experimental design. This document retains the original
> architectural specification — the core encoder/decoder design is unchanged,
> but the motivation framing has been revised and the modality competition
> mechanism now uses a dedicated same-location gate (ModalityCompetitiveFusion).

## Metadata

- **Date:** 2026-05-18 (updated 2026-06-05)
- **Status:** Design approved; framing revised per 2026-06-05 research plan
- **Target venue:** MICCAI / IEEE TMI / MedIA
- **Base project:** DDParcel (Zhang et al., IEEE TMI 2024)

---

## 1. Motivation

DDParcel uses a CNN-based Competitive Dense Block U-Net with pixel-level maxout to fuse 4 DTI scalar maps for brain parcellation. Inspired by DDParcel's modality selection philosophy, CTAT explores whether a learned sparse attention mechanism (sparsemax) can replace the hard-coded maxout rule, learning more flexible, context-dependent modality selection patterns in a Transformer architecture.

**Core idea:** Replace softmax attention with sparsemax, so that attention weights become sparse — each query selectively attends to the most relevant tokens across modalities, implementing a form of budget-constrained attention allocation rather than dense weighted averaging.

---

## 2. Architecture Overview

### 2.1 Input

- 4 DTI scalar maps: FA, Trace, MinEig, MidEig
- Each: [256, 256, 256] after conform + z-score normalization
- 2.5D thick slices: 7 slices per modality → 28 channels concatenated
- Patch size: 4×4 → 64×64 = 4096 tokens per modality per slice
- After modality concatenation: 4 × 4096 = 16384 total tokens at stage 0
- Token dimension: C = 96
- Stage 0 uses window-partitioned attention (window size=8×8) to keep attention complexity manageable (16384² → 64² per window × 256 windows ≈ 1M per head). Stage 1+ uses global attention (≤4096 tokens).

### 2.2 Pipeline

```
[B, 28, 256, 256]  (4 modalities × 7 thick slices)
        │
        ▼
Modality Split + Patch Embedding (4 independent 4×4 conv, stride=4)
        │
  4 × [B, 4096, C]  tokens per modality
        │
        ▼
  + Modality Embedding (learnable, 4 types)
  + Position Embedding (learnable, 64×64 grid)
        │
        ▼
┌──────────────────────────────────────────┐
│  ModalityCompetitiveFusion                │
│  Same-location sparse gate over M=4       │
│  Score → entmax across modalities → gate  │
│  Output: [B, 4N₀, C] (modality-gated)     │
│  Exposes: last_modality_gate for diag.    │
└──────────────┬───────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Encoder Stage 0                          │
│  CTA Block × 2                            │
│  Output: [B, 4N₀, C], N₀ = 4096           │
│  Skip connection → decoder stage 1        │
└──────────────┬───────────────────────────┘
               │
               ▼  Patch Merging: tokens → [B,4N,H,W]; 2×2→1 patch concat → [B,4N/4,4C]; Linear(4C→2C)
┌──────────────────────────────────────────┐
│  Encoder Stage 1                          │
│  CTA Block × 2                            │
│  Output: [B, 4N₁, 2C], N₁ = 1024          │
│  Skip connection → decoder stage 2        │
└──────────────┬───────────────────────────┘
               │
               ▼  Patch Merging
┌──────────────────────────────────────────┐
│  Encoder Stage 2                          │
│  CTA Block × 2                            │
│  Output: [B, 4N₂, 4C], N₂ = 256           │
│  Skip connection → decoder stage 3        │
└──────────────┬───────────────────────────┘
               │
               ▼  Patch Merging
┌──────────────────────────────────────────┐
│  Encoder Stage 3                          │
│  CTA Block × 6                            │
│  Output: [B, 4N₃, 8C], N₃ = 64            │
│  Skip connection → decoder stage 4        │
└──────────────┬───────────────────────────┘
               │
               ▼  Patch Merging
┌──────────────────────────────────────────┐
│  Bottleneck                               │
│  CTA Block × 2                            │
│  Output: [B, 4N₄, 16C], N₄ = 16           │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Decoder (4 stages, CNN-based)            │
│  Each stage: Bilinear upsample ×2 →       │
│    Concat(skip from encoder) →            │
│    Fusion Conv(1×1) → ResBlock ×2         │
└──────────────┬───────────────────────────┘
               │
               ▼
Segmentation Head: 1×1 Conv(C → 82)
        [B, 82, 256, 256]
```

---

## 3. ModalityCompetitiveFusion + CTA Block

### 3.0 ModalityCompetitiveFusion (v3 — 2026-06-05)

Before the token attention blocks, a dedicated same-location competition gate
selects which modalities contribute at each spatial patch:

```
Input: [B, M, N, C]  (M=4 modalities, N=4096 patches, C=96 dim)

1. Per-token scoring:  score = Linear(C → 1)(x)  → [B, M, N]
2. Cross-modality normalization:  gate = entmax(score, dim=M, α)  → [B, M, N]
3. Gating:  x_gated = x * gate.unsqueeze(-1) * M  (scale preserved)
   Exposes: last_modality_gate = gate.detach()
```

This is the **primary modality competition mechanism**. The sparsemax attention
in the downstream CTABlocks handles spatial/cross-token reasoning, not modality
competition.

### 3.1 CTA Block Structure

```
Input: tokens_all [B, MN, C]  (M modalities concatenated, already gated)

1. Sparse Multi-Head Attention:
   Q, K, V = Linear(tokens_all)
   A = Q @ K^T / sqrt(d)
   A_sparse = entmax(A, dim=-1, α)     ← spatial/token-level sparsity
   attn_out = A_sparse @ V

2. Sparse-Gated Feed-Forward Network (SparseGatedFFN):
   x = LN(x + attn_out)
   gate = entmax(Linear(x), dim=-1)   ← channel-level sparsity
   x = x + gate * MLP(LN(x))
   x = LN(x)

Output: [B, 4N, C]
```

### 3.2 sparsemax Definition

```
sparsemax(z)_i = max(0, z_i - τ(z))
where τ(z) solves: Σ max(0, z_i - τ) = 1
```

Properties:
- Output is a sparse probability distribution (many entries = 0)
- Differentiable almost everywhere
- Natural implementation of "token competition" — below-threshold tokens receive zero weight

### 3.3 Why Competition?

DDParcel uses pixel-level maxout: at each spatial position, keep only the most
confident modality's feature. CTAT asks whether this idea can be generalized to
Transformer tokens:

1. **ModalityCompetitiveFusion** performs explicit same-location modality
   competition — the sparse gate selects which modalities contribute at each
   anatomical patch, directly analogous to DDParcel's maxout but learned rather
   than hard-coded.

2. **Sparse attention** in downstream CTABlocks provides spatial and cross-token
   reasoning with entmax — a continuum from softmax (dense, α=1) to sparsemax
   (competitive, α=2). The sparsity here is about token selection, not modality
   competition.

This two-level design separates modality competition (location-aligned gate)
from spatial reasoning (attention), making each mechanism independently
diagnosable.

### 3.4 α-entmax Flexibility

sparsemax is the α=2 special case of α-entmax:
- α=1: softmax (dense, no competition)
- α=1.5: intermediate sparsity
- α=2: sparsemax (strongest competition)

We use this to implement an annealing schedule during training.

---

## 4. Decoder

CNN-based decoder for spatial precision. 4 stages, each consisting of:
1. Bilinear upsample (×2)
2. Concatenate with skip connection from encoder
3. Fusion 1×1 Conv
4. 2× ResBlock (Conv3×3 → BN → ReLU → Conv3×3 → BN + residual)

Skip connection handling: encoder skip output is a token sequence [B, 4N, C_dim]. Before passing to decoder, it is reshaped to a 2D feature map:
  1. Reshape: [B, 4N, C_dim] → [B, 4×C_dim, sqrt(N), sqrt(N)]  (recover spatial layout)
  2. Fusion Conv: 1×1 Conv(4×C_dim → C) → [B, C, H, W]
This flatten-and-reshape is standard in hierarchical vision transformers.

---

## 5. Loss Function

### 5.1 Main Loss

```
L_main = DiceLoss(softmax(pred), target) + Mean(CrossEntropy(pred, target) * weight_map)
```

DiceLoss and CrossEntropy with equal weight (λ=1 each), following DDParcel's validated setup.

### 5.2 Deep Supervision

Auxiliary heads at each decoder stage:

```
L_total = L_main + 0.25 * L_aux_4 + 0.5 * L_aux_3 + 0.75 * L_aux_2
```

Auxiliary heads are 1×1 Conv → num_classes, upsampled to 256×256 for loss computation.

### 5.3 Weight Map

Per-pixel weight map combining:
1. Median frequency balancing (rare classes get higher weight, capped at 5×)
2. Gradient edge weighting (boundary pixels get 5× weight)

Preserved from original DDParcel.

---

## 6. Training Strategy

### 6.1 α-entmax Annealing

```
start: α = 1.0  (softmax, no sparsity)
schedule: linear per-batch interpolation from α=1.0 to α=2.0 over all training steps
end:   α = 2.0  (sparsemax, full competition)
```

α increases continuously at each optimizer step, not per-epoch. After `total_epochs × len(train_loader)` steps, α reaches exactly 2.0. The linear schedule is a heuristic; alternative schedules (cosine, step-function) may improve results and are left to future work.

Rationale: Early training benefits from dense attention for feature learning; later training sharpens to competitive attention for precise modality selection.

### 6.2 Three-View 2.5D Fusion

Same strategy as DDParcel — three separate model instances share the same architecture but have view-specific weights:
1. Train CTAT_axial on axial slices
2. Train CTAT_coronal on coronal slices
3. Train CTAT_sagittal on sagittal slices
4. Inference: run all three, weighted voting (axial:0.4, coronal:0.4, sagittal:0.2)
5. Each view's encoder weights are trained from scratch for that view (no weight sharing across views)

### 6.3 Optimization

- Optimizer: AdamW (weight_decay=0.05)
- Learning rate: 1e-4, cosine schedule, 5-epoch warmup
- Batch size: 16 per GPU
- Epochs: 100
- Data augmentation: random flip, random rotation (±10°), brightness/contrast jitter

### 6.4 Optional: Self-supervised Pretraining

Masked Image Modeling on unlabeled DTI data (HCP dataset). Mask 50% of input patch tokens, reconstruct. Finetune on parcellation labels.

---

## 7. Three-View Post-Processing

Preserved from DDParcel:
1. Per-view inference produces probability maps
2. Weighted sum: 0.4×axial + 0.4×coronal + 0.2×sagittal
3. Argmax → hard label map
4. Label remapping (internal index → FreeSurfer IDs)
5. Centroid-based hemisphere correction
6. Gaussian smoothing for ambiguous labels
7. Optional connected component cleanup

---

## 8. Ablation Experiments

| ID | Experiment | Purpose |
|----|-----------|---------|
| E1 | softmax vs sparsemax vs entmax(α=1.5) | Prove competitive attention is necessary |
| E2 | Fixed α vs α annealing | Prove annealing strategy is effective |
| E3 | No deep supervision vs with deep supervision | Prove auxiliary loss helps |
| E4 | Single modality vs 4-modality input | Prove multi-modal benefit |
| E5 | Cross-attention fusion vs CTA competition | **Core ablation**: competition vs collaboration |
| E6 | CTA blocks per stage: [1,1,3,1] vs [2,2,6,2] vs [3,3,9,3] | Optimal network depth |
| E7 | Patch size: 2 vs 4 vs 8 | Optimal token granularity |
| E8 | CTAT vs DDParcel vs Swin UNETR | Architecture comparison |

---

## 9. Feasibility Evidence

| Component | Supporting Papers |
|-----------|------------------|
| Sparsemax in Transformer attention | BaSFormer (IEEE TASLP 2024), MultiMax (ICML 2024), ASEntmax (2025) |
| Sparse attention in medical imaging | MedFormer (PMB 2025), TCSAFormer (arXiv 2025), EMOST (CBM 2024) |
| Multi-modal fusion with sparse attention | MSFT-Net (MedIA 2026), DeMoSeg (2024), CFCI-Net (2024) |
| GPU-optimized sparse attention | AdaSplash (ICML 2025 Oral) |
| DTI-based brain parcellation | DDParcel (IEEE TMI 2024), DDEvENet (arXiv 2025) |

**Novelty gap:** No existing work combines sparsemax-based competitive attention with multi-modal DTI fusion for brain parcellation. The combination of (a) sparsemax competition + (b) multi-modal transformer + (c) 2.5D multi-view fusion + (d) DTI brain parcellation is novel.

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Sparsemax may cause training instability with too-early sparsity | Medium | α-entmax annealing schedule; monitor gradient norms |
| Patch-based tokenization may lose fine boundary detail | Medium | Small patch size (4×4); CNN decoder for spatial refinement |
| Training cost higher than original DDParcel | Low | AdaSplash-optimized Triton kernels; rent GPU servers |
| MultiMax (ICML 2024) may claim overlapping novelty | Low | Our contribution is in the application + competition mechanism + complete segmentation system; cite MultiMax as related work |
| Sparsemax at α=2.0 may silently drop critical modality info in gray matter regions | Medium | Report per-class Dice for all 82 classes; visualize attention sparsity per brain region |
| Single-dataset (HCP) limits generalizability claims | Medium | Frame results as "on HCP data"; discuss clinical DTI limitations |
| Three-view 2.5D may not capture sufficient 3D context for all structures | Low | Ablate slice count (3, 5, 7); report per-class Dice for deep structures |
| C-FFN and attention sparsemax are entangled in ablation E1 | Medium | Add 2x2 factorial ablation: {softmax, sparsemax} attention × {standard, competitive} FFN |

---

## 11. Limitations and Assumptions

This design makes several assumptions that should be acknowledged in the final paper:

1. **FreeSurfer-as-gold-standard**: Training against FreeSurfer labels embeds FreeSurfer's systematic biases as a performance ceiling (Schoemaker et al., NeuroImage 2016). Dice scores measure agreement with FreeSurfer, not ground-truth anatomy.

2. **Modality sufficiency**: The four DTI scalars (FA, Trace, MinEig, MidEig) are assumed to contain all parcellation-relevant information, though they are mathematically coupled (Trace = λ₁+λ₂+λ₃). Raw DWI or additional derived maps may provide complementary information.

3. **Competition as budget-constrained allocation**: Sparsemax implements a budget-constrained projection onto the probability simplex — tokens below threshold receive exactly zero weight. This is a specific form of "competition" that differs from biological lateral inhibition. The paper should describe the mechanism precisely rather than relying on biological metaphor.

4. **HCP-only evaluation**: HCP subjects are young, healthy adults (22-35) scanned on a custom 3T Siemens Skyra. Performance on aging, pediatric, or clinical populations with lower-quality DTI is unknown.

5. **Linear annealing heuristic**: The linear α-entmax schedule is motivated by intuition ("dense early, sparse late"), not theory. The optimal schedule shape may depend on dataset, architecture, and task.

6. **Three-view fusion weights (0.4/0.4/0.2) are inherited from DDParcel** and not re-optimized for CTAT. The optimal per-view contribution may differ due to the Transformer's different inductive biases.

7. **The decoder is CNN-based**: The Transformer encoder serves as a feature extractor; pixel-level segmentation decisions are made by the CNN decoder. The contribution is in multi-modal feature extraction, not end-to-end Transformer segmentation.

---

## 12. References

1. Zhang et al. "DDParcel: deep learning anatomical brain parcellation from diffusion MRI." IEEE TMI, 2024.
2. Henschel et al. "FastSurfer -- A fast and accurate deep learning based neuroimaging pipeline." NeuroImage, 2020.
3. Martins & Astudillo. "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification." ICML, 2016.
4. Peters et al. "Sparse Sequence-to-Sequence Models." ACL, 2019.
5. Correia et al. "Adaptively Sparse Transformers." EMNLP, 2019.
6. Zhou et al. "MultiMax: Sparse and Multi-Modal Attention Learning." ICML, 2024.
7. Jiang et al. "BaSFormer: A Balanced Sparsity Regularized Attention Network for Transformer." IEEE TASLP, 2024.
8. Gonçalves et al. "AdaSplash: Adaptive Sparse Flash Attention." ICML, 2025.
9. Vasylenko et al. "Long-Context Generalization with Sparse Attention." arXiv:2506.16640, 2025.
10. Xia et al. "MedFormer: Hierarchical Medical Vision Transformer with Content-Aware Dual Sparse Selection Attention." PMB, 2025.
11. Yang et al. "DeMoSeg: Decoupling Feature Representations for Incomplete Multi-modal Brain Tumor Segmentation." 2024.
12. Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." BrainLes, 2021.
13. Zhou et al. "nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer." IEEE TIP, 2023.
14. Isensee et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods, 2021.
