# CTAT: Modality-Competitive Token Fusion for Direct dMRI Brain Parcellation

## One-Sentence Thesis

CTAT introduces **same-location modality-competitive token fusion** — a learned
sparse gate that selects which DTI modalities contribute at each anatomical
patch — followed by sparse-entmax spatial attention for cross-token reasoning.
It extends DDParcel's pixel-level maxout philosophy from CNN feature maps to
Transformer tokens, replacing a hard-coded competition rule with a learned one.

---

## Architecture (v3, 2026-06-05)

```
[B, 28, 256, 256]  ← 4 modalities × 7 thick slices
        │
        ▼
Patch Embed (4×4 conv, stride=4) per modality
        │
  4 × [B, 4096, 96]
        │
        ▼
+ Modality Embedding (learnable, 4 types)
+ Position Embedding (learnable, 64×64 grid)
        │
        ▼
┌──────────────────────────────────────┐
│ ModalityCompetitiveFusion             │  ← PRIMARY CONTRIBUTION
│  - Score per token (Linear→1)        │
│  - entmax across M=4 at same patch   │
│  - Gate * tokens * M (scale preserved)│
│  - Exposes last_modality_gate        │
└──────────────┬───────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Hierarchical Encoder (4 stages)       │
│  Stage 0: Windowed CTABlock ×2        │  ← 8×8 windows, 16K tokens
│  Stage 1: Global CTABlock ×2          │
│  Stage 2: Global CTABlock ×2          │
│  Stage 3: Global CTABlock ×6          │
│  Bottleneck: CTABlock ×2              │
│  PatchMerging between stages          │
└──────────────┬───────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ CNN Decoder (4 stages)                │
│  - Token→spatial skip projections     │
│  - Bilinear upsample + skip fusion    │
│  - Deep supervision heads             │
│  - Final: 2× upsample to 256×256      │
└──────────────┬───────────────────────┘
        │
        ▼
[B, 82, 256, 256]  ← FreeSurfer parcellation logits
```

### CTABlock internals

```
LN → SparseMHSA(entmax, α) → +residual
  → LN → SparseGatedFFN(entmax, α) → +residual
```

- **SparseMHSA**: entmax replaces softmax in QK^T attention. α=1.0 = softmax, α=2.0 = sparsemax.
- **SparseGatedFFN** (formerly CompetitiveFFN): Channel-level entmax gate on FFN output. NOT the modality competition mechanism — that happens in ModalityCompetitiveFusion.

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Modality competition BEFORE token attention | Separates "which modality?" from "which spatial context?" — independently diagnosable |
| α-entmax annealing (1.0 → 2.0, per-batch linear) | Dense early for feature learning, sparse late for competition |
| CNN decoder (not Transformer) | CNN inductive bias better for spatial precision; contribution is in feature extraction |
| Three-view 2.5D (axial/coronal/sagittal, 0.4/0.4/0.2) | Inherited from DDParcel for fair comparison; weights not re-optimized |
| 4×4 patches with windowed stage-0 attention | Balances token granularity with 16K-token memory constraints |
| preserve_scale = True (×M after gating) | Compensates for sparsemax zeroing some modalities, keeps gradient magnitude stable |

---

## File Map

```
DDParcel/
├── models/
│   ├── cta_block.py          ← sparsemax, entmax, SparseMHSA, SparseGatedFFN, CTABlock
│   ├── ctat_encoder.py       ← PatchEmbed, ModalityEmbed, ModalityCompetitiveFusion,
│   │                            PatchMerging, WindowCTABlock, CTATEncoder
│   ├── ctat_decoder.py       ← ConvBlock, DecoderStage, CTATDecoder
│   ├── ctat_network.py       ← CTAT (full encoder-decoder + deep supervision + classifier)
│   ├── ctat_solver.py        ← CombinedLoss, AlphaScheduler, CTATSolver
│   ├── networks.py           ← Original DDParcel CNN (baseline, unchanged)
│   ├── solver.py             ← Original DDParcel solver (baseline)
│   ├── losses.py             ← Original DDParcel losses (baseline)
│   └── sub_module.py         ← Original DDParcel building blocks (baseline)
├── scripts/
│   ├── diagnose_gate.py      ← Modality gate diagnostics (winner maps, entropy, per-region stats)
│   ├── train_ctat.py         ← Per-view training entry (subject-level split, MPS/CUDA/CPU)
│   ├── infer_ctat.py         ← Three-view inference + weighted voting
│   └── demo_ctat.py          ← 4-stage feasibility demo (sparsemax→annealing→forward→training)
├── docs/
│   ├── strategy/
│   │   └── CTAT-OVERVIEW.md  ← THIS FILE — complete strategy and architecture
│   └── superpowers/
│       ├── specs/
│       │   ├── 2026-05-18-ctat-design.md           ← Original design spec (architecture detail)
│       │   └── 2026-06-05-ctat-modality-competitive-token-fusion.md  ← Current research plan
│       └── plans/
│           └── 2026-05-19-ctat-implementation.md   ← Implementation plan (historical)
├── tests/
│   └── test_ctat_core.py     ← 8 tests: imports, shapes, gate, alpha schedule, modality order
├── essay/                    ← Reference papers (U-Net, DenseNet, FastSurferCNN, etc.)
├── ARCHITECTURE_GUIDE.md     ← Chinese-language DDParcel/UNet architecture primer
└── NETWORK_MODELS_GUIDE.md   ← Network architecture reference
```

---

## Research Question

> Does modality-aware token-level competitive fusion, implemented with
> alpha-entmax gates in CTAT, improve multi-modal dMRI brain parcellation
> over DDParcel-style pixel-level maxout fusion and matched dense-attention
> Transformer baselines under the same 2.5D three-view protocol?

---

## Primary Contributions

1. **ModalityCompetitiveFusion**: Sparse competition across DTI scalar maps at
   the same anatomical patch location before token attention. This is the core
   technical contribution.

2. **Sparse attention as spatial reasoning**: Alpha-entmax provides a continuum
   from dense softmax (α=1) to sparsemax (α=2), enabling annealed training and
   fixed-sparsity ablation experiments.

3. **Direct dMRI parcellation benchmark**: CTAT compared against DDParcel under
   matched preprocessing, label mapping, subject-level splits, and three-view
   inference.

4. **Mechanism diagnostics**: `last_modality_gate` enables quantitative analysis
   of which DTI maps are selected by anatomical region and view.
   `scripts/diagnose_gate.py` produces gate entropy maps, winner modality
   frequencies, and per-region modality usage statistics.

---

## Experimental Design

### Splits

- **Subject-level only** (never split by slice): all slices from a subject stay together
- 70% train / 10-15% validation / 15-20% locked test
- Split manifest saved as `split_manifest.json` for reproducibility
- Family-structured data: keep related subjects in the same split
- External cohort recommended if available (ABCD, UK Biobank, ADNI)

### Critical experiments (minimum for paper)

| ID | Experiment | What it proves |
|----|-----------|---------------|
| E1 | DDParcel vs CTAT | Overall method validity |
| E2 | Softmax vs entmax(1.5) vs sparsemax vs annealing | Competition mechanism matters |
| E3 | Plain concat attention vs ModalityCompetitiveFusion | Same-location gate is necessary |
| E4 | With vs without position embedding | Position info matters for parcellation |
| E5 | With vs without deep supervision | Auxiliary loss helps |
| E6 | 4 modalities vs leave-one-out | Each modality contributes |
| E7 | Single-view vs three-view | Multi-view fusion benefit |
| E8 | Missing/corrupted modality robustness | Clinical applicability |

### Baselines

- **Primary**: DDParcel (retrained on same split)
- **Secondary**: CTAT-softmax, CTAT-entmax(α=1.5), CTAT-sparsemax(α=2), CTAT-annealing
- **Ablation**: 28-channel concat U-Net (no modality competition), single-modality variants
- **Strongly recommended**: nnU-Net, Swin UNETR (if compute allows)

### Metrics

- **Primary**: Subject-level foreground macro Dice after label remapping
- **Secondary**: Per-region Dice, small-structure Dice, HD95, ASSD, surface Dice
- **Mechanism**: Gate entropy, winner modality frequency by region, active modalities per patch, attention sparsity by layer
- **Efficiency**: Parameters, FLOPs, inference time, GPU memory

### Statistical testing

- Unit of analysis: **subject**
- Primary test: paired permutation test or Wilcoxon signed-rank on per-subject macro Dice
- Report: mean/median paired differences, 95% bootstrap CI, proportion improved
- FDR or Holm correction for per-region comparisons

---

## Claims to Avoid (and why)

| Avoid | Why |
|-------|-----|
| "First sparsemax Transformer" | BaSFormer (2024), MultiMax (2024), Adaptively Sparse Transformers (2019) |
| "First multi-modal medical Transformer" | mmFormer (2022), MedFormer (2025), MSFT-Net (2026) |
| "Token-level generalizes pixel-level maxout" | Mathematically different objects; say "inspired by" |
| "State-of-the-art dMRI parcellation" | Unless beating DDParcel, DDEvENet, and nnU-Net |
| "Biological lateral inhibition" | Sparsemax is budget-constrained allocation, not neural inhibition |

### Safer claims

- "CTAT introduces modality-aware sparse competitive token fusion for direct dMRI parcellation"
- "Explores whether DDParcel-style competition can move from CNN feature maps to Transformer tokens"
- "Provides modality selection diagnostics for FA, Trace, MinEig, MidEig during parcellation"
- "ModalityCompetitiveFusion separates modality competition from spatial reasoning, enabling independent mechanism analysis"

---

## Expected Reviewer Attacks and Responses

| Attack | Response |
|--------|----------|
| "Sparse attention is not modality competition" | ModalityCompetitiveFusion is the competition mechanism; attention handles spatial reasoning. Gate winner maps prove modality selection. |
| "CTAT is just a large Transformer" | Report parameter/runtime tables. Include matched dense-attention CTAT baseline. |
| "Results may be leakage from slice-level splitting" | Use subject-level splits only. Save split manifests. |
| "The method doesn't match DDParcel's multi-expert design" | Frame as inspired by DDParcel. Optionally add modality-specific stems if needed. |
| "Single dataset (HCP) limits generalizability" | Acknowledge in limitations. Frame claims as "on HCP data." Add external cohort if possible. |

---

## Known Limitations (to acknowledge in paper)

1. **FreeSurfer-as-gold-standard**: Trains against FreeSurfer labels, embedding its biases
2. **HCP-only**: Young healthy adults, custom 3T scanner; unknown clinical generalizability
3. **Linear annealing heuristic**: Schedule shape not optimized; per-batch granularity is arbitrary
4. **CNN decoder**: Segmentation decisions are CNN-based; Transformer contribution is feature extraction
5. **Three-view weights inherited**: 0.4/0.4/0.2 from DDParcel, not re-optimized for CTAT
6. **No cross-atlas validation**: Only FreeSurfer 82-class; Desikan-Killiany or Destrieux untested
7. **Modal coupling**: FA/Trace/MinEig/MidEig are derived from same diffusion tensor

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| ModalityCompetitiveFusion | Done | `ctat_encoder.py:36-65` |
| Position embedding | Done | `ctat_encoder.py:135` |
| last_modality_gate | Done | Exposed for diagnostics |
| entmax bisection fix | Done | Lower bound corrected for α<2.0 |
| Dynamic H/W tracking | Done | No hardcoded resolutions |
| SparseGatedFFN rename | Done | Clarifies what competes where |
| Subject-level split | Done | With manifest export |
| MPS/CUDA/CPU support | Done | Auto-detect in train/infer/diagnose |
| Random seed + reproducibility | Done | `--seed` argument |
| Gate diagnostic script | Done | `scripts/diagnose_gate.py` |
| Train/val split in solver | Done | val_loader passed to CTATSolver |
| Baseline training configs | TODO | Need DDParcel/comparison configs |
| DDParcel post-processing | TODO | Inherit or remove claim |
| External cohort data | TODO | If available |
| Full HCP training | TODO | Needs compute |

---

## Quick Start

```bash
# Verify everything works
python -m pytest tests/test_ctat_core.py -v

# Run feasibility demo (requires testdata/)
python scripts/demo_ctat.py

# Train one view
python scripts/train_ctat.py --hdf5_dir /path/to/hdf5 --view coronal \
    --batch_size 8 --epochs 100 --exp_dir ./experiments/ctat

# Run gate diagnostics
python scripts/diagnose_gate.py --ckpt experiments/ctat-coronal/best_model.pkl \
    --data_dir testdata/HCP-100337-b1000 --output results/gate_diagnostics/
```

---

## References

- DDParcel: Zhang et al., IEEE TMI 2024 ([PMC10994696](https://pmc.ncbi.nlm.nih.gov/articles/PMC10994696/))
- Sparsemax: Martins & Astudillo, ICML 2016
- Entmax: Peters et al., ACL 2019
- Adaptively Sparse Transformers: Correia et al., EMNLP 2019
- MultiMax: Zhou et al., ICML 2024
- Swin UNETR: Hatamizadeh et al., BrainLes 2021
- nnU-Net: Isensee et al., Nature Methods 2021
- nnFormer: Zhou et al., IEEE TIP 2023
- MedFormer: Xia et al., PMB 2025
- DDEvENet: PubMed 39787735
