# CTAT Modality-Competitive Token Fusion Research Plan

## One-Sentence Thesis

CTAT should be framed as a modality-aware competitive token fusion method for direct dMRI brain parcellation: it extends DDParcel's pixel/feature-level maxout competition into same-location token-level modality competition, then uses sparse entmax attention for spatial and cross-token reasoning.

## Research Question

Does modality-aware token-level competitive fusion, implemented with alpha-entmax or sparsemax gates in CTAT, improve multi-modal dMRI brain parcellation over DDParcel-style pixel-level maxout fusion and matched dense-attention Transformer baselines under the same 2.5D three-view protocol?

## Why This Framing Is Safer

The weak framing is "we use sparsemax in a Transformer." Sparsemax, entmax, sparse Transformers, and multi-modal medical Transformers already exist. That claim is easy to attack.

The stronger framing is narrower and better tied to this project:

- DDParcel uses competitive maxout-style fusion for direct dMRI anatomical parcellation.
- CTAT asks whether this competition idea can move from CNN feature maps to tokens.
- The core mechanism is not generic sparse attention; it is same-location modality competition across FA, Trace, MinEig, and MidEig before spatial token attention.
- Sparse attention diagnostics become a mechanism study rather than a vague interpretability claim.

## Implemented Architecture Change

The earlier CTAT version concatenated all modality tokens and applied sparse attention over the combined sequence. That created sparse token selection, but it did not prove that modalities were competing with each other.

The current implementation adds a dedicated same-location modality competition gate:

1. Split input `[B, 28, 256, 256]` into 4 modalities, each with 7 thick-slice channels.
2. Patch embed each modality into `[B, M, N, C]`, where `M=4` and `N=64*64`.
3. Add learnable modality embeddings.
4. Add learnable spatial position embeddings.
5. Apply `ModalityCompetitiveFusion`:
   - score each modality token at the same spatial patch,
   - normalize scores across the modality dimension with alpha-entmax,
   - sparsely gate modality tokens before downstream token attention.
6. Feed gated modality tokens into the hierarchical CTAT encoder.

The gate is stored as `encoder.last_modality_gate`, so later experiments can produce modality winner maps and cross-region modality usage statistics.

## Code Locations

- `models/ctat_encoder.py`
  - `ModalityCompetitiveFusion`
  - `CTATEncoder.pos_embed`
  - `CTATEncoder.last_modality_gate`
- `models/cta_block.py`
  - `sparsemax`
  - `entmax`
  - `SparsemaxAttention`
  - `CompetitiveFFN`
- `models/ctat_network.py`
  - full encoder-decoder CTAT model
- `scripts/train_ctat.py`
  - explicit modality ordering for FA, Trace, MinEig, MidEig
- `tests/test_ctat_core.py`
  - tests for package import, token layout, FFN residual semantics, modality competition, position embedding, alpha schedule, and modality ordering

## Primary Contributions

1. **Modality-aware competitive token fusion**
   CTAT performs sparse competition across DTI scalar maps at the same anatomical patch location before token attention.

2. **Sparse attention as a controlled competition mechanism**
   Alpha-entmax provides a continuum from dense softmax to sparsemax, enabling an annealed training schedule and fixed-sparsity ablations.

3. **Direct dMRI brain parcellation benchmark**
   CTAT should be compared against DDParcel under matched preprocessing, label mapping, subject-level splits, and three-view inference.

4. **Mechanism diagnostics**
   The modality gate allows quantitative analysis of which DTI maps are selected by anatomical region and view.

## Claims To Avoid

Avoid these claims unless a much deeper literature review and stronger experiments support them:

- first sparsemax Transformer,
- first sparse Transformer for medical segmentation,
- first multi-modal medical Transformer,
- first direct brain parcellation from dMRI,
- attention maps prove biological causality,
- state-of-the-art dMRI parcellation without beating DDParcel, DDEvENet, and recent direct dMRI baselines.

## Safer Claims

These are more defensible:

- CTAT introduces a modality-aware sparse competitive token fusion mechanism tailored to direct dMRI brain parcellation.
- CTAT studies whether DDParcel-style competition can be generalized from CNN feature maps to Transformer tokens.
- CTAT provides modality selection diagnostics for FA, Trace, MinEig, and MidEig during anatomical parcellation.

## Experimental Design

### Splits

Use subject-level splits only. Never split by slices.

Recommended:

- 70% train,
- 10-15% validation,
- 15-20% locked test.

If using family-structured data, keep related subjects in the same split. If possible, add one external cohort from a different scanner or acquisition setting.

### Baselines

Primary baseline:

- DDParcel or the closest available DDParcel-style fused U-Net, retrained on the same split.

Secondary baselines:

- CTAT with softmax attention,
- CTAT with fixed entmax alpha = 1.5,
- CTAT with fixed sparsemax alpha = 2.0,
- CTAT with alpha annealing,
- 28-channel concatenation U-Net without modality competition,
- Swin/UNETR-style Transformer baseline if compute allows,
- single-modality variants for FA, Trace, MinEig, and MidEig.

### Claim-Critical Ablations

These are the minimum experiments needed to defend the paper:

1. DDParcel vs CTAT.
2. Softmax vs entmax vs sparsemax vs alpha annealing.
3. Plain concatenated sparse attention vs same-location modality competition.
4. With vs without position embedding.
5. With vs without deep supervision.
6. Four modalities vs leave-one-modality-out.
7. Single-view vs three-view inference.
8. Missing or corrupted modality robustness.

### Metrics

Primary metric:

- subject-level foreground macro Dice after label remapping.

Secondary metrics:

- per-region Dice,
- small/rare-structure Dice,
- HD95,
- ASSD,
- surface Dice,
- volume similarity,
- calibration metrics such as ECE or Brier score,
- runtime, memory, and parameter count.

Mechanism metrics:

- modality gate entropy,
- winner modality frequency by anatomical region,
- active modalities per patch,
- same-modality vs cross-modality attention mass,
- attention sparsity by layer and view.

### Statistical Testing

Use subject as the unit of analysis.

Recommended primary test:

- paired permutation test or Wilcoxon signed-rank test on subject-level macro Dice differences.

Report:

- mean and median paired differences,
- 95% bootstrap confidence intervals,
- proportion of subjects improved,
- FDR or Holm correction for per-region comparisons.

## Expected Reviewer Attacks And Fixes

### Attack 1: Sparse attention is not modality competition.

Fix: keep the same-location modality gate as the central mechanism and report modality winner maps.

### Attack 2: CTAT is just a large Transformer.

Fix: include matched dense-attention CTAT and parameter/runtime tables.

### Attack 3: The method does not match DDParcel's multi-expert design.

Fix: frame CTAT as inspired by DDParcel, and optionally add modality-specific stems if experiments show the shared patch embed is too weak.

### Attack 4: The design promises position encoding but the code lacks it.

Fix: position embedding is now part of the encoder and must be included in the ablation matrix.

### Attack 5: Results may be leakage from slice-level splitting.

Fix: use subject-level splits only and store split manifests.

## Implementation Checklist

- [x] Fix CTAT package imports.
- [x] Fix skip token to spatial reshaping.
- [x] Fix CompetitiveFFN double residual.
- [x] Make alpha schedule advance by batch step.
- [x] Add same-location modality competition gate.
- [x] Add encoder position embedding.
- [x] Expose `last_modality_gate` for diagnostics.
- [x] Enforce semantic modality ordering in training loader.
- [ ] Add attention/gate export script for modality winner maps.
- [ ] Add memory/runtime benchmark script.
- [ ] Add full DDParcel-compatible post-processing or remove that claim.
- [ ] Add baseline training configs.
- [ ] Add locked subject-level split manifest.

## Minimal Next Engineering Step

Add a diagnostic script that runs CTAT on a small batch and saves:

- `last_modality_gate`,
- per-modality winner counts,
- gate entropy maps,
- optional overlay visualizations for anatomical slices.

This script will directly answer the strongest current criticism: whether CTAT is doing modality competition or merely sparse self-attention.

## Key References

- DDParcel: https://pmc.ncbi.nlm.nih.gov/articles/PMC10994696/
- Sparsemax: https://proceedings.mlr.press/v48/martins16.html
- Entmax: https://aclanthology.org/P19-1146/
- Adaptively Sparse Transformers: https://arxiv.org/abs/1909.00015
- mmFormer: https://arxiv.org/abs/2206.02425
- DDEvENet: https://pubmed.ncbi.nlm.nih.gov/39787735/
