# CTAT 改进方向调研

> 基于最新论文（CVPR/ICML/MICCAI 2025-2026）的改进方案设计
> 日期: 2026-07-01

---

## 1. D-RoPE: dMRI 专用 Rotary Position Embedding

**来源**: "Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding (D-RoPE)", 2026, arXiv:2603.25977
**代码**: github.com/gustavochau/D-RoPE

### 核心思路

标准 RoPE 只编码 2D/3D 空间位置。D-RoPE 将扩散梯度方向（b-vector, 球面坐标 θ, φ）和 b-value 也编码进旋转矩阵，让 attention 原生感知 dMRI 的球面几何。

```
标准 RoPE:        q·k 编码 (x, y) 空间距离
D-RoPE:           q·k 同时编码 (x, y, θ, φ, b) 空间+扩散方向距离
```

### 对 CTAT 的改进

当前 CTAT 使用可学习的 1D position embedding，对 4 个 modality 的 7 个 slice 一视同仁。替换为 D-RoPE 后：

1. 相同空间位置、相似扩散方向的 patch 获得更高 attention score
2. 跨 protocol 泛化（不同 b-value/direction 数量也能处理）
3. 让 modality embedding 和 position embedding 统一为旋转编码

### 实现方案

```python
# 在 ctat_encoder.py 的 PatchEmbed 后插入 D-RoPE
class DRoPE(nn.Module):
    def __init__(self, dim, max_resolution=256):
        # 空间维度: 2D rotary (x, y)
        # 扩散维度: 2D rotary (θ, φ from b-vector)
        # 总 dim 被均分: dim/2 给空间, dim/2 给扩散
        ...
    def forward(self, x, bvecs):
        # x: [B, N, C]
        # bvecs: [B, M, 3] 梯度方向表
        # 分别计算空间和扩散 rotary embedding，拼接
        ...
```

### 预期收益
- 更好的跨 subject 泛化（dMRI 采集参数不同时）
- 可能提升边界区域的分割精度
- 参数量几乎不增

### 实现难度: 中 (~100行)
### 风险: 低（RoPE 是成熟技术，D-RoPE 是学界的自然延伸）

---

## 2. AdaSplash: 加速 entmax 计算

**来源**: "AdaSplash: Adaptive Sparse Flash Attention", ICML 2025, PMLR 267:19878-19896
**来源2**: "AdaSplash-2: Faster Differentiable Sparse Attention", 2026, arXiv:2604.15180

### 核心思路

当前 CTAT 的 entmax 用纯二分法（50 次迭代），每次 forward 要跑多次 bisection。AdaSplash 用 Halley 法替代二分法，迭代从 50 降到 ~7，而且：
- AdaSplash-2 用 histogram-based 初始化，进一步降到 1-2 次迭代
- 提供 Triton kernel 实现，利用 GPU 共享内存

### 对 CTAT 的改进

直接替换 `cta_block.py` 中的 `entmax()` 函数：

```python
def entmax_halley(logits, alpha=1.5, dim=-1, n_iter=10):
    """Halley's method for entmax — converges in ~7 iterations vs 50."""
    # Halley: tau_{n+1} = tau_n - 2*f*f' / (2*(f')^2 - f*f'')
    # f(tau) = sum(clamp((α-1)*(logits-tau), 0)^(1/(α-1))) - 1
    ...
```

### 预期收益
- 每次 entmax 计算加速 5-7x
- 对于 121M 参数的 CTAT，训练时 entmax 占比约 15-20%，整体训练加速 10-15%
- 5060 笔记本上体验改善明显

### 实现难度: 低 (~30行)
### 风险: 极低（纯数值优化，不改变输出）

---

## 3. 内容感知稀疏注意力（Content-Aware Sparse Attention）

**来源**: 
- MedFormer: "Hierarchical Medical Vision Transformer with Content-Aware Dual Sparse Selection Attention", PMB 2025
- DashAttention: "Differentiable and Adaptive Sparse Hierarchical Attention", 2026, arXiv:2605.18753
- MDSA-UNet: "Multi-Scale Dynamic Sparse Attention UNet", JBHI 2025

### 核心思路

当前 CTAT 的稀疏性是"预设的"：window attention 固定窗口，alpha annealing 全程统一。内容感知稀疏注意力让网络**自己学习**哪些 token 该关注：

```
当前 CTAT:          window_size=8 固定 → 每个 token 只看邻居 8x8
DashAttention:      entmax 选 top-k blocks → k 随输入变化
MedFormer DSSA:     双阶段选择 → 先粗选 block，再精选 token
MDSA:              多尺度聚合 → 在粗粒度过滤无关 token
```

### 三种方案对比

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| DashAttention | 用 entmax 替代 top-k 做 block 选择，完全可微 | 训练端到端，不需要预设 k | block 粒度可能丢失细节 |
| MedFormer DSSA | 双阶段：先选 block 再选 token，两级稀疏 | 精细度高 | 实现复杂 |
| MDSA | 多尺度聚合 + 粗粒度过滤 + 细粒度 attention | 参数少，速度快 | 多尺度 overhead |

### 对 CTAT 的推荐方案：DashAttention 风格

CTAT 已有 window attention + entmax。改成 DashAttention 风格：
- Stage 0: 保留 window attention（局部细节）
- Stage 1-3: 用 entmax 做 block-level routing（自适应长程依赖）
- Bottleneck: 全局 attention（最深层的抽象推理）

```python
# cta_block.py 中新增
class DashCTABlock(nn.Module):
    """CTABlock with content-aware block routing."""
    def forward(self, x):
        # 1. 粗粒度 block 打分
        block_scores = self.block_scorer(x)  # [B, num_blocks]
        # 2. entmax 选择活跃 blocks（自适应稀疏）
        block_gate = entmax(block_scores, alpha=self.alpha, dim=-1)
        # 3. 只对 gate>0 的 blocks 做细粒度 attention
        active_blocks = block_gate > 0
        attn_out = self.sparse_attn(x, mask=active_blocks)
        ...
```

### 预期收益
- 对复杂解剖结构（如皮层折叠处）更好的建模
- 减少无效 token 的 attention 计算（可能加速 20-30%）
- 论文角度：从 "preset sparsity" 升级为 "learned competition"

### 实现难度: 高 (~300行，需重设计 attention 模块)
### 风险: 中（训练可能不稳定，需要调参）

---

## 4. 多尺度竞争融合（Multi-Scale Competition）

**来源**: DDParcel v3_extended (张老师原始设计) + PE-Transformer + MuViT (CVPR 2026)

### 核心思路

DDParcel 原版在 **5 个分辨率级别**（256→128→64→32→16→8→4）都做 maxout 竞争。但 CTAT 只在 patch embedding 后做一次 entmax 竞争，之后所有尺度共享同一个 gate。

**问题**: 某些模态在粗尺度有用（如 FA 对大结构），某些在细尺度有用（如 Trace 对边界）。单次 gate 丢失了多尺度互补性。

### 改进方案

在 encoder 的 4 个 stage 输出 skip connection 之前，各加一个轻量 ModalityCompetitiveFusion：

```python
# ctat_encoder.py
class CTATEncoder(nn.Module):
    def forward(self, x):
        x = self.patch_embed(x)        # [B, M, N, C]
        x = self.modality_fusion(x)    # 初始竞争（保留）
        
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 每个 stage 输出前再做一次跨模态竞争
            x = self.stage_fusions[i](x)  # 新增
            skips.append(x)
        ...
```

每个 stage_fusion 是轻量的 entmax gate (M×C 参数)，只在 skip connection 点做竞争。

### 预期收益
- 恢复 DDParcel 原始设计的多尺度竞争优势
- 可能提升细粒度结构的 Dice（如小脑区分割）
- 论文角度：结合 CNN 的多尺度优势和 Transformer 的长程依赖

### 实现难度: 中 (~80行)
### 风险: 低（保持原有 CTAT 结构，只是插入轻量模块）

---

## 5. 最优 dMRI 参数选择

**来源**: DKParcellationdMRI (2025-26, Sci Rep) + DDEvENet (2025)

### 核心思路

DKParcellationdMRI 系统研究了不同 dMRI 参数组合对 DK parcellation 的影响：

| 参数组合 | Dice |
|---------|------|
| FA only | baseline |
| FA + MD | +1.2% |
| FA + MD + Trace | +1.8% |
| **FA + Trace + Sphericity + MaxEig** | **+2.5%** (最优) |

当前 CTAT 用 7 个 slice × 4 个 modality = 28 通道输入。可以：
1. 增加 Sphericity 和 MaxEigenvalue 作为额外通道
2. 或：用 DDEvENet 风格的 evidence ensemble，每个参数一个子编码器

### 实现方案

最小改动：在 data_loader 中多提取两个 dMRI 参数，将输入从 [B,28,256,256] 改为 [B,42,256,256]（6 参数 × 7 slices）。

### 实现难度: 低（改 data_loader ~20行 + 改 in_channels 参数）
### 风险: 极低（纯粹增加输入信息）

---

## 优先级建议

| 优先级 | 改进 | 难度 | 预期收益 | 建议顺序 |
|--------|------|------|---------|---------|
| P0 | AdaSplash 加速 entmax | 低 | 训练加速 10-15% | 1st |
| P0 | 最优 dMRI 参数选择 | 低 | Dice +2-3% | 1st (并行) |
| P1 | D-RoPE 位置编码 | 中 | 泛化能力提升 | 2nd |
| P1 | 多尺度竞争融合 | 中 | 细粒度 Dice 提升 | 2nd (并行) |
| P2 | 内容感知稀疏注意力 | 高 | 最大创新性 | 3rd (需要前几个稳定后) |

---

## 参考文献

1. Chau G, et al. "D-RoPE: Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding." arXiv:2603.25977, 2026.
2. Gonçalves N, et al. "AdaSplash: Adaptive Sparse Flash Attention." ICML 2025, PMLR 267.
3. Gonçalves N, et al. "AdaSplash-2: Faster Differentiable Sparse Attention." arXiv:2604.15180, 2026.
4. Huang Y, et al. "DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention." arXiv:2605.18753, 2026.
5. Xia Z, et al. "MedFormer: Hierarchical Medical Vision Transformer with Content-Aware Dual Sparse Selection Attention." PMB, 2025.
6. Li X, et al. "MDSA-UNet: Multi-Scale Dynamic Sparse Attention UNet." JBHI, 2025.
7. Sadegheih Y, Merhof D. "Deep Learning-Based Desikan-Killiany Parcellation of the Brain Using Diffusion MRI." Sci Rep, 2026.
8. Zhang F, et al. "DDParcel: Deep Learning Anatomical Brain Parcellation From Diffusion MRI." IEEE TMI, 2024.
9. DDEvENet: Evidence-based Ensemble for Uncertainty-aware Brain Parcellation Using Diffusion MRI. 2025.
10. Yao T, et al. "Polyhedra Encoding Transformers: Enhancing Diffusion MRI Analysis." arXiv:2501.13352, 2025.
11. Song P, et al. "Uni-Encoder Meets Multi-Encoders: Representation Before Fusion." CVPR 2026.
