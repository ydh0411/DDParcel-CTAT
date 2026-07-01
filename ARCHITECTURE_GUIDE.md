# DDParcel 架构基础指南

> 面向初学者：从 Encoder-Decoder 基础到 DDParcel 完整推理管线

---

## 目录

1. [基础概念：Encoder-Decoder 与 U-Net](#1-基础概念encoder-decoder-与-u-net)
2. [DenseNet：密集连接](#2-densenet密集连接)
3. [Maxout：竞争式特征选择](#3-maxout竞争式特征选择)
4. [QuickNAT：快速脑分割](#4-quicknat快速脑分割)
5. [FastSurferCNN：竞争式 U-Net](#5-fastsurfercnn竞争式-u-net)
6. [多模态融合：从 v1 到 v3_extended 的演进](#6-多模态融合从-v1-到-v3_extended-的演进)
7. [DDParcel：2.5D 多视角推理](#7-ddparcel25d-多视角推理)
8. [FreeSurfer 脑区分割背景](#8-freesurfer-脑区分割背景)
9. [论文汇总](#9-论文汇总)

---

## 1. 基础概念：Encoder-Decoder 与 U-Net

### 1.1 什么是 Encoder（编码器）？

编码器的任务是把输入图像**逐步压缩**成一个小尺寸但高通道数的特征图。

```
输入图像 [256×256×3]
    → Conv + Pool → [128×128×64]
    → Conv + Pool → [64×64×128]
    → Conv + Pool → [32×32×256]
    → Conv + Pool → [16×16×512]
                        ↑
                   bottleneck（瓶颈层）
                空间最小、语义最抽象
```

整个过程可以类比为「看书做摘要」：
- 先看完整页（256×256），提取边缘、颜色等低级特征（64 通道）
- 再看简化版（128×128），提取纹理、形状等中级特征（128 通道）
- 继续压缩，每一步都提取更抽象、更全局的信息
- 最后 bottleneck 层虽然空间最小，但包含了整张图最核心的语义

### 1.2 什么是 Decoder（解码器）？

解码器的任务是将 bottleneck 的抽象特征**逐步恢复到原始分辨率**，同时做像素级别的预测。

```
bottleneck [16×16×512]
    → Unpool/UpConv → [32×32×256]
    → Unpool/UpConv → [64×64×128]
    → Unpool/UpConv → [128×128×64]
    → Unpool/UpConv → [256×256×C]  ← C = 类别数
```

### 1.3 U-Net 的核心创新：跳跃连接（Skip Connection）

如果只有 Encoder → Decoder 这条直路，下采样过程中丢失的空间细节（如小结构的边界位置）是无法恢复的。U-Net 的解决办法是**跳跃连接**：把编码器每一层的输出「抄近道」送给解码器对应层。

```
Encoder1_out ────────────→ concat ──→ Decoder1_in
       ↓                                  ↑
    MaxPool                          MaxUnpool
       ↓                                  ↑
Encoder2_out ────────────→ concat ──→ Decoder2_in
       ↓                                  ↑
      ...                                ...
       ↓                                  ↑
   bottleneck ────────────────────→ Decoder4_in
```

跳跃连接将**编码器的空间细节**（高分辨率、低级特征）和**解码器的语义信息**（低分辨率、高级特征）结合在一起，使得网络既能「看到森林」（全局语义），又能「看清树木」（精细边界）。

**论文：** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- arXiv: https://arxiv.org/abs/1505.04597
- 被引用超 60,000 次，是医学图像分割的基石

---

## 2. DenseNet：密集连接

### 2.1 核心思想

传统 CNN 中，每一层只从前一层接收输入：
```
Layer1 → Layer2 → Layer3 → Layer4
```

DenseNet 的密集块中，每一层从**所有前层**接收输入：
```
Layer1 → Layer2 → Layer3 → Layer4
  ↓       ↓       ↓
  └───────┴───────┴────→ 每层都连到所有后续层
```

### 2.2 优势

- **特征复用**：浅层的边缘、纹理等低级特征直接传给深层，避免被中间层「遗忘」
- **缓解梯度消失**：损失函数的梯度有多条路径回传，训练更稳定
- **参数效率高**：可以比 ResNet 用更少的参数达到相同精度

### 2.3 在 DDParcel 中的应用

DDParcel 的 `CompetitiveDenseBlock` 继承了 DenseNet 的密集连接思想，但做了关键改动：将通道拼接（concat）替换为 maxout 竞争（下文详述）。

**论文：** Huang et al., "Densely Connected Convolutional Networks", CVPR 2017 (Best Paper)
- arXiv: https://arxiv.org/abs/1608.06993

---

## 3. Maxout：竞争式特征选择

### 3.1 什么是 Maxout？

Maxout 是一种**可学习的激活函数**，其操作是在多个候选特征图之间逐像素取最大值：

```
特征图1: [0.3, 0.8, 0.1]     maxout →
特征图2: [0.5, 0.2, 0.7]     [0.5, 0.8, 0.7]
特征图3: [0.1, 0.4, 0.6]
```

在 DDParcel 的代码中，这体现为：
```python
# 沿第 5 维堆叠多个特征图，然后取 max
x1 = torch.cat((feature_a, feature_b), dim=4)  # [B,C,H,W,2]
x1_max, _ = torch.max(x1, 4)                    # [B,C,H,W] — 逐位置取最强的
```

### 3.2 为什么用 Maxout 替代 Concat？

传统 DenseNet/U-Net 用 **concat** 融合特征：将两个特征图沿通道维拼接，通道数翻倍。这导致：
- 参数量随网络深度快速增长
- 显存和计算量增加

**Maxout 的优势**：
- 通道数**不增加**（始终是 `num_filters`），参数更少
- 引入「竞争」机制：每像素在候选特征中选择最强的那个
- 网络自动学习哪种特征在哪个位置更重要

可以类比为「多专家投票，每个位置选最自信的那个专家」。

### 3.3 Competitive Dense Block（CDB）的工作流程

在 `sub_module.py` 中，CompetitiveDenseBlock 的每一层都做了 maxout 竞争：

```
输入 x
  │
  ├──→ PReLU → Conv0 → BN ──┐
  │                          ├─→ maxout(x_original, Conv0_out) → PReLU
  └──→ (短路) ──────────────┘         │
                                      ├──→ Conv1 → BN ──┐
                                      │                  ├─→ maxout(上一步, Conv1_out)
                                      └──→ (短路) ──────┘
```

每次竞争后，网络可以「选择」保留原始特征还是用新卷积提取的特征。这种「残差 + 竞争」的机制比简单的加法（ResNet）或拼接（DenseNet）更轻量，同时保持甚至提升了表达能力。

**论文：** Goodfellow et al., "Maxout Networks", ICML 2013
- arXiv: https://arxiv.org/abs/1302.4389

---

## 4. QuickNAT：快速脑分割

QuickNAT 是 FastSurferCNN 的直接前身。它率先在脑 MR 分割中使用了：

- **三个正交视角的 2D F-CNN**：轴向（Axial）、冠状（Coronal）、矢状（Sagittal）各一个网络
- **密集连接的编码器-解码器块**：受 DenseNet 启发
- **Unpooling 替代转置卷积**：减少棋盘伪影
- **20 秒完成全脑分割**：相比传统 FreeSurfer（数小时）快了几个数量级

FastSurferCNN 在 QuickNAT 基础上做了关键改进：将 concat 全部替换为 maxout 竞争，得到更轻量、更快的网络。

**论文：** Roy et al., "QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy", NeuroImage, 2019
- DOI: https://doi.org/10.1016/j.neuroimage.2018.11.042
- arXiv: https://arxiv.org/abs/1801.04161

---

## 5. FastSurferCNN：竞争式 U-Net

### 5.1 整体结构

FastSurferCNN 是该项目的核心架构，一个基于 Competitive Dense Block 的 U-Net：

```
输入 [B, 7, 256, 256]  （7 = 厚切片：目标切片 ± 相邻 3 张）
      ↓
┌──────────────────────────┐
│ Encoder Block 1 (Input)  │ → skip_1, indices_1  [256×256]
│ MaxPool                  │ → 128×128
├──────────────────────────┤
│ Encoder Block 2          │ → skip_2, indices_2  [128×128]
│ MaxPool                  │ → 64×64
├──────────────────────────┤
│ Encoder Block 3          │ → skip_3, indices_3  [64×64]
│ MaxPool                  │ → 32×32
├──────────────────────────┤
│ Encoder Block 4          │ → skip_4, indices_4  [32×32]
│ MaxPool                  │ → 16×16
├──────────────────────────┤
│ Bottleneck (CDB)         │ → 最深特征  [16×16]
├──────────────────────────┤
│ Decoder Block 4          │ ← Unpool + skip_4   [32×32]
│ Decoder Block 3          │ ← Unpool + skip_3   [64×64]
│ Decoder Block 2          │ ← Unpool + skip_2   [128×128]
│ Decoder Block 1          │ ← Unpool + skip_1   [256×256]
├──────────────────────────┤
│ Classifier (1×1 Conv)    │ → [B, num_classes, 256, 256]
└──────────────────────────┘
```

### 5.2 关键设计决策

| 设计 | 传统做法 | FastSurferCNN |
|------|---------|---------------|
| 跳跃连接融合 | concat（通道数翻倍） | maxout 竞争（通道数不变） |
| 块内连接 | 标准卷积堆叠 | Competitive Dense Block |
| 上采样 | 转置卷积（有参数） | MaxUnpool + maxout（无参数） |
| 输入通道 | 单切片（3 通道） | 厚切片（7 通道，含相邻切片） |

### 5.3 厚切片策略（Spatial Information Aggregation）

每个 2D 输入不是一张切片，而是**7 张相邻切片叠成 7 通道**：

```
┌─────────────────────────────────┐
│ 切片 i-3  │ 切片 i-2  │ ... │ 切片 i │ ... │ 切片 i+3 │
└─────────────────────────────────┘
            堆叠为 7 通道 → 送入 2D 网络
```

这样不增加空间维度（仍是 2D 卷积），但让网络「看到」了目标切片前后的上下文信息。这是一种**2.5D**策略：比纯 2D 有更多上下文，比纯 3D 有更少的计算量。

### 5.4 MaxUnpool（反池化）

解码器上采样用 MaxUnpool 而非转置卷积：

```
编码端 MaxPool 时：
  保存池化索引：最大值的空间位置
  ┌──────┐
  │1 3│2 4│  → MaxPool → ┌───┐
  │5 6│7 8│              │6 8│   indices = 记录 6 和 8 的位置
  └──────┘              └───┘

解码端 MaxUnpool 时：
  用同样的索引将特征值「放回」原来的位置：
  ┌───┐               ┌──────────┐
  │a b│  → Unpool →   │a 0│0 b 0│
  └───┘               │0 0│0 0 0│  (其他位置填 0)
                      └──────────┘
```

优势：无参数（不需要学习转置卷积核）、空间对应关系精确。

**论文：** Henschel et al., "FastSurfer — A fast and accurate deep learning based neuroimaging pipeline", NeuroImage, 2020
- DOI: https://doi.org/10.1016/j.neuroimage.2020.117012
- GitHub: https://github.com/Deep-MI/FastSurfer

---

## 6. 多模态融合：从 v1 到 v3_extended 的演进

### 6.1 问题背景

原始 FastSurferCNN 处理单模态 T1w MR 图像（7 通道厚切片）。DDParcel 需要同时处理 **4 种 DTI 标量图**：
- FA（各向异性分数）
- Trace（扩散迹）
- MinEig（最小特征值）
- MidEig（中间特征值）

问题：**如何融合这 4 种模态的信息？**

### 6.2 融合策略演进

```
v1: 最后加权相加
    各模态独立 U-Net → 把所有 logits 加起来 → 输出
    问题：模态间没有中间交互，各模态的错误无法被其他模态修正

v2: 通道拼接输入 + 编码层竞争融合
    所有 28 通道拼接输入 → 主分支（带 maxout 竞争融合）
    问题：各 backbone 不够独立，信息混合不够充分

v3 / v3_extended ★ (最终使用):
    ┌──────────────────────────────────────────────┐
    │ backbone_0(FA)    → 提取中间特征 {e1,e2,e3,e4,dec1} │
    │ backbone_1(Trace) → 提取中间特征 {e1,e2,e3,e4,dec1} │
    │ backbone_2(MinEig)→ 提取中间特征 {e1,e2,e3,e4,dec1} │
    │ backbone_3(MidEig)→ 提取中间特征 {e1,e2,e3,e4,dec1} │
    │                                              │
    │ 主融合分支(28ch)  ──┬── encode1 ── maxout ← 各 backbone_e1
    │                    ├── encode2 ── maxout ← 各 backbone_e2
    │                    ├── encode3 ── maxout ← 各 backbone_e3
    │                    ├── encode4 ── maxout ← 各 backbone_e4
    │                    ├── bottleneck ─ maxout ← 各 backbone_bottleneck
    │                    ├── decode4..1
    │                    │
    │                    └── concat(主decode1, backbone0_dec1, backbone1_dec1, ...)
    │                              ↓
    │                        fusion_layer (1x1 Conv)
    │                              ↓
    │                        classifier (1x1 Conv) → logits
    │                                              │
    │                    logits_list = [主融合logits, backbone0_logits, ...]
    └──────────────────────────────────────────────┘
```

### 6.3 v3_extended 的关键设计

1. **各模态 backbone 冻结**：每个 backbone 是一个预训练的 FastSurferCNN，参数不更新，只做特征提取。这相当于每个模态有一个固定的「专家」。

2. **编码层竞争融合**：在 encoder1~4 和 bottleneck 共 5 个层级，将主分支特征与各 backbone 特征做 maxout 竞争，让模型在每个分辨率层级「选择」哪个模态的特征最有用。

3. **解码层最终融合**：将主分支的解码输出与各 backbone 的解码输出拼接，经过 1×1 卷积（`fusion_layer`）融合后再分类。这是 v3 相比 v2 的关键改进。

4. **多路 logits 累加**：推理脚本（`DDSurfer_Pred.py`）中，最终输出是将主融合 logits 与各 backbone logits 累加（`logits += ind_return[0]`），实现**多路投票**。

---

## 7. DDParcel：2.5D 多视角推理

### 7.1 三视角融合策略

DDParcel 的推理在三个正交方向（Axial/Coronal/Sagittal）上各运行一次 v3_extended 网络：

```
                    输入：4 模态 DTI 体数据 [256×256×256×4]
                                  │
            ┌─────────────────────┼─────────────────────┐
            ↓                     ↓                     ↓
      Axial 视角              Coronal 视角           Sagittal 视角
      (沿 z 切片)             (沿 y 切片)            (沿 x 切片)
            │                     │                     │
      v3_extended            v3_extended            v3_extended
      82 类输出              82 类输出              54 类输出
      (映射到完整标签空间)
            │                     │                     │
            ↓                     ↓                     ↓
      权重 0.4               权重 0.4               权重 0.2
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ↓
                        三视角概率累加
                                  ↓
                        argmax → 标签图
```

### 7.2 为什么矢状面权重是 0.2？

- 矢状面网络输出类别数较少（54 vs 82），对左右半球标签的分辨力较弱
- 在中间矢状面（midline）附近效果好，越往两侧效果越差
- 给予较低权重，让轴向和冠状面主导结果

### 7.3 数据流总结

```
输入：FA, Trace, MinEig, MidEig 四个 NIfTI 文件
  │
  ├─ 1) load_and_conform_image() → 放大到 256³ 空间，归一化
  │
  ├─ 2) 三视角依次推理（共用同一组 backbone 权重）
  │      每个视角加载该视角训练的融合头权重
  │      2D 网络逐切片推理 → 拼回 3D 体数据
  │      按视角权重累加到 pred_prob
  │
  ├─ 3) argmax → 标签映射
  │      内部索引 (0..81) → FreeSurfer aparc+aseg 标签 (2, 41, 1001..)
  │
  ├─ 4) 半球纠错
  │      基于白质质心的高斯平滑约束，修正左右标签混淆
  │
  └─ 5) 可选连通域清理 → 保存 .mgz
```

---

## 8. FreeSurfer 脑区分割背景

DDParcel 输出的标签遵循 FreeSurfer 的 `aparc+aseg` 标注体系。

### 8.1 FreeSurfer 是什么？

FreeSurfer 是哈佛大学开发的神经影像分析软件包，用于从结构 MRI 中自动重建大脑皮层表面并进行解剖标注。传统流程（`recon-all`）需要数小时处理一个被试。

### 8.2 关键概念

| 概念 | 说明 |
|------|------|
| aparc+aseg | 皮层分区（aparc）+ 皮层下分割（aseg）的组合标注 |
| Desikan-Killiany (DK) | 基于脑回的皮层分区模板，每半球约 34 个 ROI |
| 标签编号规则 | 1xxx = 左脑皮层, 2xxx = 右脑皮层, 两位数 = 皮层下结构 |

### 8.3 常见标签示例

| 标签号 | 解剖名称 | 说明 |
|--------|---------|------|
| 2 | Left-Cerebral-White-Matter | 左脑白质 |
| 41 | Right-Cerebral-White-Matter | 右脑白质 |
| 1001 | ctx-lh-superiortemporal | 左脑颞上回 |
| 2001 | ctx-rh-superiortemporal | 右脑颞上回 |
| 1011 | ctx-lh-isthmuscingulate | 左脑扣带回峡部 |

**论文：**
- Desikan et al. (2006), "An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest", NeuroImage, 31(3):968-80
- Fischl et al. (2004), "Automatically parcellating the human cerebral cortex", Cerebral Cortex, 14(1):11-22

---

## 9. 论文汇总

### 核心架构论文

| 论文 | 贡献 | 链接 |
|------|------|------|
| **Ronneberger et al. (2015)** — U-Net | 编码器-解码器 + 跳跃连接的基础架构 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| **Huang et al. (2017)** — DenseNet | 密集连接卷积网络（CVPR Best Paper） | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
| **Goodfellow et al. (2013)** — Maxout Networks | 竞争式取最大值的激活函数 | [arXiv:1302.4389](https://arxiv.org/abs/1302.4389) |
| **Roy et al. (2019)** — QuickNAT | 快速脑 MR 分割，FastSurfer 的前身 | [DOI:10.1016/j.neuroimage.2018.11.042](https://doi.org/10.1016/j.neuroimage.2018.11.042) |

### 本项目直接相关论文

| 论文 | 贡献 | 链接 |
|------|------|------|
| **Henschel et al. (2020)** — FastSurfer | Competitive Dense Block 脑分割，本项目的 backbone | [DOI:10.1016/j.neuroimage.2020.117012](https://doi.org/10.1016/j.neuroimage.2020.117012) |
| **Zhang et al. (2024)** — DDParcel | 从 dMRI 做脑区分割，本项目的主体 | [DOI:10.1109/TMI.2023.3331691](https://doi.org/10.1109/TMI.2023.3331691) |

### 脑区标注参考论文

| 论文 | 贡献 |
|------|------|
| **Desikan et al. (2006)** | Desikan-Killiany 皮层分区模板 |
| **Fischl et al. (2004)** | FreeSurfer 自动皮层分区方法 |

### 推荐阅读顺序

```
对于初学者，建议按以下顺序理解：
 
  U-Net (2015)          理解 encoder-decoder + skip connection
      ↓
  DenseNet (2017)       理解密集连接如何提升特征复用
      ↓
  Maxout (2013)         理解竞争式特征选择的原理
      ↓
  QuickNAT (2019)       理解如何将 DenseNet 用于脑分割
      ↓
  FastSurfer (2020)     理解 CDB 如何替代 concat 做更轻量的融合
      ↓
  DDParcel (2024)       理解多模态多视角融合的完整管线
```

---

## 附录：DDParcel 代码文件速查

| 文件 | 作用 |
|------|------|
| `models/networks.py` | 11 个网络变体定义（从基础 U-Net 到 v3_extended） |
| `models/sub_module.py` | 6 个基础模块（CDB, EncoderBlock, DecoderBlock, Classifier） |
| `models/losses.py` | 损失函数（CrossEntropy + Dice Loss） |
| `models/solver.py` | 训练流程 |
| `DDSurfer_Pred.py` | 推理主入口：三视角输入 → 概率累加 → 后处理 → 保存 |
| `data_loader/load_neuroimaging_data.py` | 数据加载、标签映射、半球纠错 |
| `data_loader/conform.py` | 输入图像的空间标准化（256³ 空间） |
| `data_loader/augmentation.py` | 数据增强 |
| `normalize.py` | 预处理：DTI 标量图归一化 |
