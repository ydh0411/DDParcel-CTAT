# DDParcel 网络模型大全：逐个详解 11 个架构

> 本文逐一分析 `models/networks.py` 中全部 11 个网络类，解释每个类的结构、设计动机、以及融合细节。

---

## 前置知识速览

在开始之前，先记住这几个关键事实：

| 事实 | 说明 |
|------|------|
| 每个模态输入 = **7 通道** | 目标切片 ± 前后各 3 张相邻切片，共 7 张堆叠 |
| 输入尺寸 | `[B, 28, 256, 256]`（4 模态 × 7 通道 = 28） |
| num_filters = 64 | 所有网络的内部特征通道数统一为 64 |
| 模块定义在 sub_module.py | CompetitiveEncoderBlock、CompetitiveDecoderBlock、ClassifierBlock 等 |
| 推理实际使用 | **仅** v3_extended（轴线/冠状/矢状各一个权重） |

---

# 第一部分：单模态基础网络（3 个）

这三个类构成了「积木」——后文的融合网络都在它们的基础上搭建。

---

## 1. FastSurferCNN — 基础 U-Net

### 结构

```
输入 [B, 7, 256, 256]
    │
    ├─ encode1 (CompetitiveEncoderBlockInput) → skip1, indices1
    │      └─ MaxPool(2×2, stride=2) → [B, 64, 128, 128]
    │
    ├─ encode2 (CompetitiveEncoderBlock) → skip2, indices2
    │      └─ MaxPool → [B, 64, 64, 64]
    │
    ├─ encode3 (CompetitiveEncoderBlock) → skip3, indices3
    │      └─ MaxPool → [B, 64, 32, 32]
    │
    ├─ encode4 (CompetitiveEncoderBlock) → skip4, indices4
    │      └─ MaxPool → [B, 64, 16, 16]
    │
    ├─ bottleneck (CompetitiveDenseBlock)
    │      → [B, 64, 16, 16]
    │
    ├─ decode4: MaxUnpool(bottleneck, indices4) → maxout(skip4) → DenseBlock
    │      → [B, 64, 32, 32]
    ├─ decode3: MaxUnpool → maxout(skip3) → DenseBlock
    │      → [B, 64, 64, 64]
    ├─ decode2: MaxUnpool → maxout(skip2) → DenseBlock
    │      → [B, 64, 128, 128]
    ├─ decode1: MaxUnpool → maxout(skip1) → DenseBlock
    │      → [B, 64, 256, 256]
    │
    └─ classifier: 1×1 Conv(64 → num_classes)
           → [B, num_classes, 256, 256]
```

### 要点

- encode1 用的是 `CompetitiveEncoderBlockInput`（多了输入 BN 层），encode2~4 用的是 `CompetitiveEncoderBlock`
- bottleneck 是纯 `CompetitiveDenseBlock`，不做池化
- 解码器用 MaxUnpool（而非转置卷积）配合编码器存下的 indices 做上采样
- 跳跃连接融合：skip 特征和 unpool 后的特征做 **maxout 竞争**（不是 concat）

### 为什么它是「根」

后文所有融合网络的 backbone 都是这个结构（或它的变体 return_all）。

---

## 2. FastSurferCNN_return_all — 会「交出」所有中间结果的基础 U-Net

### 和 FastSurferCNN 的区别

**结构完全相同。** 唯一的区别在于 `forward()` 返回值：

```python
# FastSurferCNN 的 forward：
return logits                                     # 只返回 1 个结果

# FastSurferCNN_return_all 的 forward：
return (
    logits,                                        # [0]  最终分类结果
    encoder_output1, skip_encoder_1, indices_1,    # [1][2][3]
    encoder_output2, skip_encoder_2, indices_2,    # [4][5][6]
    encoder_output3, skip_encoder_3, indices_3,    # [7][8][9]
    encoder_output4, skip_encoder_4, indices_4,    # [10][11][12]
    bottleneck,                                    # [13]
    decoder_output4, decoder_output3,              # [14][15]
    decoder_output2, decoder_output1               # [16][17]
)                                                  # 共 18 个返回值
```

### 为什么要暴露中间结果？

在融合网络（v3_extended 等）中，主融合分支需要在**编码器每一层**读取各 backbone 的中间特征来做 maxout 竞争。

打个比方：FastSurferCNN 是「黑盒子操作员」，只告诉你最后结论。FastSurferCNN_return_all 是「透明操作员」，每一项中间推理都摊在桌面上供你参考。

---

## 3. FastSurferCNN_no_classifer — 去掉分类头的特征提取器

### 结构

```
输入 [B, 7, 256, 256]
    │
    ...（encode1~4 + bottleneck + decode4~1 完全同上）
    │
    └─ return decoder_output1   [B, 64, 256, 256]
       （没有 classifier！）
```

去掉了最后的 1×1 Conv 分类器。输出是 **64 通道的原始解码特征图**，而非 82 类的 logits。

### 为什么要去掉分类头？

在 `Fuse_Last_Layer` 中，每个模态先独立经过无分类头的 U-Net 提取特征，然后**所有模态的解码特征拼接后共用同一个分类器**。

如果保留了各自的分类头，每个 backbone 就会产出独立的 logits，无法在特征层面做融合——只能像 v1 那样在 logits 层加权相加（次优方案）。

---

# 第二部分：多模态融合网络（8 个）

以下每个模型解决同一个问题：**如何融合 FA、Trace、MinEig、MidEig 四个模态的信息？**

---

## 4. FastSurferCNN_Fuse_Last_Layer — 最简融合：仅在最后融合

### 结构

```
输入 x [B, 28, 256, 256]

  ├─ x[:, 0:7, :, :]  →  FastSurferCNN_no_classifer  →  dec1_0  [B,64,256,256]
  ├─ x[:, 7:14, :, :] →  FastSurferCNN_no_classifer  →  dec1_1  [B,64,256,256]
  ├─ x[:, 14:21,:, :] →  FastSurferCNN_no_classifer  →  dec1_2  [B,64,256,256]
  └─ x[:, 21:28,:, :] →  FastSurferCNN_no_classifer  →  dec1_3  [B,64,256,256]
                                                           │
         4 个 dec1 在 dim=1 上拼接 → [B, 256, 256, 256]
                           │
                     fusion_layer: 1×1 Conv(256 → 64)
                           │
                     classifier: 1×1 Conv(64 → num_classes)
                           ↓
                      logits [B, num_classes, 256, 256]
```

### 设计动机

最直觉的想法：给每个模态配一个独立的 U-Net，各提取各的，最后把结果拼起来再分类。相当于「分头行动，最后汇总」。

### 融合的是什么？

融合的是**各 backbone 解码器最后一层的 64 通道特征图**。4 个 64 通道拼接成 256 通道，然后 1×1 卷积压缩回 64 通道，再分类。

### 缺点

各模态在编码器的每一层**完全独立**，互不知情。模态 A 在网络浅层就做出了错误判断，模态 B 无法在早期纠正它——只能等到最后融合时才能「补救」。

---

## 5. FastSurferCNN_Fuse_Unet — 深层融合：编码器每层都做融合

### 结构

```
输入 x [B, 28, 256, 256]

  各 backbone（return_all，预训练并冻结）：
  ├─ backbone0(x[:, 0:7])   → 返回 e1,e2,e3,e4,bottleneck,... 等 18 个值
  ├─ backbone1(x[:, 7:14])  → 同上
  ├─ backbone2(x[:, 14:21]) → 同上
  └─ backbone3(x[:, 21:28]) → 同上

  主融合分支（只用模态0 的 7 通道）：
  ├─ encode1(x[:, 0:7])
  │     → 输出 e1_main
  │     → e1_main ╋ backbone0.e1 ╋ backbone1.e1 ╋ backbone2.e1 ╋ backbone3.e1
  │             └─── maxout ──┘  (沿第5维堆叠5个[B,64,H,W]，取max)
  │             → fusion1 (1×1 Conv)
  │
  ├─ encode2(上一步输出)
  │     → maxout(主分支, 各backbone的e2) → fusion2
  │
  ├─ encode3 → maxout → fusion3
  ├─ encode4 → maxout → fusion4
  ├─ bottleneck → maxout → fusion5
  │
  ├─ decode4(bottleneck后, skip4, indices4)
  ├─ decode3 → decode2 → decode1
  │
  └─ classifier → logits
```

### 设计动机

`Fuse_Last_Layer` 的缺点是只在最后融合。`Fuse_Unet` 的改进是：在编码的**每一个阶段**（encode1, encode2, encode3, encode4, bottleneck），都把主分支的中间特征和各 backbone 对应层的中间特征做 maxout 竞争。

### 融合的是什么？

每个编码层融合的是**同一空间分辨率下的特征图**：
- encode1 层：5 个 `[B,64,256,256]` 的 maxout
- encode2 层：5 个 `[B,64,128,128]` 的 maxout
- encode3 层：5 个 `[B,64,64,64]` 的 maxout
- encode4 层：5 个 `[B,64,32,32]` 的 maxout
- bottleneck 层：5 个 `[B,64,16,16]` 的 maxout

maxout 的具体操作：
```python
# 设 backbone0.e1 和 backbone1.e1 分别是各 backbone 在 encoder1 的输出
encoder_output1 = torch.unsqueeze(encoder_output1, 4)    # [B,64,H,W] → [B,64,H,W,1]
for idx in range(len(backbones)):
    encoder_output1 = torch.cat((encoder_output1,
                                  backbone_return[idx][1].unsqueeze(4)), dim=4)
# 结果: [B,64,H,W,5]   (5 = 主分支 + 4个backbone)

encoder_output1, _ = torch.max(encoder_output1, 4)      # → [B,64,H,W]
# maxout: 对每个像素位置 (h,w) 的每个通道 c，在 5 个候选值中取最大
```

这等价于问：**「在位置 (h,w) 上，5 个专家（主分支 + 4 个 backbone）谁对这个通道最自信？选最强的。」**

### 注意

v1/Fuse_Unet 的主分支只用模态 0 的 7 通道（而非全部 28 通道），这和后文的 v2/v3 不同。

---

## 6. FastSurferCNN_Fuse_Unet_v1 — 带辅助 backbone 加权输出

### 结构

```
输入 x [B, 28, 256, 256]

  主融合分支（只用模态0: x[:, 0:7]）
  ├─ encode1~4 + bottleneck（每层做 maxout 竞争 + fusion conv）
  ├─ decode4~1
  └─ classifier → logits_main

  辅助 backbone（backbone1~3，处理 x[:, 7:14], x[:, 14:21], x[:, 21:28]）
  ├─ backbone_1(x[:, 7:14])   → logits_1
  ├─ backbone_2(x[:, 14:21])  → logits_2
  └─ backbone_3(x[:, 21:28])  → logits_3

  最终输出：logits_main + logits_1 + logits_2 + logits_3
```

### 和 Fuse_Unet 的区别

| | Fuse_Unet | v1 |
|---|---|---|
| 主分支输入 | 模态0 (7ch) | 模态0 (7ch) |
| backbone 数量 | 全部 4 个 | 模态1~3 共 3 个（主分支自己就是模态0 的 backbone）|
| 最终输出 | 仅主分支 logits | 主分支 logits + 各 backbone logits 之和 |

### 设计动机

v1 的策略是「主分支做主力，backbone 做辅助」。最终输出的 logits 是主分支结果加上各 backbone 结果，形成一个「加权投票」——backbone 的独立判断直接加到最终投票里。

---

## 7. FastSurferCNN_Fuse_Unet_v2 — 模态拼接输入 + 竞争融合

### 结构

```
输入 x [B, 28, 256, 256]

  各 backbone（return_all，预训练并冻结）：
  ├─ backbone0(x[:, 0:7])    → returns[0] (18个中间值)
  ├─ backbone1(x[:, 7:14])   → returns[1]
  ├─ backbone2(x[:, 14:21])  → returns[2]
  └─ backbone3(x[:, 21:28])  → returns[3]

  主融合分支（★ 接收全部 28 通道）：
  ├─ encode1(x[:, :, :, :])   ← 全部 28 通道！
  │     → maxout(encode1_out, 各backbone.e1) → fusion1
  ├─ encode2 → maxout → fusion2
  ├─ encode3 → maxout → fusion3
  ├─ encode4 → maxout → fusion4
  ├─ bottleneck → maxout → fusion5
  ├─ decode4~1
  └─ classifier → logits_main

  最终：logits_main + backbone0_logits + backbone1_logits + ...
```

### v2 vs 之前的版本：关键差异

| | Fuse_Unet / v1 | v2 |
|---|---|---|
| 主分支输入 | 单模态 7 通道 | **全部模态 28 通道** |
| encode1 输入层 | `CompetitiveEncoderBlockInput(7→64)` | `CompetitiveEncoderBlockInput(28→64)` |

v2 是第一个让主融合分支**一开始就看到所有模态数据**的版本。之前的 v1 和 Fuse_Unet 的主分支只能看到模态 0，其他模态的信息只能通过 backbone 的特征间接注入。

### 设计动机

如果主分支只看模态 0，它在 encode1 层的判断就完全依赖 FA 数据。如果某个脑区在 FA 上没有明显特征但在 Trace 上很清楚，主分支就无法在第一层利用这个信息——只能等 backbone 在 maxout 时补救。

v2 让主分支直接「看到」所有模态的原始数据，每层再用 backbone 的特征做 maxout 补充。这相当于主分支既有全景视角（28 通道输入），又能参考各模态专家的独立意见（backbone 特征）。

---

## 8. FastSurferCNN_Fuse_Unet_v2_extended — v2 + 返回列表

### 结构和 v2 完全相同，唯一的区别：

```python
# v2 的 return：
return logits                      # 只返回融合后的 logits

# v2_extended 的 return：
return logits, logits_list         # 返回 (融合logits, [融合logits, backbone0_logits, ...])
```

### 为什么需要 extended？

推理脚本 `DDSurfer_Pred.py` 需要 `logits_list` 来做**多路概率累加**：

```python
temp, temp_list = model(images_batch)
for t_idx, temp in enumerate(temp_list):
    # 第0路: 主融合分支的输出 → 写入 pred_prob[0]
    # 第1路: backbone0 的输出 → 写入 pred_prob[1]
    # ...
    # 各路独立累加，最后三视角在 pred_prob[0] 上取 argmax
    prediction_probability[t_idx, ...] += temp * 视角权重
```

如果不返回 list，推理脚本就无法做多路概率累加。

---

## 9. FastSurferCNN_Fuse_Unet_v3 — ★ 核心改进：解码层融合同样做特征拼接

### 结构（只标注 v2 → v3 的改变）

```
输入 x [B, 28, 256, 256]

  前面和 v2 完全相同：
  ├─ 4 个 backbone return_all
  ├─ encode1~4 每层 maxout + fusion1~5
  └─ decode4~1 → decoder_output1

  ★ v3 新增：最终融合步骤
  ├─ 收集 decoder_output1（主分支） + backbone0.decoder_output1 + backbone1.decoder_output1 + ...
  │    共 (1 + 4) = 5 个 [B, 64, 256, 256] 的特征图
  │
  ├─ torch.cat(5个特征图, dim=1) → [B, 320, 256, 256]
  │
  ├─ fusion_layer: 1×1 Conv(320 → 64)
  │
  ├─ classifier: 1×1 Conv(64 → num_classes)
  │
  └─ logits = 主输出 + backbone0_logits + backbone1_logits + ...
```

### v3 相比 v2 的核心改进

在 `__init__` 中多了这一行：

```python
# v3 新增：融合各 backbone 的解码特征
self.fusion_layer = nn.Conv2d(
    params['num_filters'] * (params['num_modality'] + 1),  # 64 × 5 = 320
    params['num_filters'],                                   # → 64
    params['kernel_c'], params['stride_conv']
)
```

`forward()` 中多了这一段：

```python
decoder_outputs = []
decoder_outputs.append(decoder_output1)          # 主分支的解码输出
for ind_return in returns:
    decoder_outputs.append(ind_return[17])       # 各 backbone 的 decoder_output1

decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
logits = self.classifier.forward(decoder_outputs_fused)
```

### 为什么这是关键改进？

v2 的分类器只看到主融合分支的 decoder_output1，各 backbone 的 logits 只是事后加到主 logits 上。

v3 的分类器看到的输入是**拼接后的特征**：可以理解为，v3 在分类之前多了一步「让主分支和各 backbone 的解码层特征先融合再分类」，而不是各自独立分类后再加权求和。

打个比方：
- v2：5 个评委独立打分，最后取平均
- v3：5 个评委先开会讨论（`fusion_layer` 学习如何组合他们的意见），再给出一致打分

---

## 10. FastSurferCNN_Fuse_Unet_v3_extended — ★★★ DDParcel 推理实际使用的架构

### 结构和 v3 完全相同，唯一的区别：

```python
# v3 的 return：
return logits                        # logits = 主 + 各 backbone 之和

# v3_extended 的 return：
return logits, logits_list           # logits_list = [主融合结果, backbone0结果, backbone1结果, ...]
```

### 为什么是最终版本？

推理脚本在三个视角都需要 `logits_list` 来做：
- 第 0 个分支（主融合结果）写入 `pred_prob[0]`，三视角累加后用于最终 argmax
- 其他分支用于额外的概率累加槽位

v3_extended 是结构最完整 + 推理接口最方便的版本。

---

## 11. FastSurferCNN_Fuse_Unet_v4 / v4_extended — 进一步扩展

### v4：和 v3 结构相同，但只返回主 logits

```python
# v4 的 return：
return logits    # 不返回 list
```

### v4_extended：v4 + 额外的 logits_sum

```python
# v4_extended 的 return：
logits_list = [主融合logits, backbone0_logits, backbone1_logits, ..., logits_sum]
return logits, logits_list
```

其中 `logits_sum` 是所有 backbone logits 的总和：

```python
logits_sum = logits
for ind_return in returns:
    logits_sum += ind_return[0]
logits_list.append(logits_sum)
```

### 设计动机

v4_extended 的 logits_list 比 v3_extended 多了一个 `logits_sum`，推理脚本可以用它做额外的消融实验或加权策略。但在 DDParcel 的公开推理中，实际使用的是 v3_extended。

---

# 第三部分：融合机制深度解析

## 融合的是什么？逐层看

以 v3_extended 为例，整个网络中有 **三种不同类型的融合**：

### 融合类型 1：编码层 maxout 竞争（5 处：encode1~4 + bottleneck）

```python
# 以 encode1 层为例：
# 参与方：主分支编码器输出 + 4 个 backbone 的 encoder_output1
# 每个的尺寸：[B, 64, 256, 256]

encoder_output1 = torch.unsqueeze(主分支的 encode1 输出, 4)   # [B,64,256,256,1]
for idx in range(4):
    encoder_output1 = torch.cat(                                # 逐个堆叠
        (encoder_output1, backbone_returns[idx][1].unsqueeze(4)), dim=4)
# → [B, 64, 256, 256, 5]   (5 = 主分支 + 4 backbone)

encoder_output1, _ = torch.max(encoder_output1, 4)            # → [B,64,256,256]
encoder_output1 = self.fusion1(encoder_output1)                # 1×1 conv 精炼
```

**「融合」的是：5 个特征图上每个像素位置 (h,w) 在每个通道 c 上的响应值。** maxout 选择最大值，相当于「谁对这个特征最自信就用谁的」。

注意 maxout 后的 `fusion1`（1×1 Conv）也很重要——它不是简单的取 max 就完了，而是用可学习的卷积进一步混合 maxout 后的特征。

### 融合类型 2：解码层特征拼接（1 处：fusion_layer）

```python
# 参与方：主分支的 decoder_output1 + 4 个 backbone 的 decoder_output1
# 每个的尺寸：[B, 64, 256, 256]
# 共 5 个

decoder_outputs = [主分支dec1, backbone0_dec1, backbone1_dec1, backbone2_dec1, backbone3_dec1]
fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
# torch.cat → [B, 320, 256, 256]
# fusion_layer(1×1 Conv: 320 → 64) → [B, 64, 256, 256]
```

**「融合」的是：各分支解码器最终输出的 64 通道特征图。** 用 1×1 卷积学习如何组合这些特征。这不是 maxout，是通道降维+信息混合——5 个 64 通道压缩成 1 个 64 通道。

为什么这里用 concat+conv 而不是 maxout？因为解码层特征已经非常接近最终输出了，这时候需要的是「综合考虑所有专家的意见」而非「选最强的」。concat+1×1 conv 可以实现加权混合，而 maxout 只能选一个。

### 融合类型 3：logits 层累加

```python
logits = 主分类器的输出
for ind_return in returns:
    logits += ind_return[0]    # 累加每个 backbone 的独立 logits
```

**「融合」的是：各分支的最终分类预测。** 在 logits 空间中直接相加，等价于「每个专家投票权重相同」。之后 softmax 会把累加后的 logits 转成概率。

---

## 融合演变路线图

```
Fuse_Last_Layer:
  只在解码后做 1 次特征拼接
  「分头行动，最后碰面」
      │
      ▼
Fuse_Unet / v1:
  编码层 5 处 maxout 竞争
  但主分支只看单模态
  「每层交流，但队长只有一个信息来源」
      │
      ▼
v2:
  编码层 5 处 maxout + 主分支看全部模态
  「队长也有全景视角了」
      │
      ▼
v3 / v3_extended ★:
  编码层 5 处 maxout + 解码层 1 次特征拼接 + logits 累加
  「前期每层竞赛，最终开会表决，各人独立意见也计入」
```

---

## 为什么需要 11 个网络类？

设计演进的真实原因：

| 类 | 设计目的 |
|-----|---------|
| FastSurferCNN | 基础 U-Net，单模态训练和测试的基准 |
| return_all | 作为 fusion 网络的 backbone，需要暴露中间特征 |
| no_classifer | Fuse_Last_Layer 中各模态共用分类器，所以 backbone 不需要自己的分类器 |
| Fuse_Last_Layer | 最早的融合实验：验证多模态是否有帮助（结果：有帮助，但不够） |
| Fuse_Unet | 验证深层 maxout 竞争是否更好（结果：比 Last_Layer 好） |
| v1 | 验证 backbone 独立输出加权是否有帮助 |
| v2 | 验证主分支接收全部模态是否有提升（结果：有提升） |
| v2_extended | 满足推理接口需要（返回 logits_list） |
| v3 | 验证解码层加特征拼接融合是否有提升（结果：有提升，这就是核心贡献） |
| v3_extended | **最终版本**：解码层拼接融合 + 推理接口完整 |
| v4/v4_extended | 额外实验变体，供消融分析 |

---

## 复习：一张图看懂 v3_extended 全部融合位置

```
输入 28ch [B,28,256,256]
     │
     ├─ 切分 ──→ backbone0(7ch) ──→ e1,e2,e3,e4,bn,dec1  (FA)
     │         backbone1(7ch) ──→ e1,e2,e3,e4,bn,dec1  (Trace)
     │         backbone2(7ch) ──→ e1,e2,e3,e4,bn,dec1  (MinEig)
     │         backbone3(7ch) ──→ e1,e2,e3,e4,bn,dec1  (MidEig)
     │
     └─ 全 28ch ──→ 主融合分支 ──────────────────────────┐
                                                          │
  encode1_out ──╋── b0.e1 ╋ b1.e1 ╋ b2.e1 ╋ b3.e1      │  ← 融合类型1: maxout
       │         └── maxout ──→ fusion1 ──→ encode2      │
       │                                                 │
  encode2_out ──╋── b0.e2 ╋ b1.e2 ╋ b2.e2 ╋ b3.e2      │  ← 融合类型1: maxout
       │         └── maxout ──→ fusion2 ──→ encode3      │
       │                                                 │
  encode3_out ──╋── b0.e3 ╋ b1.e3 ╋ b2.e3 ╋ b3.e3      │  ← 融合类型1: maxout
       │         └── maxout ──→ fusion3 ──→ encode4      │
       │                                                 │
  encode4_out ──╋── b0.e4 ╋ b1.e4 ╋ b2.e4 ╋ b3.e4      │  ← 融合类型1: maxout
       │         └── maxout ──→ fusion4 ──→ bottleneck   │
       │                                                 │
  bottleneck ───╋── b0.bn ╋ b1.bn ╋ b2.bn ╋ b3.bn      │  ← 融合类型1: maxout
       │         └── maxout ──→ fusion5 ──→ decode4..1   │
       │                                                 │
  decoder1 ─────╋── b0.dec1 ╋ b1.dec1 ╋ b2.dec1 ╋ b3.dec1  ← 融合类型2: concat+conv
       │         └── concat(dim=1) ──→ fusion_layer ──→ classifier
       │                                                 │
  logits_main   +  logits_b0 + logits_b1 + logits_b2 + logits_b3  ← 融合类型3: sum
       │
  最终 logits
```
